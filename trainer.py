from transformers import GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup
from tokenizer import Tokenizer
from dataset import build_data_iterator, MyDataset
import torch
from typing import List
import random
from torch.utils.tensorboard import SummaryWriter
from training_arguments import TrainingArguments
import logging
import os
from utils import tqdm
import argparse


def eval(tokenizer: Tokenizer, model: GPT2LMHeadModel, dataset: MyDataset, args: TrainingArguments):
    model.eval()
    loss = 0
    iterator = build_data_iterator(tokenizer, dataset, args.eval_batch_size, args.block_size, random_sampler=True)
    n = min(args.n_eval_batch, len(dataset))
    for _ in tqdm(range(n), desc="eval"):
        ids, attention_mask = next(iter(iterator))
        ids = ids.to(args.device)
        with torch.no_grad():
            loss += model(ids, attention_mask=attention_mask.to(args.device), labels=ids)[0].item()
    model.train()
    return loss / n


def get_corpus(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        dataset = f.readlines()
    return dataset


def train(tokenizer: Tokenizer, model: GPT2LMHeadModel, args: TrainingArguments, writer: SummaryWriter, logger):
    dataset = get_corpus(args.corpus_path)
    n = int(len(dataset) * 0.9)
    train_dataset, test_dataset = dataset[:n], dataset[n:]
    train_dataset = MyDataset(train_dataset, tokenizer, args.block_size)
    test_dataset = MyDataset(test_dataset, tokenizer, args.block_size)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataset) // args.train_batch_size * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_training_steps)
    i = 0
    try:
        os.mkdir(args.output_dir)
    except FileExistsError:
        pass
    prev_loss = eval(tokenizer, model, test_dataset, args) if args.load else 1000
    no_save_counter = 0
    for _ in range(args.num_train_epochs):
        iterator = build_data_iterator(tokenizer, train_dataset, args.train_batch_size, args.block_size)
        for ids, attention_mask in tqdm(iterator, desc='train'):
            ids = ids.to(args.device)
            loss = model(ids, attention_mask=attention_mask.to(args.device), labels=ids)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if args.evaluate_during_training and i % args.logging_steps == 0 and i > 0:
                logger.info(f"epoch: {i / len(iterator)}")
                logger.info(f"train loss: {loss.item()}")
                lr = scheduler.get_last_lr()[0]
                logger.info(f"lr: {lr}")
                eval_loss = eval(tokenizer, model, test_dataset, args)
                logger.info(f"eval loss: {eval_loss}")
                writer.add_scalar('Loss/train', loss.item(), i)
                writer.add_scalar('Loss/eval', eval_loss, i)
                writer.add_scalar('LR', lr, i)
                if prev_loss > eval_loss:
                    prev_loss = eval_loss
                    model.save_pretrained(args.output_dir)
                    no_save_counter = 0
                else:
                    no_save_counter += 1
                    if no_save_counter > args.no_save_count:
                        logger.info(f"модель не сохранялась {no_save_counter} раз подряд")
                        return
            if not args.evaluate_during_training and i % args.save_steps == 0 and i > 0:
                dir = args.output_dir + "/" + f"iter{i}"
                os.mkdir(dir)
                model.save_pretrained(dir)
            i += 1
    eval_loss = eval(tokenizer, model, test_dataset, args)
    logger.info(f"eval loss: {eval_loss}")
    if prev_loss > eval_loss:
        model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train_args = TrainingArguments()
    parser = argparse.ArgumentParser(description='Тренировка модели')
    parser.add_argument('--load', help='загрузить модель', action='store_true')
    parser.add_argument('--eval', help='только проверить', action='store_true')
    parser.add_argument('--output_dir', type=str, default="model", help='место для модели')
    parser.add_argument('--log_dir', type=str, default="runs", help='логи')
    parser.add_argument('--train_batch_size', type=int, default=8, help='размер тренировочного батча')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='размер тестового батча')
    parser.add_argument('--block_size', type=int, default=512, help='размер блока текста')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='количество эпох')
    parser.add_argument('--logging_steps', type=int, default=100, help='шаг проверки и информирования')
    parser.add_argument('--save_steps', type=int, default=500, help='шаг сохранения')
    parser.add_argument('--no_save_count', type=int, default=5)
    args = parser.parse_args()
    train_args.output_dir = args.output_dir
    train_args.train_batch_size = args.train_batch_size
    train_args.eval_batch_size = args.eval_batch_size
    train_args.block_size = args.block_size
    train_args.num_train_epochs = args.num_train_epochs
    train_args.logging_steps = args.logging_steps
    train_args.save_steps = args.save_steps
    train_args.no_save_count = args.no_save_count
    train_args.load = args.load
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger("rugpt2")
    writer = SummaryWriter(log_dir=args.log_dir)
    tokenizer = Tokenizer(train_args.tokenizer_path)
    config = GPT2Config(vocab_size=tokenizer.vocab_size, bos_token_id=2, eos_token_id=3, n_positions=128, n_ctx=128,
                        n_embd=768, n_layer=6, n_head=6)
    assert config.n_embd % config.n_head == 0
    model = (GPT2LMHeadModel.from_pretrained(train_args.output_dir) if args.load else GPT2LMHeadModel(config)).to(train_args.device)
    if args.eval:
        dataset = get_corpus(train_args.corpus_path)
        n = int(len(dataset) * 0.9)
        test_dataset = MyDataset(dataset[n:], tokenizer, train_args.block_size)
        print(f"eval loss: {eval(tokenizer, model, test_dataset, train_args)}")
    else:
        train(tokenizer, model, train_args, writer, logger)
    if train_args.device == "cuda":
        torch.cuda.empty_cache()
# tensorboard --logdir=runs
