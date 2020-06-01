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
import shutil
from time import sleep
from tqdm import tqdm as tqdm_base


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def eval(tokenizer: Tokenizer, model: GPT2LMHeadModel, dataset: List[str], batch_size: int, block_size: int, n_batch=1):
    model.eval()
    loss = 0
    iter_count = min(len(dataset) // batch_size, n_batch)
    data = dataset[:]
    random.shuffle(data)
    for i in tqdm(range(iter_count), desc="eval"):
        batch = tokenizer.encode(data[i*batch_size:(i+1)*batch_size], block_size).cuda()
        mask = tokenizer.mask(batch).cuda()
        with torch.no_grad():
            loss += model(batch, attention_mask=mask, labels=batch)[0].item()
        sleep(0.01)
    model.train()
    return loss / iter_count


def get_corpus(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        dataset = f.readlines()
    return dataset


def train(tokenizer: Tokenizer, model: GPT2LMHeadModel, args: TrainingArguments, writer: SummaryWriter, logger):
    dataset = get_corpus(args.corpus_path)
    n = int(len(dataset) * 0.9)
    train_dataset, test_dataset = dataset[:n], dataset[n:]
    iterator = build_data_iterator(tokenizer, train_dataset, args.eval_batch_size, args.block_size)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(iterator) // args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    i = 0
    try:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    except PermissionError:
        pass
    os.mkdir(args.output_dir)
    for _ in range(args.num_train_epochs):
        for ids, attention_mask in tqdm(iterator, desc='train'):
            ids = ids.cuda()
            loss = model(ids, attention_mask=attention_mask.cuda(), labels=ids)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            writer.add_scalar('Loss/train', loss.item(), i)
            if args.evaluate_during_training and i % args.logging_steps == 0 and i > 0:
                logger.info(f"epoch: {i / len(iterator)}")
                logger.info(f"train loss: {loss.item()}")
                logger.info(f"lr: {scheduler.get_last_lr()}")
                eval_loss = eval(tokenizer, model, test_dataset, args.eval_batch_size, args.block_size, args.n_eval_batch)
                logger.info(f"eval loss: {eval_loss}")
                writer.add_scalar('Loss/eval', eval_loss, i)
            if i % args.save_steps == 0 and i > 0:
                dir = args.output_dir + "/" + f"iter{i}"
                os.mkdir(dir)
                model.save_pretrained(dir)
            i += 1
            sleep(0.01)
    eval_loss = eval(tokenizer, model, test_dataset, args.eval_batch_size, args.block_size, args.n_eval_batch)
    logger.info(f"eval loss: {eval_loss}")
    model.save_pretrained(args.output_dir)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger("rugpt2")
    args = TrainingArguments()
    writer = SummaryWriter()
    tokenizer = Tokenizer("tokenizer.model")
    config = GPT2Config(vocab_size=tokenizer.vocab_size)
    model = GPT2LMHeadModel(config).cuda()
    train(tokenizer, model, args, writer, logger)
# tensorboard --logdir=runs
