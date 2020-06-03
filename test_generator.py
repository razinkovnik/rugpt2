from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tokenizer import Tokenizer
from training_arguments import TrainingArguments
import torch
import argparse

if __name__ == "__main__":
    train_args = TrainingArguments()
    parser = argparse.ArgumentParser(description='Проверка генерации')
    parser.add_argument('--output_dir', type=str, help='место для модели')
    parser.add_argument('--start', type=str, default='Я', help='начало строки')
    parser.add_argument('--length', type=int, default=20, help='размер генерируемого текста')
    parser.add_argument('--num_beams', type=int)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--no_repeat_ngram_size', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--top_p', type=float)
    args = parser.parse_args()
    model = GPT2LMHeadModel.from_pretrained(args.output_dir or train_args.output_dir)
    tokenizer = Tokenizer(train_args.tokenizer_path)
    model.eval()

    input_ids = tokenizer.encode([args.start])
    if args.top_k:
        if args.top_p:
            outputs = model.generate(
                input_ids,
                do_sample=True,
                max_length=args.length,
                top_k=args.top_k,
                top_p=args.top_p,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                num_return_sequences=args.num_return_sequences
            )
        else:
            outputs = model.generate(
                input_ids,
                do_sample=True,
                max_length=args.length,
                top_k=args.top_k,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                num_return_sequences=args.num_return_sequences
            )
    elif args.top_p:
        outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=args.length,
            top_p=args.top_p,
            top_k=0,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_return_sequences=args.num_return_sequences
        )
    elif args.temperature:
        outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=args.length,
            top_k=0,
            temperature=args.temperature,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_return_sequences=args.num_return_sequences
        )
    else:
        outputs = model.generate(
            input_ids,
            max_length=args.length,
            num_beams=args.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_return_sequences=args.num_return_sequences
        )

    for i, sample_output in enumerate(outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output)[0].split('<EOS>')[0]))

