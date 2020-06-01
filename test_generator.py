from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tokenizer import Tokenizer
from training_arguments import TrainingArguments
import torch
import argparse

if __name__ == "__main__":
    train_args = TrainingArguments()
    parser = argparse.ArgumentParser(description='Проверка генерации')
    parser.add_argument('--output_dir', type=str, help='место для модели')
    parser.add_argument('--start', type=str, help='начало строки')
    parser.add_argument('--length', type=int, help='размер генерируемого текста')
    args = parser.parse_args()
    model = GPT2LMHeadModel.from_pretrained(args.output_dir or train_args.output_dir)
    tokenizer = Tokenizer(train_args.tokenizer_path)
    model.eval()

    print("greedy_output")
    input_ids = tokenizer.encode([args.start])
    greedy_output = model.generate(input_ids, max_length=args.length)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0]))

    print("\n\nactivate beam search and early_stopping")
    beam_output = model.generate(
        input_ids,
        max_length=args.length,
        num_beams=5,
        early_stopping=True
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(beam_output[0]))

    print("\n\nset no_repeat_ngram_size to 2")
    beam_output = model.generate(
        input_ids,
        max_length=args.length,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(beam_output[0]))

    print("\n\nset return_num_sequences > 1")
    beam_outputs = model.generate(
        input_ids,
        max_length=args.length,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True
    )

    print("\n\nnow we have 3 output sequences")
    print("Output:\n" + 100 * '-')
    for i, beam_output in enumerate(beam_outputs):
        print("{}: {}".format(i, tokenizer.decode(beam_output)))

    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=args.length,
        top_k=0
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0]))

    print("\n\nuse temperature to decrease the sensitivity to low probability candidates")
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=args.length,
        top_k=0,
        temperature=0.7
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0]))

    print("set top_k to 50")
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=args.length,
        top_k=50
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0]))

    print("deactivate top_k sampling and sample only from 92% most likely words")
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=args.length,
        top_p=0.92,
        top_k=0
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0]))

    print("\n\nset top_k = 50 and set top_p = 0.95 and num_return_sequences = 3")
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=args.length,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3
    )

    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output)))
