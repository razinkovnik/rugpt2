import youtokentome as yttm
import torch
from typing import List
from training_arguments import TrainingArguments


class Tokenizer:
    def __init__(self, tokenizer_model_path):
        self.tokenizer = yttm.BPE(tokenizer_model_path)

    @staticmethod
    def mask(input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(input_ids)
        mask[input_ids == 0] = 0
        return mask

    def decode(self, out_ids: torch.Tensor) -> str:
        ids = out_ids.squeeze().tolist()
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    def encode(self, batch: List[str], max_len=0):
        ids = self.tokenizer.encode(batch)
        if max_len:
            ids = [input_ids[:max_len - 2] for input_ids in ids]
            ids = [[2] + input_ids + [3] for input_ids in ids]
            ids = [input_ids + [0]*(max_len-len(input_ids)) for input_ids in ids]
        return torch.tensor(ids)


def train(train_data_path: str, model_path: str, vocab_size: int):
    yttm.BPE.train(data=train_data_path, vocab_size=vocab_size, model=model_path)


if __name__ == "__main__":
    args = TrainingArguments()
    train(args.corpus_path, args.tokenizer_path, args.vocab_size)
