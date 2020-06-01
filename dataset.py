from collections import namedtuple
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import List
from tokenizer import Tokenizer

Batch = namedtuple(
    "Batch", ["ids", "attention_mask"]
)


class MyDataset(Dataset):
    def __init__(self, data: List[str]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate(data: List[str], tokenizer: Tokenizer, block_size: int) -> Batch:
    ids = tokenizer.encode(data, block_size)
    mask = tokenizer.mask(ids)
    return Batch(ids=ids, attention_mask=mask)


def build_data_iterator(tokenizer, dataset, batch_size, block_size) -> DataLoader:
    sampler = SequentialSampler(dataset)
    iterator = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, collate_fn=lambda data: collate(data, tokenizer, block_size),
    )
    return iterator


if __name__ == "__main__":
    tokenizer = Tokenizer("tokenizer.model")
    with open("corpus.txt", encoding="utf-8") as f:
        dataset = f.readlines()
    iterator = build_data_iterator(tokenizer, dataset, 8, 32)
    print(next(iter(iterator)))
