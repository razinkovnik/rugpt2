import pandas as pd
import re
from training_arguments import TrainingArguments


def get_text_by_id(textid: str) -> str:
    with open(f"corpus/texts/{textid}.txt", encoding="utf-8") as f:
        text = f.read()
    text = re.sub("\s+", " ", text)
    return ".".join(text.split('.')[:-1]) + ".\n"


if __name__ == "__main__":
    train_args = TrainingArguments()
    df = pd.read_csv("corpus/newmetadata.csv", encoding="utf-8", sep='\t', comment='#')
    corpus = [get_text_by_id(textid) for textid in df.textid]

    with open(train_args.corpus_path, "w", encoding="utf-8") as f:
        f.writelines(corpus)


