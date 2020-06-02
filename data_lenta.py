import pandas as pd
import re
from training_arguments import TrainingArguments


# https://github.com/natasha/corus


if __name__ == "__main__":
    train_args = TrainingArguments()
    df = pd.read_csv("corpus/lenta-ru-news.csv")
    corpus = df.text.to_list()
    corpus = [txt + "\n" for txt in corpus if type(txt) is str]

    with open(train_args.corpus_path, "w", encoding="utf-8") as f:
        f.writelines(corpus)


