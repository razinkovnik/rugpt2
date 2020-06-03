from corus import load_wiki
from training_arguments import TrainingArguments
import re
import itertools


# https://github.com/natasha/corus


def clean(body: str):
    text = body.split('\n')[2:]
    text = " ".join(text)
    text = re.sub("\([^)]*\)", "", text)
    text = re.sub("<[^)]*>", "", text)
    text = text.split('= Издания =')[0]
    text = text.split('= Примечания =')[0]
    text = text.split('= Литература =')[0]
    text = text.split('= Ссылки =')[0]
    text = re.sub("=", "\n", text)
    text = re.sub("\s+", " ", text)
    return text


if __name__ == "__main__":
    start, end = 0, 10000
    train_args = TrainingArguments()
    path = 'corpus/ruwiki-latest-pages-articles.xml.bz2'
    records = load_wiki(path)
    with open(train_args.corpus_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            if i < start:
                continue
            text = clean(record.text)
            if len(text) > 20:
                f.write(text + "\n")
            if i > end:
                break
