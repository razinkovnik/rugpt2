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
    path = 'corpus/ruwiki-latest-pages-articles.xml.bz2'
    records = load_wiki(path)
    lines = []
    for i, record in enumerate(records):
        text = clean(record.text + "\n")
        lines.append(text)
        if i % 50000 == 0 and i > 0:
            with open(f'corpus/data/corpus_{i}.txt', "w", encoding="utf-8") as f:
                f.writelines(lines)
                print("iter:", i)
                lines = []
