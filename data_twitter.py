from corus import load_mokoron
from training_arguments import TrainingArguments
import re


# https://github.com/natasha/corus


def clean(tweet: str):
    tweet: str = record.text
    tweet = tweet.replace("\\n", " ")
    tweet = tweet.replace("\\", "")
    tweet = tweet.replace("RT", "")
    tweet = re.sub("http[s]*[^\s]*", "", tweet)
    tweet = re.sub("#[\w:]+", "", tweet)
    tweet = re.sub("&[\w]+;", "", tweet)
    tweet = re.sub("@[\w:]+", "", tweet)
    # tweet = re.sub("[^А-Яа-я.,();:!\-?\s]", "", tweet)
    tweet = re.sub(" +", " ", tweet).strip()
    return tweet


if __name__ == "__main__":
    train_args = TrainingArguments()
    path = 'corpus/db.sql'
    records = load_mokoron(path)
    with open(train_args.corpus_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            tweet = clean(record.text)
            if len(tweet) > 20:
                f.write(tweet + "\n")


