import argparse
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from common import classify_topic_from_title, load_metadata, map_to_ids, parse_articles, save_json


TOPIC2ID = {
    "Politics": 0,
    "Sports": 1,
    "Economy": 2,
    "International": 3,
    "Health & Society": 4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned", default="cleaned.txt")
    ap.add_argument("--metadata", default="Metadata.json")
    ap.add_argument("--word2idx", default="embeddings/word2idx.json")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", default="data/topic_cls.npz")
    args = ap.parse_args()

    docs = parse_articles(args.cleaned)
    meta = load_metadata(args.metadata)
    word2idx = load_json(args.word2idx)

    X, y = [], []
    for i, d in enumerate(docs):
        title = meta[i].get("title", "") if i < len(meta) else ""
        topic = classify_topic_from_title(title)
        ids = map_to_ids(d, word2idx)[: args.max_len]
        if len(ids) < args.max_len:
            ids += [0] * (args.max_len - len(ids))
        X.append(ids)
        y.append(TOPIC2ID[topic])

    X = np.asarray(X, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)

    x_train, x_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    np.savez(args.out, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    save_json("data/topic_distribution.json", {
        "train": dict(Counter(y_train.tolist())),
        "val": dict(Counter(y_val.tolist())),
        "test": dict(Counter(y_test.tolist())),
    })


def load_json(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
