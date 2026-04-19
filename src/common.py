import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ARTICLE_RE = re.compile(r"Article\s+(\d+)\s*:\s*")
UNK = "<UNK>"
PAD = "<PAD>"


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def parse_articles(path: str) -> List[List[str]]:
    text = read_text(path)
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [simple_tokenize(text)]

    docs: List[List[str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        docs.append(simple_tokenize(text[start:end]))
    return docs


def simple_tokenize(text: str) -> List[str]:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split(" ")


def build_vocab(docs: List[List[str]], max_vocab: int = 10000) -> Tuple[Dict[str, int], List[str], Counter]:
    freq = Counter()
    for d in docs:
        freq.update(d)

    most = [w for w, _ in freq.most_common(max_vocab - 2)]
    idx2word = [PAD, UNK] + most
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word, freq


def map_to_ids(tokens: List[str], word2idx: Dict[str, int]) -> List[int]:
    unk = word2idx[UNK]
    return [word2idx.get(t, unk) for t in tokens]


def load_metadata(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def classify_topic_from_title(title: str) -> str:
    t = title.lower()
    rules = {
        "Politics": ["election", "government", "minister", "parliament", "pti", "hukumat"],
        "Sports": ["cricket", "match", "team", "player", "score", "cup"],
        "Economy": ["inflation", "trade", "bank", "gdp", "budget", "economy"],
        "International": ["un", "treaty", "foreign", "bilateral", "conflict", "india", "us"],
        "Health & Society": ["hospital", "disease", "vaccine", "flood", "education", "sehat", "taleem"],
    }
    for label, kws in rules.items():
        if any(k in t for k in kws):
            return label
    return "Politics"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
