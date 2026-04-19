import argparse
import json

import numpy as np

from common import parse_articles
from part1_skipgram import train_skipgram


def cosine_neighbors(emb, word2idx, query, k=10):
    idx2word = {i: w for w, i in word2idx.items()}
    if query not in word2idx:
        return []
    x = emb[word2idx[query]]
    sims = (emb @ x) / (np.linalg.norm(emb, axis=1) * (np.linalg.norm(x) + 1e-12) + 1e-12)
    order = np.argsort(-sims)
    out = []
    for i in order:
        w = idx2word[int(i)]
        if w == query:
            continue
        out.append((w, float(sims[i])))
        if len(out) == k:
            break
    return out


def analogy(emb, word2idx, a, b, c, topk=3):
    idx2word = {i: w for w, i in word2idx.items()}
    if any(w not in word2idx for w in [a, b, c]):
        return []
    v = emb[word2idx[b]] - emb[word2idx[a]] + emb[word2idx[c]]
    sims = (emb @ v) / (np.linalg.norm(emb, axis=1) * (np.linalg.norm(v) + 1e-12) + 1e-12)
    order = np.argsort(-sims)
    out = []
    ban = {a, b, c}
    for i in order:
        w = idx2word[int(i)]
        if w in ban:
            continue
        out.append((w, float(sims[i])))
        if len(out) == topk:
            break
    return out


def mrr_score(emb, word2idx, labeled_pairs):
    rr = []
    idx2word = {i: w for w, i in word2idx.items()}
    norms = np.linalg.norm(emb, axis=1) + 1e-12
    for q, target in labeled_pairs:
        if q not in word2idx or target not in word2idx:
            rr.append(0.0)
            continue
        x = emb[word2idx[q]]
        sims = (emb @ x) / (norms * (np.linalg.norm(x) + 1e-12))
        order = np.argsort(-sims)
        rank = 0
        for j, i in enumerate(order, start=1):
            if idx2word[int(i)] == q:
                continue
            rank += 1
            if idx2word[int(i)] == target:
                rr.append(1.0 / rank)
                break
        else:
            rr.append(0.0)
    return float(np.mean(rr))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_condition(name, corpus, dim, out_dir):
    docs = parse_articles(corpus)
    emb, word2idx, _, _ = train_skipgram(docs, dim=dim, out_dir=out_dir, tag=name)
    return emb, word2idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned", default="cleaned.txt")
    ap.add_argument("--raw", default="raw.txt")
    ap.add_argument("--out_dir", default="embeddings")
    args = ap.parse_args()

    with open(f"{args.out_dir}/word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    emb = np.load(f"{args.out_dir}/embeddings_w2v.npy")

    queries = ["Pakistan", "Hukumat", "Adalat", "Maeeshat", "Fauj", "Sehat", "Taleem", "Aabadi"]
    nearest = {q: cosine_neighbors(emb, word2idx, q, 10) for q in queries}

    analogies = [
        ("Lahore", "Punjab", "Karachi"),
        ("Hukumat", "Wazir", "Adalat"),
        ("Cricket", "Player", "Team"),
        ("Doctor", "Hospital", "Teacher"),
        ("Pakistan", "Islamabad", "India"),
        ("TalibIlm", "Taleem", "Mareez"),
        ("Fauj", "Sipahi", "Adalat"),
        ("Bank", "Maeeshat", "Hospital"),
        ("Wazir", "Siyasat", "Khilari"),
        ("Vaccine", "Sehat", "Budget"),
    ]
    analogy_res = {f"{a}:{b}::{c}:?": analogy(emb, word2idx, a, b, c, 3) for a, b, c in analogies}

    labeled_pairs = [
        ("Pakistan", "Lahore"), ("Karachi", "Pakistan"), ("Cricket", "Match"), ("Team", "Player"),
        ("Hukumat", "Wazir"), ("Adalat", "Qanoon"), ("Bank", "Maeeshat"), ("Hospital", "Sehat"),
        ("Taleem", "School"), ("Aabadi", "Mulk"), ("GDP", "Budget"), ("Trade", "Maeeshat"),
        ("UN", "Treaty"), ("Foreign", "Bilateral"), ("Disease", "Vaccine"), ("Flood", "Relief"),
        ("Election", "Parliament"), ("Minister", "Government"), ("Fauj", "Sipahi"), ("Score", "Match"),
    ]

    # C1: PPMI baseline
    ppmi = np.load(f"{args.out_dir}/ppmi_matrix.npy")
    c1_mrr = mrr_score(ppmi, word2idx, labeled_pairs)

    # C2-C4
    c2_emb, c2_w = run_condition("raw", args.raw, 100, args.out_dir)
    c3_emb, c3_w = run_condition("cleaned", args.cleaned, 100, args.out_dir)
    c4_emb, c4_w = run_condition("cleaned_d200", args.cleaned, 200, args.out_dir)

    res = {
        "nearest_neighbors": nearest,
        "analogies": analogy_res,
        "conditions": {
            "C1_PPMI": {
                "mrr": c1_mrr,
                "neighbors": {q: cosine_neighbors(ppmi, word2idx, q, 5) for q in queries[:5]},
            },
            "C2_skipgram_raw": {
                "mrr": mrr_score(c2_emb, c2_w, labeled_pairs),
                "neighbors": {q: cosine_neighbors(c2_emb, c2_w, q, 5) for q in queries[:5]},
            },
            "C3_skipgram_cleaned": {
                "mrr": mrr_score(c3_emb, c3_w, labeled_pairs),
                "neighbors": {q: cosine_neighbors(c3_emb, c3_w, q, 5) for q in queries[:5]},
            },
            "C4_skipgram_cleaned_d200": {
                "mrr": mrr_score(c4_emb, c4_w, labeled_pairs),
                "neighbors": {q: cosine_neighbors(c4_emb, c4_w, q, 5) for q in queries[:5]},
            },
        },
    }

    with open(f"{args.out_dir}/part1_eval.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
