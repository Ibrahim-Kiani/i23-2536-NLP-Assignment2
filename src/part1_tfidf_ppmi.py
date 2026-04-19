import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from common import build_vocab, classify_topic_from_title, ensure_dir, load_metadata, map_to_ids, parse_articles, save_json


def tfidf_matrix(docs, word2idx):
    n_docs = len(docs)
    v = len(word2idx)
    tf = np.zeros((n_docs, v), dtype=np.float32)
    df = np.zeros(v, dtype=np.int32)

    for i, d in enumerate(docs):
        ids = map_to_ids(d, word2idx)
        if not ids:
            continue
        counts = np.bincount(ids, minlength=v).astype(np.float32)
        tf[i] = counts / max(1.0, counts.sum())
        df += (counts > 0).astype(np.int32)

    idf = np.log(n_docs / (1 + df))
    return tf * idf[None, :]


def top_words_per_topic(tfidf, docs, word2idx, metadata):
    idx2word = {i: w for w, i in word2idx.items()}
    topic_docs = defaultdict(list)
    for i, m in enumerate(metadata[: len(docs)]):
        topic_docs[classify_topic_from_title(m.get("title", ""))].append(i)

    out = {}
    for topic, ids in topic_docs.items():
        if not ids:
            continue
        mean_scores = tfidf[ids].mean(axis=0)
        top = np.argsort(-mean_scores)[:10]
        out[topic] = [idx2word[int(t)] for t in top]
    return out


def ppmi_matrix(docs, word2idx, window=5):
    v = len(word2idx)
    cooc = np.zeros((v, v), dtype=np.float32)
    for d in docs:
        ids = map_to_ids(d, word2idx)
        for i, c in enumerate(ids):
            l = max(0, i - window)
            r = min(len(ids), i + window + 1)
            for j in range(l, r):
                if i == j:
                    continue
                o = ids[j]
                cooc[c, o] += 1
                cooc[o, c] += 1

    total = cooc.sum()
    p_w = cooc.sum(axis=1, keepdims=True) / max(total, 1.0)
    p_c = cooc.sum(axis=0, keepdims=True) / max(total, 1.0)
    p_wc = cooc / max(total, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2((p_wc + 1e-12) / (p_w @ p_c + 1e-12))
    ppmi = np.maximum(0.0, pmi)
    return ppmi


def token_category(token):
    groups = {
        "politics": {"hukumat", "wazir", "parliament", "election", "adalat"},
        "sports": {"cricket", "match", "team", "player", "score"},
        "geography": {"pakistan", "lahore", "karachi", "islamabad", "punjab"},
    }
    t = token.lower()
    for g, s in groups.items():
        if t in s:
            return g
    return "other"


def plot_tsne(ppmi, idx2word, out_path):
    top_n = min(200, ppmi.shape[0])
    vecs = ppmi[:top_n]
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=20)
    z = tsne.fit_transform(vecs)

    labels = [token_category(idx2word[i]) for i in range(top_n)]
    colors = {"politics": "red", "sports": "green", "geography": "blue", "other": "gray"}

    plt.figure(figsize=(10, 7))
    for c in colors:
        mask = [l == c for l in labels]
        plt.scatter(z[mask, 0], z[mask, 1], s=20, label=c, c=colors[c], alpha=0.7)
    plt.title("t-SNE of top 200 tokens (PPMI)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def cosine_neighbors(matrix, idx2word, word2idx, query, k=5):
    if query not in word2idx:
        return []
    x = matrix[word2idx[query]]
    denom = (np.linalg.norm(matrix, axis=1) * (np.linalg.norm(x) + 1e-12) + 1e-12)
    sims = (matrix @ x) / denom
    order = np.argsort(-sims)
    ans = []
    for i in order:
        w = idx2word[i]
        if w == query:
            continue
        ans.append((w, float(sims[i])))
        if len(ans) == k:
            break
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned", default="cleaned.txt")
    ap.add_argument("--metadata", default="Metadata.json")
    ap.add_argument("--max_vocab", type=int, default=10000)
    ap.add_argument("--out_dir", default="embeddings")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    docs = parse_articles(args.cleaned)
    meta = load_metadata(args.metadata)
    word2idx, idx2word, _ = build_vocab(docs, args.max_vocab)

    tfidf = tfidf_matrix(docs, word2idx)
    np.save(f"{args.out_dir}/tfidf_matrix.npy", tfidf)

    top = top_words_per_topic(tfidf, docs, word2idx, meta)
    save_json(f"{args.out_dir}/tfidf_top_words.json", top)

    ppmi = ppmi_matrix(docs, word2idx, window=5)
    np.save(f"{args.out_dir}/ppmi_matrix.npy", ppmi)

    plot_tsne(ppmi, idx2word, f"{args.out_dir}/ppmi_tsne.png")

    query_words = ["Pakistan", "Hukumat", "Adalat", "Maeeshat", "Fauj", "Sehat", "Taleem", "Aabadi", "Lahore", "Karachi"]
    nn = {q: cosine_neighbors(ppmi, idx2word, word2idx, q, 5) for q in query_words}
    save_json(f"{args.out_dir}/ppmi_neighbors.json", nn)
    save_json(f"{args.out_dir}/word2idx.json", word2idx)


if __name__ == "__main__":
    main()
