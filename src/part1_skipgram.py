import argparse
import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import UNK, build_vocab, ensure_dir, map_to_ids, parse_articles, save_json


class SkipGramPairs(Dataset):
    def __init__(self, docs, word2idx, window=5):
        pairs = []
        for d in docs:
            ids = map_to_ids(d, word2idx)
            for i, c in enumerate(ids):
                l = max(0, i - window)
                r = min(len(ids), i + window + 1)
                for j in range(l, r):
                    if i != j:
                        pairs.append((c, ids[j]))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.V = nn.Embedding(vocab_size, dim)
        self.U = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.V.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.U.weight)

    def forward(self, centers, pos, neg):
        vc = self.V(centers)
        uo = self.U(pos)
        uk = self.U(neg)

        pos_score = torch.sum(vc * uo, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-12)

        neg_score = torch.bmm(uk, vc.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-12), dim=1)

        return (pos_loss + neg_loss).mean()


def make_noise_table(freq, word2idx):
    p = np.zeros(len(word2idx), dtype=np.float64)
    for w, c in freq.items():
        if w in word2idx:
            p[word2idx[w]] = c
    p = np.power(p, 0.75)
    p = p / max(p.sum(), 1e-12)
    return p


def train_skipgram(
    docs,
    dim=100,
    window=5,
    neg_k=10,
    lr=1e-3,
    epochs=5,
    batch_size=512,
    max_vocab=10000,
    out_dir="embeddings",
    tag="cleaned",
):
    ensure_dir(out_dir)
    word2idx, idx2word, freq = build_vocab(docs, max_vocab=max_vocab)
    ds = SkipGramPairs(docs, word2idx, window=window)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SkipGramNeg(len(word2idx), dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    noise = make_noise_table(freq, word2idx)

    losses = []
    for ep in range(epochs):
        running = 0.0
        n = 0
        pbar = tqdm(dl, desc=f"epoch {ep + 1}/{epochs}")
        for centers, pos in pbar:
            neg = np.random.choice(len(word2idx), size=(len(centers), neg_k), p=noise)
            neg = torch.tensor(neg, dtype=torch.long)

            loss = model(centers.long(), pos.long(), neg)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1
            if n % 20 == 0:
                pbar.set_postfix(loss=running / n)

        losses.append(running / max(1, n))

    emb = 0.5 * (model.V.weight.detach().cpu().numpy() + model.U.weight.detach().cpu().numpy())
    np.save(f"{out_dir}/embeddings_w2v_{tag}.npy", emb)
    if tag == "cleaned" and dim == 100:
        np.save(f"{out_dir}/embeddings_w2v.npy", emb)

    save_json(f"{out_dir}/word2idx_{tag}.json", word2idx)
    if tag == "cleaned" and dim == 100:
        save_json(f"{out_dir}/word2idx.json", word2idx)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title(f"Skip-gram training loss ({tag}, d={dim})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/skipgram_loss_{tag}_d{dim}.png")
    plt.close()

    return emb, word2idx, idx2word, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="cleaned.txt")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--neg_k", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_vocab", type=int, default=10000)
    ap.add_argument("--out_dir", default="embeddings")
    ap.add_argument("--tag", default="cleaned")
    args = ap.parse_args()

    docs = parse_articles(args.corpus)
    train_skipgram(
        docs,
        dim=args.dim,
        window=args.window,
        neg_k=args.neg_k,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_vocab=args.max_vocab,
        out_dir=args.out_dir,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
