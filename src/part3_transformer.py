import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk):
        super().__init__()
        self.scale = math.sqrt(dk)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        w = torch.softmax(scores, dim=-1)
        out = torch.matmul(w, v)
        return out, w


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dk=32, dv=32):
        super().__init__()
        self.n_heads = n_heads
        self.dk = dk
        self.dv = dv

        self.wq = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(n_heads)])
        self.wk = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(n_heads)])
        self.wv = nn.ModuleList([nn.Linear(d_model, dv) for _ in range(n_heads)])
        self.wo = nn.Linear(n_heads * dv, d_model)
        self.attn = ScaledDotProductAttention(dk)

    def forward(self, x, mask=None):
        heads, weights = [], []
        for h in range(self.n_heads):
            q = self.wq[h](x)
            k = self.wk[h](x)
            v = self.wv[h](x)
            out, w = self.attn(q, k, v, mask)
            heads.append(out)
            weights.append(w)
        cat = torch.cat(heads, dim=-1)
        return self.wo(cat), weights


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=128, dff=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model))

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=257):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dff=512, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dk=32, dv=32)
        self.ffn = PositionwiseFFN(d_model=d_model, dff=dff)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        a, w = self.mha(self.ln1(x), mask)
        x = x + self.drop(a)
        f = self.ffn(self.ln2(x))
        x = x + self.drop(f)
        return x, w


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes=5, d_model=128, n_layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model, max_len=257)
        self.blocks = nn.ModuleList([EncoderBlock(d_model=d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x, pad_mask=None):
        b = x.size(0)
        e = self.emb(x)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, e], dim=1)
        x = self.pos(x)

        attn_last = None
        for blk in self.blocks:
            x, w = blk(x, pad_mask)
            attn_last = w

        logits = self.head(x[:, 0])
        return logits, attn_last


def make_mask(x):
    b, t = x.shape
    base = (x != 0).unsqueeze(1).unsqueeze(2)
    cls = torch.ones((b, 1, 1, 1), dtype=torch.bool, device=x.device)
    k_mask = torch.cat([cls, base], dim=-1)
    return k_mask


def cosine_warmup(step, warmup=50, total=1000):
    if step < warmup:
        return max(step, 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def train_eval(npz_path, word2idx_path, out_model="models/transformer_cls.pt"):
    data = np.load(npz_path)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    with open(word2idx_path, "r", encoding="utf-8") as f:
        w2i = json.load(f)

    train_dl = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=64)
    test_dl = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=64)

    model = TransformerClassifier(vocab_size=len(w2i), n_classes=5)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    total_steps = 20 * max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: cosine_warmup(s, warmup=50, total=total_steps))

    tr_loss_hist, va_loss_hist, tr_acc_hist, va_acc_hist = [], [], [], []
    best_acc = -1

    step = 0
    for _ in range(20):
        model.train()
        losses, ytrue, ypred = [], [], []
        for xb, yb in train_dl:
            mask = make_mask(xb)
            logits, _ = model(xb, mask)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1

            losses.append(float(loss.item()))
            ytrue.extend(yb.tolist())
            ypred.extend(torch.argmax(logits, dim=-1).tolist())

        tr_loss_hist.append(float(np.mean(losses)))
        tr_acc_hist.append(float(accuracy_score(ytrue, ypred)))

        model.eval()
        losses, ytrue, ypred = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                mask = make_mask(xb)
                logits, _ = model(xb, mask)
                loss = nn.functional.cross_entropy(logits, yb)
                losses.append(float(loss.item()))
                ytrue.extend(yb.tolist())
                ypred.extend(torch.argmax(logits, dim=-1).tolist())

        va_loss = float(np.mean(losses))
        va_acc = float(accuracy_score(ytrue, ypred))
        va_loss_hist.append(va_loss)
        va_acc_hist.append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), out_model)

    model.load_state_dict(torch.load(out_model, map_location="cpu"))
    model.eval()
    ytrue, ypred = [], []
    attn_dump = []
    with torch.no_grad():
        for xb, yb in test_dl:
            mask = make_mask(xb)
            logits, attn = model(xb, mask)
            ytrue.extend(yb.tolist())
            ypred.extend(torch.argmax(logits, dim=-1).tolist())
            if len(attn_dump) < 3:
                attn_dump.append(attn)

    acc = accuracy_score(ytrue, ypred)
    mf1 = f1_score(ytrue, ypred, average="macro")
    cm = confusion_matrix(ytrue, ypred).tolist()

    plt.figure(figsize=(9, 5))
    plt.plot(tr_loss_hist, label="train loss")
    plt.plot(va_loss_hist, label="val loss")
    plt.plot(tr_acc_hist, label="train acc")
    plt.plot(va_acc_hist, label="val acc")
    plt.title("Transformer training curves")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/transformer_curves.png")
    plt.close()

    with open("models/transformer_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_acc": acc, "macro_f1": mf1, "confusion_matrix": cm}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/topic_cls.npz")
    ap.add_argument("--word2idx", default="embeddings/word2idx.json")
    args = ap.parse_args()
    train_eval(args.data, args.word2idx)


if __name__ == "__main__":
    main()
