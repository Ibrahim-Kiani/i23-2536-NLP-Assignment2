import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


def read_conll(path):
    sents, tags = [], []
    cur_s, cur_t = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_s:
                    sents.append(cur_s)
                    tags.append(cur_t)
                cur_s, cur_t = [], []
                continue
            tok, tag = line.split("\t")
            cur_s.append(tok)
            cur_t.append(tag)
    if cur_s:
        sents.append(cur_s)
        tags.append(cur_t)
    return sents, tags


class SeqDataset(Dataset):
    def __init__(self, sents, tags, word2idx, tag2idx):
        self.x = [[word2idx.get(t, word2idx.get("<UNK>", 1)) for t in s] for s in sents]
        self.y = [[tag2idx[t] for t in z] for z in tags]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


def collate(batch):
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs])
    xpad = pad_sequence(xs, batch_first=True, padding_value=0)
    ypad = pad_sequence(ys, batch_first=True, padding_value=-100)
    return xpad, ypad, lens


class CRF(nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.n_tags = n_tags
        self.trans = nn.Parameter(torch.randn(n_tags, n_tags) * 0.1)

    def forward_alg(self, emissions, mask):
        score = emissions[:, 0]
        for t in range(1, emissions.size(1)):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.trans.unsqueeze(0)
            nxt = score.unsqueeze(2) + trans + emit
            score = torch.logsumexp(nxt, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), score, score)
        return torch.logsumexp(score, dim=1)

    def score_gold(self, emissions, tags, mask):
        b, t, _ = emissions.shape
        s = emissions[torch.arange(b), 0, tags[:, 0]]
        for i in range(1, t):
            emit = emissions[torch.arange(b), i, tags[:, i]]
            trans = self.trans[tags[:, i - 1], tags[:, i]]
            s = s + (emit + trans) * mask[:, i]
        return s

    def neg_log_likelihood(self, emissions, tags, mask):
        z = self.forward_alg(emissions, mask)
        g = self.score_gold(emissions, tags, mask)
        return (z - g).mean()

    def viterbi_decode(self, emissions, mask):
        b, t, n = emissions.shape
        score = emissions[:, 0]
        back = []
        for i in range(1, t):
            nxt = score.unsqueeze(2) + self.trans.unsqueeze(0)
            best_score, best_tag = torch.max(nxt, dim=1)
            score = best_score + emissions[:, i]
            back.append(best_tag)

        out = []
        best_last = torch.argmax(score, dim=1)
        for bi in range(b):
            length = int(mask[bi].sum().item())
            seq = [int(best_last[bi])]
            for i in range(length - 2, 0, -1):
                seq.append(int(back[i - 1][bi, seq[-1]]))
            out.append(list(reversed(seq)))
        return out


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, n_tags, emb_dim=100, hid=128, emb_weights=None, freeze=False, use_crf=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if emb_weights is not None:
            self.emb.weight.data.copy_(torch.tensor(emb_weights, dtype=torch.float32))
        self.emb.weight.requires_grad = not freeze

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
        )
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * hid, n_tags)
        self.use_crf = use_crf
        self.crf = CRF(n_tags) if use_crf else None

    def forward(self, x, lens):
        e = self.emb(x)
        p = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(p)
        o, _ = pad_packed_sequence(o, batch_first=True)
        o = self.drop(o)
        return self.fc(o)


def run_epoch(model, dl, optimizer=None, use_crf=False):
    train = optimizer is not None
    model.train() if train else model.eval()
    total = 0.0
    all_y, all_p = [], []

    for x, y, lens in dl:
        mask = y != -100
        logits = model(x, lens)

        if use_crf:
            y2 = y.clone()
            y2[~mask] = 0
            loss = model.crf.neg_log_likelihood(logits, y2, mask)
            preds = model.crf.viterbi_decode(logits.detach(), mask)
            flat_pred, flat_gold = [], []
            for i in range(len(preds)):
                g = y[i][mask[i]].tolist()
                p = preds[i]
                flat_pred.extend(p[: len(g)])
                flat_gold.extend(g)
        else:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            pred = torch.argmax(logits, dim=-1)
            flat_pred = pred[mask].tolist()
            flat_gold = y[mask].tolist()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total += float(loss.item())
        all_y.extend(flat_gold)
        all_p.extend(flat_pred)

    acc = accuracy_score(all_y, all_p) if all_y else 0.0
    f1 = f1_score(all_y, all_p, average="macro") if all_y else 0.0
    return total / max(1, len(dl)), acc, f1, all_y, all_p


def train_model(train_path, val_path, test_path, word2idx_path, emb_path, out_model, use_crf=False):
    tr_s, tr_t = read_conll(train_path)
    va_s, va_t = read_conll(val_path)
    te_s, te_t = read_conll(test_path)

    with open(word2idx_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    emb = np.load(emb_path)

    tagset = sorted({t for seq in tr_t + va_t + te_t for t in seq})
    tag2idx = {t: i for i, t in enumerate(tagset)}

    tr_ds = SeqDataset(tr_s, tr_t, word2idx, tag2idx)
    va_ds = SeqDataset(va_s, va_t, word2idx, tag2idx)
    te_ds = SeqDataset(te_s, te_t, word2idx, tag2idx)

    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False, collate_fn=collate)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False, collate_fn=collate)

    full_emb = np.random.normal(0, 0.01, (len(word2idx), emb.shape[1])).astype(np.float32)
    for w, i in word2idx.items():
        if i < len(emb):
            full_emb[i] = emb[i]

    histories = {}
    best = {}
    for freeze in [True, False]:
        model = BiLSTMTagger(len(word2idx), len(tagset), emb_dim=emb.shape[1], emb_weights=full_emb, freeze=freeze, use_crf=use_crf)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_f1, patience = -1.0, 0
        tr_loss_hist, va_loss_hist = [], []
        for _ in range(40):
            tr_loss, _, _, _, _ = run_epoch(model, tr_dl, optimizer=opt, use_crf=use_crf)
            va_loss, _, va_f1, _, _ = run_epoch(model, va_dl, optimizer=None, use_crf=use_crf)
            tr_loss_hist.append(tr_loss)
            va_loss_hist.append(va_loss)
            if va_f1 > best_f1:
                best_f1 = va_f1
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if patience >= 5:
                break

        model.load_state_dict(best_state)
        te_loss, te_acc, te_f1, y_true, y_pred = run_epoch(model, te_dl, optimizer=None, use_crf=use_crf)
        cm = confusion_matrix(y_true, y_pred).tolist()
        key = "frozen" if freeze else "finetuned"

        histories[key] = {"train_loss": tr_loss_hist, "val_loss": va_loss_hist}
        best[key] = {"test_loss": te_loss, "test_acc": te_acc, "test_f1": te_f1, "confusion_matrix": cm}

        torch.save(model.state_dict(), out_model.replace(".pt", f"_{key}.pt"))

    plt.figure(figsize=(8, 5))
    for mode in ["frozen", "finetuned"]:
        plt.plot(histories[mode]["train_loss"], label=f"train {mode}")
        plt.plot(histories[mode]["val_loss"], label=f"val {mode}")
    plt.title("BiLSTM training/validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_model.replace(".pt", "_loss.png"))
    plt.close()

    with open(out_model.replace(".pt", "_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pos", "ner"], required=True)
    ap.add_argument("--word2idx", default="embeddings/word2idx.json")
    ap.add_argument("--emb", default="embeddings/embeddings_w2v.npy")
    args = ap.parse_args()

    if args.task == "pos":
        train_model("data/pos_train.conll", "data/pos_val.conll", "data/pos_test.conll", args.word2idx, args.emb, "models/bilstm_pos.pt", use_crf=False)
    else:
        train_model("data/ner_train.conll", "data/ner_val.conll", "data/ner_test.conll", args.word2idx, args.emb, "models/bilstm_ner.pt", use_crf=True)


if __name__ == "__main__":
    main()
