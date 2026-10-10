"""Train the Chhat Pakistani-address parser.

Pipeline:
  1. generate synthetic labeled addresses (chhat.parser.generate_data)
  2. build word vocab, encode BIO labels
  3. train BiLSTM tagger on CPU
  4. evaluate: token accuracy + span-level entity F1
  5. save model bundle -> models/address_parser.pt (+ metrics.json)

Run: .venv/bin/python -m chhat.parser.train --n-train 8000 --epochs 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .generate_data import generate
from .model import AddressTagger

PAD, UNK = "<pad>", "<unk>"


def build_vocab(rows: list[dict], min_freq: int = 1) -> dict[str, int]:
    counts = Counter(t for r in rows for t in r["tokens"])
    vocab = {PAD: 0, UNK: 1}
    for tok, c in counts.most_common():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


LABELS = ["O",
          "B-HOUSE", "I-HOUSE", "B-STREET", "I-STREET", "B-BLOCK",
          "B-COLONY", "I-COLONY", "B-CITY"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def tokenize_text(text: str) -> list[str]:
    return text.replace(",", " ,").split()


class AddrDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in r["tokens"]]
        labs = [LABEL2ID[l] for l in r["labels"]]
        return torch.tensor(ids), torch.tensor(labs)


def collate(batch):
    ids, labs = zip(*batch)
    lens = [len(x) for x in ids]
    T = max(lens)
    b = len(batch)
    id_t = torch.zeros(b, T, dtype=torch.long)
    lab_t = torch.full((b, T), -100, dtype=torch.long)
    mask = torch.zeros(b, T, dtype=torch.float)
    for i, (x, y) in enumerate(zip(ids, labs)):
        id_t[i, :len(x)] = x
        lab_t[i, :len(y)] = y
        mask[i, :len(x)] = 1.0
    return id_t, lab_t, mask


def spans_of(labels: list[str]) -> set[tuple[int, int, str]]:
    spans, start, tag = set(), None, None
    for i, l in enumerate(labels + ["O"]):
        if l.startswith("B-"):
            if start is not None:
                spans.add((start, i, tag))
            start, tag = i, l[2:]
        elif l.startswith("I-") and start is not None and l[2:] == tag:
            continue
        else:
            if start is not None:
                spans.add((start, i, tag))
                start, tag = None, None
    return spans


def evaluate(model, loader, device) -> dict:
    model.eval()
    tok_ok = tok_tot = 0
    pred_spans: set = set()
    gold_spans: set = set()
    with torch.no_grad():
        for ids, labs, mask in loader:
            ids, labs, mask = ids.to(device), labs.to(device), mask.to(device)
            logits = model(ids, mask)
            pred = logits.argmax(-1)
            valid = labs != -100
            tok_ok += ((pred == labs) & valid.bool()).sum().item()
            tok_tot += valid.sum().item()
            for p_row, g_row, m_row in zip(pred.cpu(), labs.cpu(), mask.cpu()):
                T = int(m_row.sum())
                p = [LABELS[i] for i in p_row[:T].tolist()]
                g = [LABELS[i] for i in g_row[:T].tolist()]
                off = len(pred_spans) + len(gold_spans)
                for s in spans_of(p):
                    pred_spans.add((off, *s))
                for s in spans_of(g):
                    gold_spans.add((off, *s))
    tp = len(pred_spans & gold_spans)
    prec = tp / max(1, len(pred_spans))
    rec = tp / max(1, len(gold_spans))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"token_accuracy": tok_ok / max(1, tok_tot),
            "entity_precision": prec, "entity_recall": rec, "entity_f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/address_parser.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print("generating data...", flush=True)
    train_rows = generate(args.n_train, seed=args.seed)
    test_rows = generate(args.n_test, seed=args.seed + 999)
    vocab = build_vocab(train_rows)

    train_loader = DataLoader(AddrDataset(train_rows, vocab), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    test_loader = DataLoader(AddrDataset(test_rows, vocab), batch_size=args.batch_size,
                             collate_fn=collate)

    model = AddressTagger(len(vocab), len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ids, labs, mask in train_loader:
            ids, labs, mask = ids.to(device), labs.to(device), mask.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, mask).reshape(-1, len(LABELS)), labs.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n += 1
        m = evaluate(model, test_loader, device)
        print(f"epoch {epoch:02d} loss={total / n:.4f} "
              f"tok_acc={m['token_accuracy']:.4f} entity_f1={m['entity_f1']:.4f}", flush=True)

    final = evaluate(model, test_loader, device)
    print("FINAL:", json.dumps(final, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "labels": LABELS, "config": {"emb_dim": 64, "hidden_dim": 128,
                                            "num_layers": 2, "dropout": 0.2}},
               args.out)
    with open(os.path.splitext(args.out)[0] + "_metrics.json", "w") as f:
        json.dump({"test": final, "n_train": args.n_train, "n_test": args.n_test,
                   "epochs": args.epochs, "seed": args.seed}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
