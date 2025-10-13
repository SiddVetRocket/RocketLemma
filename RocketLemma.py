#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medsum_pipeline.py

End-to-end scaffold for medical-report processing:
- CSV loading (expects columns like 'report', 'findings', 'conclusions')
- spaCy preprocessing (tokenization/lemmatization if model available)
- Dynamic vocabulary construction (with <PAD>, <SOS>, <EOS>, <UNK>)
- Seq2Seq (Encoder/Decoder) model in PyTorch (LSTM-based)
- Custom training loop with teacher forcing
- Inference (greedy / optional beam) for summarization
- Condition classifier: "yes / no / maybe" using negation/uncertainty cues
- Keyword filtering over 'findings'/'conclusions'
- Simple jargon -> layman simplifier

NOTE:
- You can train the model by providing --train and setting hyperparameters.
- If spaCy's 'en_core_web_sm' isn't installed, we fallback to a blank English pipeline.
- Adjust COLUMN_NAMES below if your CSV uses different headers.

Usage examples:
    python medsum_pipeline.py --csv /mnt/data/mv_reports_10k.csv --sample 5 --demo
    python medsum_pipeline.py --csv /mnt/data/mv_reports_10k.csv --filter pneumonia --columns findings conclusions
    python medsum_pipeline.py --csv /mnt/data/mv_reports_10k.csv --classify pneumonia
    python medsum_pipeline.py --csv /mnt/data/mv_reports_10k.csv --summarize "Right lower-lobe consolidation..."

"""

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional

import pandas as pd

# ---- spaCy setup ----
import spacy
from spacy.language import Language

def load_spacy() -> Language:
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])  # fast enough; keep tagger/lemmatizer by default
    except Exception:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp

NLP = load_spacy()

# ---- Configurable column names ----
COLUMN_NAMES = {
    "report": "report",           # full free-text report
    "findings": "findings",       # findings section (if available)
    "conclusions": "conclusions", # conclusions/impression section (if available)
}

# ---- Tokenization ----
def spacy_tokenize(text: str, lowercase: bool = True, lemmatize: bool = False) -> List[str]:
    if not isinstance(text, str):
        return []
    if lowercase:
        text = text.lower()
    doc = NLP(text)
    toks = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if lemmatize and tok.lemma_:
            toks.append(tok.lemma_)
        else:
            toks.append(tok.text)
    return toks

# ---- Vocabulary ----
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
PAD, SOS, EOS, UNK = range(4)

@dataclass
class Vocab:
    word2idx: Dict[str, int]
    idx2word: Dict[int, str]

def build_vocab(tokenized_texts: Iterable[List[str]], max_size: int = 20000, min_freq: int = 2) -> Vocab:
    counter = Counter()
    for toks in tokenized_texts:
        counter.update(toks)

    most_common = [w for w, f in counter.items() if f >= min_freq]
    most_common = sorted(most_common, key=lambda w: -counter[w])[: max(0, max_size - len(SPECIAL_TOKENS))]

    vocab_list = SPECIAL_TOKENS + most_common
    word2idx = {w: i for i, w in enumerate(vocab_list)}
    idx2word = {i: w for w, i in word2idx.items()}
    return Vocab(word2idx, idx2word)

def numericalize(tokens: List[str], vocab: Vocab, add_sos_eos: bool = True, max_len: Optional[int] = None) -> List[int]:
    ids = [vocab.word2idx.get(t, UNK) for t in tokens]
    if add_sos_eos:
        ids = [SOS] + ids + [EOS]
    if max_len is not None:
        ids = ids[:max_len] + ([PAD] * max(0, max_len - len(ids)))
    return ids

# ---- Simple Dataset wrapper ----
class TextDataset:
    def __init__(self, rows: Iterable[str], vocab: Vocab, max_len: int = 200, lemmatize=False):
        self.rows = list(rows)
        self.vocab = vocab
        self.max_len = max_len
        self.lemmatize = lemmatize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text = self.rows[idx]
        toks = spacy_tokenize(text, lemmatize=self.lemmatize)
        ids = numericalize(toks, self.vocab, add_sos_eos=True, max_len=self.max_len)
        return ids

# ---- PyTorch Seq2Seq (LSTM) ----
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=512, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens=None):
        # src: (B, T)
        embedded = self.dropout(self.embedding(src))  # (B, T, E)
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: (B, T, H)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=512, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        # input_token: (B,) -> we need (B, 1)
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))  # (B, 1, E)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # output: (B, 1, H)
        logits = self.fc_out(output.squeeze(1))  # (B, V)
        return logits, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (B, T_src), trg: (B, T_trg)
        B, T_trg = trg.size()
        hidden, cell = self.encoder(src)
        outputs = torch.zeros(B, T_trg, self.decoder.fc_out.out_features, device=self.device)

        input_token = trg[:, 0]  # <SOS>
        for t in range(1, T_trg):
            logits, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

# ---- Training Loop ----
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for src_batch, trg_batch in dataloader:
        src_batch = src_batch.to(device)
        trg_batch = trg_batch.to(device)
        optimizer.zero_grad()
        logits = model(src_batch, trg_batch, teacher_forcing_ratio=0.5)
        # Shift targets to align with predictions
        logits_2d = logits[:, 1:, :].reshape(-1, logits.size(-1))    # exclude first (SOS) time step
        targets_2d = trg_batch[:, 1:].reshape(-1)
        loss = criterion(logits_2d, targets_2d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

@torch.no_grad()
def greedy_decode(model, src_ids: List[int], vocab: Vocab, max_len=60, device="cpu") -> str:
    model.eval()
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    hidden, cell = model.encoder(src)
    input_token = torch.tensor([SOS], dtype=torch.long, device=device)
    outputs = []
    for _ in range(max_len):
        logits, hidden, cell = model.decoder(input_token, hidden, cell)
        next_id = int(logits.argmax(-1).item())
        if next_id == EOS:
            break
        outputs.append(next_id)
        input_token = torch.tensor([next_id], dtype=torch.long, device=device)
    words = [vocab.idx2word.get(i, "<UNK>") for i in outputs]
    return " ".join(words)

# ---- Keyword Filtering ----
def filter_rows(df: pd.DataFrame, keyword: str, columns: List[str]) -> pd.DataFrame:
    pat = fr"\b{pd.regex.escape(keyword.lower())}\b"
    mask = False
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].fillna("").str.lower().str.contains(pat, regex=True)
    return df.loc[mask]

# ---- Condition Classifier (yes/no/maybe) ----
NEGATION_CUES = [
    "no", "not", "without", "denies", "deny", "negative for", "free of", "rule out", "ruled out", "absent"
]
UNCERTAIN_CUES = [
    "possible", "possibly", "cannot exclude", "can not exclude", "could be", "suggests", "suspicious for",
    "concern for", "consistent with", "likely", "unlikely", "probable", "question of", "indeterminate"
]

def classify_condition(text: str, condition: str = "pneumonia") -> str:
    """
    Returns 'yes' if condition affirmed, 'no' if negated, 'maybe' if uncertain or ambiguous.
    Uses a simple windowed cue check. You can replace with a proper negation scope detector (e.g., negspaCy).
    """
    if not isinstance(text, str) or not text.strip():
        return "maybe"

    t = text.lower()
    cond = condition.lower()

    # quick contains
    if cond not in t:
        return "maybe"

    # windowed check using spaCy tokens
    doc = NLP(t)
    tokens = [tok.text for tok in doc]

    # find all indices where condition appears
    idxs = [i for i, w in enumerate(tokens) if w == cond or w.rstrip('s') == cond.rstrip('s')]

    def window(i, w=5):
        lo, hi = max(0, i - w), min(len(tokens), i + w + 1)
        return " ".join(tokens[lo:hi])

    neg_hit = False
    unc_hit = False
    for i in idxs:
        ctx = window(i, w=6)
        if any(cue in ctx for cue in NEGATION_CUES):
            neg_hit = True
        if any(cue in ctx for cue in UNCERTAIN_CUES):
            unc_hit = True

    if neg_hit and not unc_hit:
        return "no"
    if unc_hit and not neg_hit:
        return "maybe"
    # if both present, lean to 'maybe' (ambiguous) else 'yes'
    return "maybe" if unc_hit else "yes"

# ---- Jargon Simplifier ----
DEFAULT_JARGON_MAP = {
    "infiltrate": "abnormal substance spread",
    "atelectasis": "partial lung collapse",
    "consolidation": "areas of filled-in lung",
    "cardiomegaly": "enlarged heart",
    "pneumothorax": "collapsed lung",
    "edema": "swelling",
    "effusion": "fluid buildup",
    "ischemia": "reduced blood flow",
    "lesion": "abnormal area",
}

def simplify_jargon(text: str, mapping: Dict[str, str] = DEFAULT_JARGON_MAP) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for term, simple in mapping.items():
        out = pd.Series([out]).str.replace(rf"\b{term}\b", simple, regex=True, case=False).iloc[0]
    return out

# ---- Data Loading ----
def load_csv(path: str, sample: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample is not None and sample > 0:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    return df

# ---- CLI / Demo ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="/mnt/data/mv_reports_10k.csv", help="Path to CSV file")
    p.add_argument("--sample", type=int, default=0, help="Optional sample size for quick tests")
    p.add_argument("--filter", type=str, default="", help="Keyword to filter rows by (e.g., pneumonia)")
    p.add_argument("--columns", nargs="*", default=["findings", "conclusions"], help="Columns to search for --filter")
    p.add_argument("--classify", type=str, default="", help="Condition to classify as yes/no/maybe (e.g., pneumonia)")
    p.add_argument("--summarize", type=str, default="", help="Free-text to summarize via greedy decode (demo)")
    p.add_argument("--demo", action="store_true", help="Run a small end-to-end demo over a few rows")
    p.add_argument("--pneumonia_report", type=int, default=0, help="Pretty print N rows with findings, conclusions, layman text, and pneumonia label")
    p.add_argument("--train", action="store_true", help="(Optional) Train seq2seq on report->conclusions (toy)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=120)
    args = p.parse_args()

    # Load data
    df = load_csv(args.csv, sample=args.sample if args.sample else None)
    print(f"Loaded CSV: {args.csv}  rows={len(df)}  cols={list(df.columns)}")
    # Normalize headers so we can use 'findings'/'conclusions' everywhere
    df = df.rename(columns=str.lower)


    # Tokenize corpus for vocab (use 'report' and 'conclusions' if they exist)
    corpus_cols = [c for c in [COLUMN_NAMES["report"], COLUMN_NAMES["findings"], COLUMN_NAMES["conclusions"]] if c in df.columns]
    all_tokens = []
    for c in corpus_cols:
        all_tokens.extend(spacy_tokenize(" ".join(df[c].fillna("").astype(str))))
    vocab = build_vocab([all_tokens], max_size=20000, min_freq=2)
    print(f"Vocab size: {len(vocab.word2idx)} (with specials: {SPECIAL_TOKENS})")

    # Optional keyword filter
    if args.filter:
        cols = args.columns or ["findings", "conclusions"]
        filt = filter_rows(df, args.filter, columns=cols)
        print(f"\n--- Filter results for '{args.filter}' in {cols}: {len(filt)} rows ---")
        for i, row in filt.head(5).iterrows():
            txt = " ".join([str(row.get(c, "")) for c in cols if c in df.columns])
            print(f"[{i}] {txt[:200]}...")

    # Optional condition classification (yes/no/maybe) over conclusions+findings
    if args.classify:
        col_text = (df.get("conclusions", pd.Series([""] * len(df))).fillna("") + " " +
                    df.get("findings", pd.Series([""] * len(df))).fillna(""))
        preds = col_text.apply(lambda t: classify_condition(t, condition=args.classify))
        counts = preds.value_counts()
        print(f"\n--- Condition '{args.classify}' distribution ---")
        for k, v in counts.items():
            print(f"{k}: {v}")
        # Show a few examples
        print("\nExamples:")
        for label in ["yes", "no", "maybe"]:
            ex_rows = df[preds == label].head(2)
            if not ex_rows.empty:
                print(f"\n[{label.upper()}]")
                for _, r in ex_rows.iterrows():
                    sample_txt = (str(r.get('conclusions', '')) + " " + str(r.get('findings',''))).strip()
                    print("-", sample_txt[:240].replace("\n", " "), "...")

    # Optional demo: simplify jargon & summarize first few reports
    if args.demo:
        take = min(5, len(df))
        print("\n--- Demo: Simplify + Classify + (Stub) Summarize ---")
        for i in range(take):
            rep = str(df.iloc[i].get(COLUMN_NAMES["report"], ""))
            find = str(df.iloc[i].get(COLUMN_NAMES["findings"], ""))
            concl = str(df.iloc[i].get(COLUMN_NAMES["conclusions"], ""))
            merged = " ".join([concl, find]).strip()
            simple = simplify_jargon(merged)
            label = classify_condition(merged, condition="pneumonia")
            print(f"\nRow {i}:")
            print("Original:", merged[:300].replace("\n"," "))
            print("Layman :", simple[:300])
            print("Pneumonia:", label)

    if args.pneumonia_report and args.pneumonia_report > 0:
        print_pneumonia_report(df, limit=args.pneumonia_report, width=100)


    # Optional toy training (report -> conclusions). This is a minimal example and
    # may require cleaning/shortening and more epochs to be meaningful.
    if args.train and COLUMN_NAMES["report"] in df.columns and COLUMN_NAMES["conclusions"] in df.columns:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTraining on device: {device}")

        # Prepare tiny parallel data (report -> conclusions)
        src_texts = df[COLUMN_NAMES["report"]].fillna("").astype(str).tolist()
        trg_texts = df[COLUMN_NAMES["conclusions"]].fillna("").astype(str).tolist()

        # Tokenize and numericalize each row separately for both src and trg
        src_toks = [spacy_tokenize(t) for t in src_texts]
        trg_toks = [spacy_tokenize(t) for t in trg_texts]

        # Build joint vocab over both sides for simplicity
        joint_vocab = build_vocab(src_toks + trg_toks, max_size=20000, min_freq=3)
        vocab_size = len(joint_vocab.word2idx)

        def enc(toks): return numericalize(toks, joint_vocab, add_sos_eos=True, max_len=args.max_len)

        src_ids = [enc(t) for t in src_toks]
        trg_ids = [enc(t) for t in trg_toks]

        # Tensorize
        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        trg_tensor = torch.tensor(trg_ids, dtype=torch.long)

        # Basic batching
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(src_tensor, trg_tensor)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Model
        enc_model = Encoder(vocab_size)
        dec_model = Decoder(vocab_size)
        model = Seq2Seq(enc_model, dec_model, device=device).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        crit = nn.CrossEntropyLoss(ignore_index=PAD)

        for ep in range(1, args.epochs + 1):
            loss = train_epoch(model, loader, opt, crit, device)
            print(f"Epoch {ep}/{args.epochs}  loss={loss:.4f}")

        # Quick qualitative check
        if len(src_ids):
            sample_src = src_ids[0]
            summary = greedy_decode(model, sample_src, joint_vocab, max_len=60, device=device)
            print("\nSample greedy decode summary:\n", summary)



# ---- Pretty Pneumonia Report ----
import textwrap

def make_layman(findings: str, conclusions: str) -> str:
    merged = " ".join([str(conclusions or ""), str(findings or "")]).strip()
    return simplify_jargon(merged)

def print_pneumonia_report(df: pd.DataFrame, limit: int = 10, width: int = 100):
    print("\n=== Pneumonia Report (pretty view) ===")
    cols = [c for c in ["findings", "conclusions"] if c in df.columns]
    if not cols:
        print("No 'findings'/'conclusions' columns found.")
        return

    # compute labels
    merged_text = (df.get("conclusions", pd.Series([""] * len(df))).fillna("") + " " +
                   df.get("findings", pd.Series([""] * len(df))).fillna(""))
    labels = merged_text.apply(lambda t: classify_condition(t, condition="pneumonia"))

    for i, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        findings = str(row.get("findings", "")).strip()
        conclusions = str(row.get("conclusions", "")).strip()
        layman = make_layman(findings, conclusions)
        label = labels.iloc[row.name]

        print(f"\nRow {i}")
        print("-" * width)
        def bl(h, t):
            print(h)
            for line in textwrap.wrap(t, width=width):
                print("  " + line)
        bl("Original Findings:", findings or "(none)")
        bl("Original Conclusions:", conclusions or "(none)")
        bl("Layman Summary:", layman or "(none)")
        print(f"Pneumonia: {label}")
        print("-" * width)


if __name__ == "__main__":
    main()
