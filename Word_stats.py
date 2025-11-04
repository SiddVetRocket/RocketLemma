# word_stats.py
"""
Compute word frequencies and surface 'common' vs. 'medical-jargon' terms from a CSV,
with convenient CSV exports and optional integration with your existing spaCy/medspaCy pipeline.

Install:
    pip install pandas spacy
    python -m spacy download en_core_web_sm

Usage (CLI):
    python word_stats.py --csv reports.csv --cols Findings Conclusions --out ./word_counts \
        --min-count 3 --min-doc-frac 0.002 --max-doc-frac 0.5 --top-k 200 \
        --keep-pos NOUN PROPN ADJ

Python API:
    from word_stats import analyze_terms
    res = analyze_terms("reports.csv", ["Findings","Conclusions"], output_dir="out")

Outputs (in output_dir):
    - terms_table.csv   : full table with tf/df/idf/doc_frac + jargon_score
    - common_terms.csv  : top-K frequent terms
    - jargon_terms.csv  : top-K jargon-ranked terms
"""

from __future__ import annotations
import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import pandas as pd
import spacy

DEFAULT_MODEL = "en_core_web_sm"


@dataclass
class AnalyzeConfig:
    min_count: int = 3
    min_doc_frac: float = 0.002
    max_doc_frac: float = 0.50
    top_k: int = 100
    lowercase: bool = True
    keep_pos: Optional[set] = None     # e.g., {"NOUN","PROPN","ADJ"}
    remove_stop: bool = False
    keep_hyphenated: bool = True
    min_len: int = 2
    output_dir: Optional[Path] = None  # write CSVs if provided


def _iter_docs(df: pd.DataFrame, text_cols: List[str]) -> Iterable[str]:
    for _, row in df.iterrows():
        parts = []
        for c in text_cols:
            val = row.get(c, "")
            if isinstance(val, str):
                parts.append(val)
        yield " ".join(parts).strip()


def _tokenize_docs(
    docs: Iterable[str],
    nlp,
    cfg: AnalyzeConfig
) -> Tuple[List[List[str]], Counter, Counter]:
    tokenized_docs: List[List[str]] = []
    tf = Counter()
    df = Counter()

    for doc in nlp.pipe(docs, batch_size=200, n_process=1, disable=["ner"]):
        seen_in_doc = set()
        toks = []
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            if not cfg.keep_hyphenated and "-" in t.text:
                continue

            lemma = t.lemma_ if t.lemma_ != "-PRON-" else t.lower_
            text_norm = lemma.lower() if cfg.lowercase else lemma

            if cfg.remove_stop and t.is_stop:
                continue
            if cfg.keep_pos and t.pos_ not in cfg.keep_pos:
                continue
            if not t.is_alpha and (not cfg.keep_hyphenated or "-" not in text_norm):
                continue
            if len(text_norm) < cfg.min_len:
                continue

            toks.append(text_norm)
            tf[text_norm] += 1
            seen_in_doc.add(text_norm)

        for w in seen_in_doc:
            df[w] += 1

        tokenized_docs.append(toks)

    return tokenized_docs, tf, df


def analyze_terms(
    csv_path: str,
    text_cols: List[str],
    min_count: int = 3,
    min_doc_frac: float = 0.002,
    max_doc_frac: float = 0.50,
    top_k: int = 100,
    keep_pos: Optional[set] = None,
    remove_stop: bool = False,
    nlp: Optional["spacy.Language"] = None,
    model: str = DEFAULT_MODEL,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    Analyze terms and (optionally) export CSVs.

    If you already have a spaCy/medspaCy pipeline, pass it via `nlp` and we won't load a model.
    """
    cfg = AnalyzeConfig(
        min_count=min_count,
        min_doc_frac=min_doc_frac,
        max_doc_frac=max_doc_frac,
        top_k=top_k,
        keep_pos=keep_pos,
        remove_stop=remove_stop,
        output_dir=Path(output_dir) if output_dir else None,
    )

    df = pd.read_csv(csv_path)
    docs = list(_iter_docs(df, text_cols))
    n_docs = len(docs)

    if nlp is None:
        nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = max(1_000_000, int(sum(len(d) for d in docs) * 1.1))

    _, tf, df_counter = _tokenize_docs(docs, nlp, cfg)

    rows = []
    for term, count in tf.items():
        if count < cfg.min_count:
            continue
        df_term = df_counter[term]
        doc_frac = df_term / max(1, n_docs)
        idf = math.log((n_docs + 1) / (df_term + 1)) + 1.0
        rows.append((term, count, df_term, doc_frac, idf))

    table = pd.DataFrame(rows, columns=["term", "tf", "df", "doc_frac", "idf"])
    table["jargon_score"] = table["idf"] * (table["tf"] + 1).map(math.log)

    # Common by TF
    common_terms_df = table.sort_values("tf", ascending=False).head(top_k)[["term", "tf"]]

    # Jargon candidates
    mask = (table["doc_frac"] >= min_doc_frac) & (table["doc_frac"] <= max_doc_frac)
    jargon_df = (
        table.loc[mask]
             .sort_values(["jargon_score", "idf", "tf"], ascending=False)
             .head(top_k)[["term", "jargon_score", "tf", "df"]]
    )

    # Exports
    if cfg.output_dir:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        table.sort_values("tf", ascending=False).to_csv(cfg.output_dir / "terms_table.csv", index=False)
        common_terms_df.to_csv(cfg.output_dir / "common_terms.csv", index=False)
        jargon_df.to_csv(cfg.output_dir / "jargon_terms.csv", index=False)

    return {
        "table": table.sort_values("tf", ascending=False).reset_index(drop=True),
        "common_terms": list(common_terms_df.itertuples(index=False, name=None)),
        "jargon_terms": list(jargon_df.itertuples(index=False, name=None)),
        "n_docs": n_docs,
        "config": cfg,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank common vs. jargon terms from CSV clinical text.")
    p.add_argument("--csv", required=True, help="Path to CSV file.")
    p.add_argument("--cols", nargs="+", required=True, help="Text columns to concatenate (e.g., Findings Conclusions).")
    p.add_argument("--out", default=None, help="Output directory for CSVs (optional).")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"spaCy model (default: {DEFAULT_MODEL}).")
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--min-doc-frac", type=float, default=0.002)
    p.add_argument("--max-doc-frac", type=float, default=0.50)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--keep-pos", nargs="*", default=None,
                   help="POS tags to keep (e.g., NOUN PROPN ADJ). If omitted, keep all.")
    p.add_argument("--remove-stop", action="store_true", help="Remove stopwords before counting.")
    return p.parse_args()


def _main():
    args = _parse_args()
    keep_pos = set(args.keep_pos) if args.keep_pos else None

    analyze_terms(
        csv_path=args.csv,
        text_cols=args.cols,
        min_count=args.min_count,
        min_doc_frac=args.min_doc_frac,
        max_doc_frac=args.max_doc_frac,
        top_k=args.top_k,
        keep_pos=keep_pos,
        remove_stop=args.remove_stop,
        model=args.model,
        output_dir=args.out,
    )


if __name__ == "__main__":
    _main()
