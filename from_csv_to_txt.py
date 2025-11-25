# from_csv_to_txt.py
import argparse
import os
from pathlib import Path

import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Convert a CSV of reports to a folder of .txt files")
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument("--text-col", default="report", help="Name of the column with free-text clinical report")
    ap.add_argument("--id-col", default=None, help="Optional unique id column to name files; else uses row index")
    ap.add_argument("--outdir", default="input_from_csv", help="Folder to write .txt files")
    ap.add_argument("--limit", type=int, default=None, help="Optional: process only first N rows")
    args = ap.parse_args()

    in_csv = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if args.limit is not None:
        df = df.head(args.limit)

    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' was not found in {in_csv}. "
                         f"Found columns: {list(df.columns)}")

    if args.id_col and args.id_col not in df.columns:
        raise SystemExit(f"ID column '{args.id_col}' was not found in {in_csv}. "
                         f"Found columns: {list(df.columns)}")

    manifest_rows = []
    for idx, row in df.iterrows():
        # Pick file stem
        if args.id_col:
            stem_raw = str(row[args.id_col])
        else:
            stem_raw = str(idx)

        # Sanitize filename
        stem = "".join(c for c in stem_raw if c.isalnum() or c in ("-_")).strip() or f"row{idx}"
        text = str(row[args.text_col]) if not pd.isna(row[args.text_col]) else ""

        # Write text file
        txt_path = outdir / f"{stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        manifest_rows.append({
            "row_index": idx,
            "file_name": txt_path.name,
            "file_path": str(txt_path),
            "id"       : row[args.id_col] if args.id_col else None
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest_rows)} files to: {outdir}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
