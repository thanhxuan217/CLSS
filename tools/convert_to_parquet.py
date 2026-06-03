#!/usr/bin/env python3
"""
Convert CoNLL column-format text files (.txt) ↔ Parquet format (.parquet).

Parquet files are ~5-10x smaller than plain text, faster to read, and ideal
for transferring datasets between environments (e.g., Kaggle → Slurm server).

Parquet schema:
    sentence_id (int)   — groups tokens into sentences (replaces blank-line separators)
    text        (str)   — token text (column 0)
    <label>     (str)   — label column(s) (column 1, 2, ...)

═══════════════════════════════════════════════════════════════════════════════
Usage (CLI):
═══════════════════════════════════════════════════════════════════════════════

  # Convert a single data folder (train.txt, dev.txt, test.txt → .parquet)
  python tools/convert_to_parquet.py data/sino_nom_punct

  # Convert multiple folders at once
  python tools/convert_to_parquet.py data/sino_nom_punct data/sino_nom_punct_doc

  # Custom column format (default is "0:text,1:punct")
  python tools/convert_to_parquet.py data/my_ner_data --columns "0:text,1:ner"

═══════════════════════════════════════════════════════════════════════════════
Usage (Kaggle Notebook):
═══════════════════════════════════════════════════════════════════════════════

  # ── Cell 1: Install pyarrow ──────────────────────────────────────────────
  !pip install pyarrow -q

  # ── Cell 2: Convert txt → parquet ────────────────────────────────────────
  !python tools/convert_to_parquet.py data/sino_nom_punct data/sino_nom_punct_doc

  # ── Cell 3: Verify & download ────────────────────────────────────────────
  import os
  for folder in ['data/sino_nom_punct', 'data/sino_nom_punct_doc']:
      for f in os.listdir(folder):
          if f.endswith('.parquet'):
              path = os.path.join(folder, f)
              size_mb = os.path.getsize(path) / 1024 / 1024
              print(f"  {path}  ({size_mb:.2f} MB)")

  # ── Cell 4: (Optional) Zip & download ───────────────────────────────────
  !zip -j sino_nom_punct.zip data/sino_nom_punct/*.parquet
  !zip -j sino_nom_punct_doc.zip data/sino_nom_punct_doc/*.parquet

  from IPython.display import FileLink
  display(FileLink('sino_nom_punct.zip'))
  display(FileLink('sino_nom_punct_doc.zip'))

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import re
from pathlib import Path


def parse_column_format(fmt_str: str) -> dict:
    """Parse '0:text,1:punct' → {0: 'text', 1: 'punct'}."""
    result = {}
    for pair in fmt_str.split(","):
        idx_str, name = pair.strip().split(":")
        result[int(idx_str)] = name.strip()
    return result


def conll_txt_to_parquet(
    input_path: str | Path,
    output_path: str | Path,
    column_format: dict,
) -> "pd.DataFrame":
    """
    Convert a single CoNLL column-format text file to Parquet.

    Parameters
    ----------
    input_path : path to the .txt file
    output_path : path to write the .parquet file
    column_format : e.g. {0: 'text', 1: 'punct'}

    Returns
    -------
    pd.DataFrame that was written
    """
    import pandas as pd

    records = []
    sentence_id = 0
    has_tokens = False

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            # Blank line = sentence boundary
            if line.isspace() or line.strip() == "":
                if has_tokens:
                    sentence_id += 1
                    has_tokens = False
                continue

            fields = re.split(r"\s+", line.strip())
            record = {"sentence_id": sentence_id}
            for col_idx, col_name in column_format.items():
                if col_idx < len(fields):
                    record[col_name] = fields[col_idx]
            records.append(record)
            has_tokens = True

    df = pd.DataFrame(records)

    # Ensure correct dtypes
    df["sentence_id"] = df["sentence_id"].astype("int32")
    for col_name in column_format.values():
        if col_name in df.columns:
            df[col_name] = df[col_name].astype("string")

    df.to_parquet(str(output_path), index=False, engine="pyarrow")
    return df


def convert_folder(data_folder: str | Path, column_format: dict) -> None:
    """Convert all .txt files in a folder to .parquet."""
    data_folder = Path(data_folder)
    if not data_folder.exists():
        print(f"  ✗ ERROR: folder not found: {data_folder}")
        return

    txt_files = sorted(data_folder.glob("*.txt"))
    if not txt_files:
        print(f"  ✗ No .txt files found in {data_folder}")
        return

    for txt_file in txt_files:
        parquet_file = txt_file.with_suffix(".parquet")
        df = conll_txt_to_parquet(txt_file, parquet_file, column_format)

        # Stats
        n_sentences = df["sentence_id"].nunique()
        n_tokens = len(df)
        txt_size = txt_file.stat().st_size
        pq_size = parquet_file.stat().st_size
        ratio = txt_size / pq_size if pq_size > 0 else 0

        print(
            f"  ✓ {txt_file.name} → {parquet_file.name}  "
            f"({n_sentences} sentences, {n_tokens} tokens, "
            f"{txt_size/1024:.0f}KB → {pq_size/1024:.0f}KB, {ratio:.1f}x smaller)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoNLL text files to Parquet format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python tools/convert_to_parquet.py data/sino_nom_punct data/sino_nom_punct_doc",
    )
    parser.add_argument(
        "data_folders",
        nargs="+",
        help="One or more data folders containing train.txt, dev.txt, test.txt",
    )
    parser.add_argument(
        "--columns",
        default="0:text,1:punct",
        help='Column format string (default: "0:text,1:punct")',
    )
    args = parser.parse_args()

    column_format = parse_column_format(args.columns)
    print(f"Column format: {column_format}")
    print()

    for folder in args.data_folders:
        print(f"Converting {folder}/")
        convert_folder(folder, column_format)
        print()

    print("Done! You can now use .parquet files with ColumnCorpus.")
    print("Tip: Remove .txt files from the data folder to avoid ambiguity,")
    print("     or keep both — ColumnCorpus auto-detects by filename.")


if __name__ == "__main__":
    main()
