#!/usr/bin/env python3
"""
Convert CoNLL column-format text files (.txt) → Parquet format (.parquet).

Parquet files are ~5-10x smaller than plain text, faster to read, and ideal
for transferring datasets between environments (e.g., Kaggle → Slurm server).

Parquet schema:
    sentence_id (int)   — groups tokens into sentences (replaces blank-line separators)
    text        (str)   — token text (column 0)
    <label>     (str)   — label column(s) (column 1, 2, ...)

═══════════════════════════════════════════════════════════════════════════════
Usage (CLI):
═══════════════════════════════════════════════════════════════════════════════

  # Convert in-place (output cùng thư mục với input)
  python tools/convert_to_parquet.py data/sino_nom_punct

  # Convert ra thư mục khác (dùng khi input là read-only, ví dụ Kaggle)
  python tools/convert_to_parquet.py /kaggle/input/.../sino_nom_punct --output /kaggle/working/sino_nom_punct

  # Custom column format (default: "0:text,1:punct")
  python tools/convert_to_parquet.py data/my_ner_data --columns "0:text,1:ner"

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
    input_path,
    output_path,
    column_format: dict,
):
    """
    Convert a single CoNLL column-format text file to Parquet.

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
    df["sentence_id"] = df["sentence_id"].astype("int32")
    for col_name in column_format.values():
        if col_name in df.columns:
            df[col_name] = df[col_name].astype("string")

    os.makedirs(Path(output_path).parent, exist_ok=True)
    df.to_parquet(str(output_path), index=False, engine="pyarrow")
    return df


def convert_folder(data_folder, column_format: dict, output_folder=None):
    """Convert all .txt files in a folder to .parquet."""
    data_folder = Path(data_folder)
    if not data_folder.exists():
        print(f"  ✗ ERROR: folder not found: {data_folder}")
        return

    txt_files = sorted(data_folder.glob("*.txt"))
    if not txt_files:
        print(f"  ✗ No .txt files found in {data_folder}")
        return

    if output_folder is not None:
        out_dir = Path(output_folder)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = data_folder

    for txt_file in txt_files:
        parquet_file = out_dir / txt_file.with_suffix(".parquet").name
        df = conll_txt_to_parquet(txt_file, parquet_file, column_format)

        n_sentences = df["sentence_id"].nunique()
        n_tokens = len(df)
        txt_size = txt_file.stat().st_size
        pq_size = parquet_file.stat().st_size
        ratio = txt_size / pq_size if pq_size > 0 else 0

        print(
            f"  ✓ {txt_file.name} → {parquet_file.name}  "
            f"({n_sentences:,} sentences, {n_tokens:,} tokens, "
            f"{txt_size/1024:.0f}KB → {pq_size/1024:.0f}KB, {ratio:.1f}x smaller)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoNLL text files to Parquet format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_folder",
        help="Data folder containing train.txt, dev.txt, test.txt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder for .parquet files (default: same as input). "
             "Use this when input folder is read-only (e.g., Kaggle /kaggle/input/).",
    )
    parser.add_argument(
        "--columns",
        default="0:text,1:punct",
        help='Column format (default: "0:text,1:punct")',
    )
    args = parser.parse_args()

    column_format = parse_column_format(args.columns)
    print(f"Column format: {column_format}")
    print(f"Input:  {args.data_folder}")
    print(f"Output: {args.output or args.data_folder}")
    print()

    convert_folder(args.data_folder, column_format, args.output)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
