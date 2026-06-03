#!/usr/bin/env python3
"""
Convert CoNLL column-format text files (.txt) → Parquet format (.parquet).
Memory-efficient: ghi từng batch nhỏ, KHÔNG load toàn bộ dataset vào RAM.

Usage:
  python tools/convert_to_parquet.py /path/to/data_folder --output /path/to/output
  python tools/convert_to_parquet.py /path/to/data_folder --columns "0:text,1:punct"
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


def conll_txt_to_parquet(input_path, output_path, column_format: dict, chunk_size=50_000):
    """
    Convert CoNLL text → Parquet using streaming writer.
    Only keeps `chunk_size` rows in memory at a time.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build schema: sentence_id + columns from column_format
    fields = [("sentence_id", pa.int32())]
    col_names_ordered = []
    for idx in sorted(column_format.keys()):
        name = column_format[idx]
        fields.append((name, pa.string()))
        col_names_ordered.append(name)
    schema = pa.schema(fields)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    writer = pq.ParquetWriter(str(output_path), schema, compression="snappy")

    batch_rows = {col: [] for col in ["sentence_id"] + col_names_ordered}
    sentence_id = 0
    has_tokens = False
    total_tokens = 0
    total_sentences = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.isspace() or line.strip() == "":
                if has_tokens:
                    sentence_id += 1
                    total_sentences += 1
                    has_tokens = False
                continue

            fields_vals = re.split(r"\s+", line.strip())
            batch_rows["sentence_id"].append(sentence_id)
            for col_idx, col_name in column_format.items():
                val = fields_vals[col_idx] if col_idx < len(fields_vals) else ""
                batch_rows[col_name].append(val)
            has_tokens = True
            total_tokens += 1

            # Flush batch khi đủ chunk_size
            if total_tokens % chunk_size == 0:
                batch = pa.RecordBatch.from_pydict(batch_rows, schema=schema)
                writer.write_batch(batch)
                batch_rows = {col: [] for col in ["sentence_id"] + col_names_ordered}

    # Đếm câu cuối nếu file không kết thúc bằng dòng trống
    if has_tokens:
        total_sentences += 1

    # Flush batch còn lại
    if batch_rows["sentence_id"]:
        batch = pa.RecordBatch.from_pydict(batch_rows, schema=schema)
        writer.write_batch(batch)

    writer.close()
    del batch_rows

    return total_tokens, total_sentences


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

    out_dir = Path(output_folder) if output_folder else data_folder
    os.makedirs(out_dir, exist_ok=True)

    for txt_file in txt_files:
        parquet_file = out_dir / txt_file.with_suffix(".parquet").name

        n_tokens, n_sents = conll_txt_to_parquet(
            txt_file, parquet_file, column_format
        )

        txt_kb = txt_file.stat().st_size / 1024
        pq_kb = parquet_file.stat().st_size / 1024
        ratio = txt_kb / pq_kb if pq_kb > 0 else 0

        print(
            f"  ✓ {txt_file.name} → {parquet_file.name}  "
            f"({n_sents:,} sentences, {n_tokens:,} tokens, "
            f"{txt_kb:.0f}KB → {pq_kb:.0f}KB, {ratio:.1f}x smaller)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoNLL text files to Parquet (memory-efficient).",
    )
    parser.add_argument(
        "data_folder",
        help="Folder containing train.txt, dev.txt, test.txt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder (default: same as input). "
             "Dùng khi input read-only (Kaggle /kaggle/input/).",
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
    print("\nDone!")


if __name__ == "__main__":
    main()
