#!/usr/bin/env python3
"""
create_subset.py — Tạo subset từ dataset CoNLL-format (cho cả View 1 và View 2).

Cách hoạt động:
  - Đọc file CoNLL (mỗi câu phân cách bằng dòng trống)
  - Random sample một tỷ lệ % câu
  - Ghi ra thư mục output

Usage:
    python tools/create_subset.py \
        --input_dir  /path/to/full_dataset \
        --output_dir /path/to/subset_dataset \
        --ratio 0.25 \
        --seed 42
"""

import argparse
import os
import random
import sys
from pathlib import Path


def read_conll_sentences_fast(filepath: str) -> list:
    """Đọc file CoNLL nhanh — chỉ lưu byte offsets cho từng câu,
    rồi sample và đọc lại các câu cần thiết.
    Trả về list[str] — mỗi phần tử là toàn bộ text của 1 câu (gồm cả newlines).
    """
    sentences = []
    current_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "":
                if current_lines:
                    sentences.append("".join(current_lines))
                    current_lines = []
            else:
                current_lines.append(line)

    # Câu cuối (nếu file không kết thúc bằng dòng trống)
    if current_lines:
        sentences.append("".join(current_lines))

    return sentences


def create_subset(input_dir: str, output_dir: str, ratio: float, seed: int):
    """Tạo subset cho tất cả file .txt trong input_dir."""
    random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tìm tất cả file .txt trong input (train, dev, test)
    splits = ["train.txt", "dev.txt", "test.txt"]

    found_any = False
    for split_name in splits:
        input_file = input_path / split_name
        output_file = output_path / split_name

        if not input_file.exists():
            print(f"  ⚠️  Skipping {split_name} — file not found at {input_file}")
            continue

        found_any = True
        file_size = input_file.stat().st_size
        print(f"  Reading {split_name} ({file_size / 1024 / 1024:.1f} MB)...", flush=True)

        sentences = read_conll_sentences_fast(str(input_file))
        total = len(sentences)

        # Sample subset
        n_subset = max(1, int(total * ratio))  # Ít nhất 1 câu
        indices = random.sample(range(total), n_subset)
        indices.sort()  # Giữ thứ tự gốc

        # Ghi file
        print(f"  Writing {n_subset}/{total} sentences to {output_file}...", flush=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for idx in indices:
                f.write(sentences[idx])
                f.write("\n")  # Dòng trống phân cách câu

        # Verify
        actual_size = output_file.stat().st_size
        print(f"  ✅ {split_name}: {total} → {n_subset} sentences ({ratio*100:.0f}%) "
              f"[{actual_size / 1024:.1f} KB]", flush=True)

    if not found_any:
        print(f"\n  ❌ ERROR: No .txt files found in {input_dir}")
        print(f"  Listing directory contents:")
        if input_path.exists():
            for item in sorted(input_path.iterdir()):
                print(f"    {item.name} ({'dir' if item.is_dir() else f'{item.stat().st_size} bytes'})")
        else:
            print(f"    Directory does not exist!")
        sys.exit(1)

    # Verify output
    print(f"\n  📁 Output directory ({output_path}):")
    for item in sorted(output_path.iterdir()):
        print(f"    {item.name} ({item.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Tạo subset từ dataset CoNLL-format"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Thư mục chứa dataset gốc (train.txt, dev.txt, test.txt)")
    parser.add_argument("--output_dir", required=True,
                        help="Thư mục xuất dataset subset")
    parser.add_argument("--ratio", type=float, default=0.25,
                        help="Tỷ lệ subset (0.0 - 1.0). Default: 0.25 (25%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Default: 42")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Creating {args.ratio*100:.0f}%% subset")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Seed:   {args.seed}")
    print(f"{'='*60}\n", flush=True)

    create_subset(args.input_dir, args.output_dir, args.ratio, args.seed)

    print(f"\n✅ Done! Subset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
