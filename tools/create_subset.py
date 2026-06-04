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

Lưu ý: Script xử lý cả 2 dataset (View 1 non-RAG, View 2 RAG doc) bằng cách
chạy 2 lần với input_dir/output_dir khác nhau.
"""

import argparse
import os
import random
from pathlib import Path


def read_conll_sentences(filepath: str) -> list[list[str]]:
    """Đọc file CoNLL và trả về danh sách các câu.
    Mỗi câu là list các dòng (bao gồm cả content lines).
    Câu phân cách bởi dòng trống.
    """
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)

    # Câu cuối (nếu file không kết thúc bằng dòng trống)
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def write_conll_sentences(filepath: str, sentences: list[list[str]]):
    """Ghi danh sách câu ra file CoNLL (phân cách bằng dòng trống)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences):
            for line in sentence:
                f.write(line + "\n")
            f.write("\n")  # Dòng trống phân cách câu


def create_subset(input_dir: str, output_dir: str, ratio: float, seed: int):
    """Tạo subset cho tất cả file .txt trong input_dir."""
    random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = ["train.txt", "dev.txt", "test.txt"]

    for split_name in splits:
        input_file = input_path / split_name
        output_file = output_path / split_name

        if not input_file.exists():
            print(f"  ⚠️  Skipping {split_name} — file not found")
            continue

        sentences = read_conll_sentences(str(input_file))
        total = len(sentences)

        # Sample subset
        n_subset = max(1, int(total * ratio))  # Ít nhất 1 câu
        subset = random.sample(sentences, n_subset)

        write_conll_sentences(str(output_file), subset)

        print(f"  ✅ {split_name}: {total} → {n_subset} sentences ({ratio*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Tạo subset từ dataset CoNLL-format"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Thư mục chứa dataset gốc (train.txt, dev.txt, test.txt)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Thư mục xuất dataset subset",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.25,
        help="Tỷ lệ subset (0.0 - 1.0). Default: 0.25 (25%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Creating {args.ratio*100:.0f}%% subset")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Seed:   {args.seed}")
    print(f"{'='*60}\n")

    create_subset(args.input_dir, args.output_dir, args.ratio, args.seed)

    print(f"\n✅ Done! Subset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
