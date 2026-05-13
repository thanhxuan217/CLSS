#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_raw_corpus.py
────────────────────────
Tiền xử lý raw corpus cho Sino-Nom / Classical Chinese trước khi build
MinHash LSH index.

Quy trình:
  1. Đọc đệ quy tất cả file .txt trong thư mục raw
  2. Chuẩn hóa Unicode (NFC)
  3. Tách câu dựa trên dấu câu (。，：、；？！ và Latin .,;:?!)
  4. Loại bỏ câu quá ngắn / quá dài
  5. Deduplicate bằng exact hash
  6. Xuất file processed + thống kê

Usage:
  python tools/preprocess_raw_corpus.py \\
      --raw_data_dir data/raw \\
      --output_file  data/processed/corpus_sentences.txt \\
      [--min_length 5] [--max_length 200] [--encoding utf-8]
"""

import argparse
import hashlib
import logging
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Dấu câu dùng để tách câu (Sino-Nom + Latin)
SENTENCE_DELIMITERS = re.compile(r"[。！？；\n]+")

# Ký tự để loại bỏ / chuẩn hoá khoảng trắng
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC và khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """
    Tách câu theo dấu câu.
    Trả về list các câu đã strip whitespace, không rỗng.
    """
    parts = SENTENCE_DELIMITERS.split(text)
    sentences = []
    for part in parts:
        sent = part.strip()
        if sent:
            sentences.append(sent)
    return sentences


def is_valid_sentence(sent: str, min_len: int, max_len: int) -> bool:
    """Kiểm tra độ dài câu (tính theo ký tự, bỏ khoảng trắng)."""
    char_count = len(sent.replace(" ", ""))
    return min_len <= char_count <= max_len


def fingerprint(sent: str) -> str:
    """Hash MD5 để deduplicate."""
    return hashlib.md5(sent.encode("utf-8")).hexdigest()


def iter_txt_files(directory: Path):
    """Duyệt đệ quy tất cả file .txt trong thư mục."""
    for path in sorted(directory.rglob("*.txt")):
        yield path


def process_corpus(
    raw_data_dir: str,
    output_file: str,
    min_length: int = 5,
    max_length: int = 200,
    encoding: str = "utf-8",
) -> None:
    raw_dir = Path(raw_data_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    txt_files = list(iter_txt_files(raw_dir))
    if not txt_files:
        logger.error("Không tìm thấy file .txt nào trong: %s", raw_dir)
        return

    logger.info("Tìm thấy %d file .txt", len(txt_files))

    seen_hashes: set[str] = set()
    all_sentences: list[str] = []

    stats = Counter(
        total_raw=0,
        too_short=0,
        too_long=0,
        duplicates=0,
        kept=0,
    )
    length_distribution: list[int] = []

    for file_path in txt_files:
        logger.info("Đang xử lý: %s", file_path.name)
        try:
            raw_text = file_path.read_text(encoding=encoding, errors="ignore")
        except Exception as exc:
            logger.warning("Không đọc được %s: %s", file_path, exc)
            continue

        normalized = normalize_text(raw_text)
        sentences = split_sentences(normalized)

        for sent in sentences:
            stats["total_raw"] += 1
            char_len = len(sent.replace(" ", ""))

            if char_len < min_length:
                stats["too_short"] += 1
                continue
            if char_len > max_length:
                stats["too_long"] += 1
                continue

            fp = fingerprint(sent)
            if fp in seen_hashes:
                stats["duplicates"] += 1
                continue

            seen_hashes.add(fp)
            all_sentences.append(sent)
            length_distribution.append(char_len)
            stats["kept"] += 1

    # Ghi output
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(sent + "\n")

    # In thống kê
    logger.info("=" * 60)
    logger.info("THỐNG KÊ XỬ LÝ CORPUS")
    logger.info("=" * 60)
    logger.info("Số file xử lý       : %d", len(txt_files))
    logger.info("Tổng câu raw        : %d", stats["total_raw"])
    logger.info("Quá ngắn (< %d c)   : %d", min_length, stats["too_short"])
    logger.info("Quá dài  (> %d c)   : %d", max_length, stats["too_long"])
    logger.info("Trùng lặp           : %d", stats["duplicates"])
    logger.info("Câu giữ lại         : %d", stats["kept"])
    if length_distribution:
        avg_len = sum(length_distribution) / len(length_distribution)
        logger.info("Độ dài TB (char)    : %.1f", avg_len)
        logger.info("Độ dài min/max      : %d / %d",
                    min(length_distribution), max(length_distribution))
        # Phân phối theo dải
        buckets = [(0, 20), (20, 50), (50, 100), (100, 150), (150, 200)]
        logger.info("Phân phối độ dài:")
        for lo, hi in buckets:
            count = sum(1 for l in length_distribution if lo <= l < hi)
            logger.info("  [%3d–%3d): %d câu", lo, hi, count)
    logger.info("Output -> %s", out_path)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Tiền xử lý raw corpus Sino-Nom cho MinHash RAG"
    )
    parser.add_argument(
        "--raw_data_dir",
        required=True,
        help="Thư mục gốc chứa các file .txt raw (duyệt đệ quy)",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="File output: mỗi dòng 1 câu đã xử lý",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Độ dài tối thiểu tính bằng ký tự (bỏ khoảng trắng). Default: 5",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Độ dài tối đa tính bằng ký tự (bỏ khoảng trắng). Default: 200",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding của file input. Default: utf-8",
    )
    args = parser.parse_args()

    process_corpus(
        raw_data_dir=args.raw_data_dir,
        output_file=args.output_file,
        min_length=args.min_length,
        max_length=args.max_length,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
