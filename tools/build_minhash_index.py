#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_minhash_index.py
──────────────────────
Build MinHash LSH index từ raw corpus hoặc file đã xử lý trước
(output của preprocess_raw_corpus.py).

Quy trình:
  1. Đọc câu từ raw dir (*.txt, đệ quy) HOẶC file processed (1 dòng/câu)
  2. Tạo character n-gram shingles cho mỗi câu
  3. Tạo MinHash signature (datasketch)
  4. Nạp vào MinHashLSH index
  5. Lưu index + danh sách câu vào file pickle

Usage:
  # Từ raw directory:
  python tools/build_minhash_index.py \\
      --raw_data_dir  data/raw \\
      --output_index  data/index/minhash.pkl \\
      --output_sentences data/index/sentences.pkl

  # Từ file đã preprocess:
  python tools/build_minhash_index.py \\
      --sentences_file data/processed/corpus_sentences.txt \\
      --output_index   data/index/minhash.pkl \\
      --output_sentences data/index/sentences.pkl
"""

import argparse
import hashlib
import logging
import os
import pickle
import re
import unicodedata
from pathlib import Path

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    raise ImportError(
        "Thư viện 'datasketch' chưa được cài. Chạy: pip install datasketch"
    )

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SENTENCE_DELIMITERS = re.compile(r"[。！？；\n]+")
WHITESPACE_RE = re.compile(r"\s+")


# ──────────────────────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def split_sentences_from_text(text: str) -> list[str]:
    parts = SENTENCE_DELIMITERS.split(text)
    return [p.strip() for p in parts if p.strip()]


def make_shingles(text: str, n: int = 2) -> set[str]:
    """
    Tạo character n-gram shingles (bỏ khoảng trắng trước khi shingling).
    Phù hợp cho Classical Chinese vì không có word boundaries.
    """
    text_no_space = text.replace(" ", "")
    if len(text_no_space) < n:
        return {text_no_space}
    return {text_no_space[i : i + n] for i in range(len(text_no_space) - n + 1)}


def fingerprint(sent: str) -> str:
    return hashlib.md5(sent.encode("utf-8")).hexdigest()


def make_minhash(shingles: set[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update(shingle.encode("utf-8"))
    return m


def is_valid(sent: str, min_len: int, max_len: int) -> bool:
    char_count = len(sent.replace(" ", ""))
    return min_len <= char_count <= max_len


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_from_processed_file(file_path: Path, encoding: str = "utf-8") -> list[str]:
    """Đọc file mỗi dòng 1 câu (output của preprocess_raw_corpus.py)."""
    logger.info("Đọc câu từ file: %s", file_path)
    sentences = []
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            sent = line.strip()
            if sent:
                sentences.append(sent)
    logger.info("Đọc được %d câu", len(sentences))
    return sentences


def load_from_raw_dir(
    raw_dir: Path,
    min_len: int,
    max_len: int,
    encoding: str = "utf-8",
) -> list[str]:
    """Duyệt đệ quy thư mục raw, tách câu và lọc."""
    txt_files = sorted(raw_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Không tìm thấy file .txt nào trong: {raw_dir}")
    logger.info("Tìm thấy %d file .txt trong %s", len(txt_files), raw_dir)

    seen: set[str] = set()
    sentences: list[str] = []

    for fpath in txt_files:
        try:
            raw = fpath.read_text(encoding=encoding, errors="ignore")
        except Exception as exc:
            logger.warning("Bỏ qua %s: %s", fpath, exc)
            continue

        for sent in split_sentences_from_text(normalize_text(raw)):
            if not is_valid(sent, min_len, max_len):
                continue
            fp = fingerprint(sent)
            if fp in seen:
                continue
            seen.add(fp)
            sentences.append(sent)

    logger.info("Tổng câu hợp lệ (sau dedup): %d", len(sentences))
    return sentences


# ──────────────────────────────────────────────────────────────────────────────
# Index builder
# ──────────────────────────────────────────────────────────────────────────────

def build_index(
    sentences: list[str],
    num_perm: int,
    ngram_size: int,
    threshold: float,
) -> MinHashLSH:
    """Build MinHashLSH index từ danh sách câu."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    logger.info(
        "Building MinHash LSH index: %d câu | num_perm=%d | ngram=%d | threshold=%.2f",
        len(sentences),
        num_perm,
        ngram_size,
        threshold,
    )

    errors = 0
    for idx, sent in enumerate(sentences):
        shingles = make_shingles(sent, ngram_size)
        m = make_minhash(shingles, num_perm)
        key = str(idx)
        try:
            lsh.insert(key, m)
        except ValueError:
            # Trùng key — không xảy ra vì key là index
            errors += 1

        if (idx + 1) % 50_000 == 0:
            logger.info("  … %d / %d câu đã index", idx + 1, len(sentences))

    if errors:
        logger.warning("Bỏ qua %d câu do lỗi khi insert vào LSH", errors)

    logger.info("Hoàn thành index: %d câu", len(sentences) - errors)
    return lsh


def save_artifacts(
    lsh: MinHashLSH,
    sentences: list[str],
    index_path: str,
    sentences_path: str,
) -> None:
    idx_p = Path(index_path)
    sent_p = Path(sentences_path)
    idx_p.parent.mkdir(parents=True, exist_ok=True)
    sent_p.parent.mkdir(parents=True, exist_ok=True)

    with open(idx_p, "wb") as f:
        pickle.dump(lsh, f)
    logger.info("LSH index đã lưu -> %s", idx_p)

    with open(sent_p, "wb") as f:
        pickle.dump(sentences, f)
    logger.info("Sentence list đã lưu -> %s", sent_p)


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

def self_test(lsh: MinHashLSH, sentences: list[str], num_perm: int, ngram_size: int) -> None:
    """Thử query 3 câu đầu để verify index hoạt động."""
    logger.info("─── Self-test (3 câu đầu) ───")
    for sent in sentences[:3]:
        shingles = make_shingles(sent, ngram_size)
        m = make_minhash(shingles, num_perm)
        results = lsh.query(m)
        retrieved = [sentences[int(r)] for r in results[:5]]
        logger.info("Query: %s", sent[:60])
        logger.info("  → %d kết quả, ví dụ: %s", len(results), retrieved[:2])


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build MinHash LSH index cho Sino-Nom RAG"
    )

    # Input (chọn 1 trong 2)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--raw_data_dir",
        help="Thư mục chứa file .txt raw (duyệt đệ quy, dùng nếu chưa preprocess)",
    )
    input_group.add_argument(
        "--sentences_file",
        help="File đã preprocess (1 dòng = 1 câu). Output của preprocess_raw_corpus.py",
    )

    # Output
    parser.add_argument(
        "--output_index",
        required=True,
        help="Đường dẫn lưu MinHashLSH index (pickle)",
    )
    parser.add_argument(
        "--output_sentences",
        required=True,
        help="Đường dẫn lưu danh sách câu tương ứng (pickle)",
    )

    # MinHash parameters
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Số permutation MinHash. Default: 128")
    parser.add_argument("--ngram_size", type=int, default=2,
                        help="Kích thước character n-gram. Default: 2 (bigram)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Ngưỡng Jaccard similarity cho LSH bucket. Default: 0.3")

    # Filter (chỉ dùng khi --raw_data_dir)
    parser.add_argument("--min_length", type=int, default=5,
                        help="Độ dài ký tự tối thiểu (chỉ dùng với --raw_data_dir). Default: 5")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Độ dài ký tự tối đa (chỉ dùng với --raw_data_dir). Default: 200")
    parser.add_argument("--encoding", default="utf-8",
                        help="Encoding file input. Default: utf-8")
    parser.add_argument("--no_self_test", action="store_true",
                        help="Bỏ qua self-test sau khi build")

    args = parser.parse_args()

    # Load sentences
    if args.sentences_file:
        sentences = load_from_processed_file(Path(args.sentences_file), args.encoding)
    else:
        sentences = load_from_raw_dir(
            Path(args.raw_data_dir), args.min_length, args.max_length, args.encoding
        )

    if not sentences:
        logger.error("Không có câu nào để index. Kiểm tra lại input.")
        return

    # Build index
    lsh = build_index(sentences, args.num_perm, args.ngram_size, args.threshold)

    # Save
    save_artifacts(lsh, sentences, args.output_index, args.output_sentences)

    # Self-test
    if not args.no_self_test:
        self_test(lsh, sentences, args.num_perm, args.ngram_size)

    logger.info("✓ Hoàn thành!")


if __name__ == "__main__":
    main()
