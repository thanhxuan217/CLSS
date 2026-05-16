#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_minhash_index.py
──────────────────────
Build MinHash LSH index từ raw corpus hoặc file đã xử lý trước
(output của preprocess_raw_corpus.py).

Phiên bản tối ưu bộ nhớ — streaming + batch processing để chạy trên
môi trường hạn chế RAM (Kaggle ~13 GB, Colab ~12 GB).

Quy trình:
  1. Stream câu từ file processed (1 dòng/câu) HOẶC raw dir
  2. Xử lý theo batch: tạo shingles → MinHash → insert LSH
  3. Giải phóng MinHash ngay sau khi insert (LSH chỉ giữ band hash)
  4. Lưu index (pickle) + danh sách câu (text file, 1 dòng/câu)

Usage:
  # Từ file đã preprocess (khuyến nghị):
  python tools/build_minhash_index.py \
      --sentences_file data/processed/corpus_sentences.txt \
      --output_index   data/index/minhash.pkl \
      --output_sentences data/index/sentences.txt \
      --batch_size 50000

  # Từ raw directory:
  python tools/build_minhash_index.py \
      --raw_data_dir  data/raw \
      --output_index  data/index/minhash.pkl \
      --output_sentences data/index/sentences.txt
"""

import argparse
import gc
import hashlib
import logging
import os
import pickle
import re
import sys
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


def get_mem_mb() -> float:
    """Trả về RSS hiện tại (MB). Hỗ trợ Linux (Kaggle) và Windows."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Streaming data loading
# ──────────────────────────────────────────────────────────────────────────────

def count_lines(file_path: Path) -> int:
    """Đếm số dòng không rỗng trong file (nhanh, không load toàn bộ vào RAM)."""
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def iter_sentences_from_file(file_path: Path, encoding: str = "utf-8"):
    """Generator — yield từng câu từ file processed (1 dòng = 1 câu)."""
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            sent = line.strip()
            if sent:
                yield sent


def iter_sentences_from_raw_dir(
    raw_dir: Path,
    min_len: int,
    max_len: int,
    encoding: str = "utf-8",
):
    """Generator — yield từng câu hợp lệ từ raw dir (đã dedup)."""
    txt_files = sorted(raw_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Không tìm thấy file .txt nào trong: {raw_dir}")
    logger.info("Tìm thấy %d file .txt trong %s", len(txt_files), raw_dir)

    seen: set[str] = set()

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
            yield sent


# ──────────────────────────────────────────────────────────────────────────────
# Index builder — batch streaming
# ──────────────────────────────────────────────────────────────────────────────

def build_index_streaming(
    sentence_iter,
    num_perm: int,
    ngram_size: int,
    threshold: float,
    batch_size: int,
    sentences_out_path: Path,
) -> tuple:
    """
    Build MinHashLSH index bằng streaming.

    Thay vì load toàn bộ sentences vào list rồi tạo MinHash:
      - Đọc từng batch câu
      - Tạo MinHash + insert vào LSH ngay
      - Ghi câu ra file text ngay (không giữ trong RAM)
      - Giải phóng MinHash objects sau mỗi batch

    Returns:
        (lsh, total_indexed) — LSH index + số câu đã index
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    logger.info(
        "Building MinHash LSH index (streaming): "
        "num_perm=%d | ngram=%d | threshold=%.2f | batch_size=%d",
        num_perm, ngram_size, threshold, batch_size,
    )

    sentences_out_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(sentences_out_path, "w", encoding="utf-8")

    total = 0
    errors = 0
    batch = []

    try:
        for sent in sentence_iter:
            batch.append(sent)

            if len(batch) >= batch_size:
                indexed, errs = _process_batch(
                    batch, total, lsh, num_perm, ngram_size, f_out
                )
                total += indexed
                errors += errs
                batch.clear()
                gc.collect()

                mem = get_mem_mb()
                logger.info(
                    "  … %d câu đã index | RAM ≈ %.0f MB", total, mem
                )

        # Batch cuối
        if batch:
            indexed, errs = _process_batch(
                batch, total, lsh, num_perm, ngram_size, f_out
            )
            total += indexed
            errors += errs
            batch.clear()
            gc.collect()

    finally:
        f_out.close()

    if errors:
        logger.warning("Bỏ qua %d câu do lỗi khi insert vào LSH", errors)

    logger.info("Hoàn thành index: %d câu (bỏ qua %d lỗi)", total, errors)
    return lsh, total


def _process_batch(
    batch: list[str],
    start_idx: int,
    lsh: MinHashLSH,
    num_perm: int,
    ngram_size: int,
    f_out,
) -> tuple:
    """Xử lý 1 batch: tạo MinHash, insert LSH, ghi câu ra file."""
    indexed = 0
    errors = 0

    for i, sent in enumerate(batch):
        global_idx = start_idx + i
        shingles = make_shingles(sent, ngram_size)
        m = make_minhash(shingles, num_perm)
        key = str(global_idx)
        try:
            lsh.insert(key, m)
            f_out.write(sent + "\n")
            indexed += 1
        except ValueError:
            errors += 1

        # Giải phóng MinHash ngay — LSH chỉ lưu band hash, không cần giữ
        del m
        del shingles

    return indexed, errors


# ──────────────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────────────

def save_index(lsh: MinHashLSH, index_path: str) -> None:
    idx_p = Path(index_path)
    idx_p.parent.mkdir(parents=True, exist_ok=True)

    with open(idx_p, "wb") as f:
        pickle.dump(lsh, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = idx_p.stat().st_size / (1024 * 1024)
    logger.info("LSH index đã lưu -> %s (%.1f MB)", idx_p, size_mb)


def load_sentences_from_file(path: str) -> list[str]:
    """Load sentences từ text file (dùng khi query)."""
    sents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                sents.append(s)
    return sents


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

def self_test(
    lsh: MinHashLSH,
    sentences_path: str,
    num_perm: int,
    ngram_size: int,
) -> None:
    """Thử query 3 câu đầu để verify index hoạt động."""
    logger.info("─── Self-test (3 câu đầu) ───")

    # Chỉ đọc 3 dòng đầu
    test_sents = []
    with open(sentences_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                test_sents.append(s)
            if len(test_sents) >= 3:
                break

    # Đọc đủ sentences để hiển thị kết quả (load lazy, chỉ khi self-test)
    all_sents = load_sentences_from_file(sentences_path)

    for sent in test_sents:
        shingles = make_shingles(sent, ngram_size)
        m = make_minhash(shingles, num_perm)
        results = lsh.query(m)
        retrieved = [all_sents[int(r)] for r in results[:5] if int(r) < len(all_sents)]
        logger.info("Query: %s", sent[:60])
        logger.info("  → %d kết quả, ví dụ: %s", len(results), retrieved[:2])

    del all_sents
    gc.collect()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build MinHash LSH index cho Sino-Nom RAG (memory-efficient)"
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
        help="Đường dẫn lưu danh sách câu tương ứng (text file, 1 dòng/câu)",
    )

    # MinHash parameters
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Số permutation MinHash. Default: 128")
    parser.add_argument("--ngram_size", type=int, default=2,
                        help="Kích thước character n-gram. Default: 2 (bigram)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Ngưỡng Jaccard similarity cho LSH bucket. Default: 0.3")

    # Memory optimization
    parser.add_argument("--batch_size", type=int, default=50_000,
                        help="Số câu xử lý mỗi batch. Giảm nếu OOM. Default: 50000")

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

    # ── Tạo sentence iterator (streaming, không load hết vào RAM) ──
    if args.sentences_file:
        sent_path = Path(args.sentences_file)
        logger.info("Đếm số câu trong %s …", sent_path)
        total_lines = count_lines(sent_path)
        logger.info("Tổng câu: %d", total_lines)
        sentence_iter = iter_sentences_from_file(sent_path, args.encoding)
    else:
        sentence_iter = iter_sentences_from_raw_dir(
            Path(args.raw_data_dir), args.min_length, args.max_length, args.encoding
        )

    # ── Build index (streaming) ──
    out_sentences = Path(args.output_sentences)
    lsh, total_indexed = build_index_streaming(
        sentence_iter=sentence_iter,
        num_perm=args.num_perm,
        ngram_size=args.ngram_size,
        threshold=args.threshold,
        batch_size=args.batch_size,
        sentences_out_path=out_sentences,
    )

    if total_indexed == 0:
        logger.error("Không có câu nào được index. Kiểm tra lại input.")
        return

    # ── Save index ──
    save_index(lsh, args.output_index)
    logger.info("Sentence list đã lưu -> %s", out_sentences)

    # ── Self-test ──
    if not args.no_self_test:
        self_test(lsh, str(out_sentences), args.num_perm, args.ngram_size)

    mem = get_mem_mb()
    logger.info("✓ Hoàn thành! Tổng: %d câu | Peak RAM ≈ %.0f MB", total_indexed, mem)


if __name__ == "__main__":
    main()
