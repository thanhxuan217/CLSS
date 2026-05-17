#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_minhash_index.py
──────────────────────
Build MinHash LSH index từ raw corpus hoặc file đã xử lý trước
(output của preprocess_raw_corpus.py).

★ DISK-BASED — dùng SQLite thay vì RAM.
  Index + sentences nằm hoàn toàn trên disk.
  RAM chỉ dùng SQLite page cache (~64 MB) bất kể corpus size.
  → 46M câu, 100M câu, ... đều chạy được.

Quy trình:
  1. Stream câu từ file processed HOẶC raw dir
  2. Mỗi batch: tạo shingles → MinHash → insert vào SQLite
  3. MinHash objects giải phóng ngay, SQLite ghi xuống disk
  4. Sau khi insert xong: tạo index trên SQLite (1 lần)

Output: 1 file .db duy nhất chứa cả LSH index + sentences.

Usage:
  python tools/build_minhash_index.py \\
      --sentences_file data/processed/corpus_sentences.txt \\
      --output_db      data/index/minhash.db \\
      --num_perm       128 \\
      --ngram_size     2 \\
      --threshold      0.3
"""

import argparse
import gc
import hashlib
import logging
import re
import unicodedata
from pathlib import Path

try:
    from datasketch import MinHash
except ImportError:
    raise ImportError(
        "Thư viện 'datasketch' chưa được cài. Chạy: pip install datasketch"
    )

from sqlite_lsh import SqliteMinHashLSH

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


def iter_sentences_from_file(
    file_path: Path,
    encoding: str = "utf-8",
    max_sentences: int = 0,
):
    """Generator — yield từng câu từ file processed (1 dòng = 1 câu).

    Args:
        max_sentences: Dừng sau khi yield đủ số câu này. 0 = không giới hạn.
    """
    count = 0
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            sent = line.strip()
            if sent:
                yield sent
                count += 1
                if max_sentences > 0 and count >= max_sentences:
                    return


def iter_sentences_from_raw_dir(
    raw_dir: Path,
    min_len: int,
    max_len: int,
    encoding: str = "utf-8",
    max_sentences: int = 0,
):
    """Generator — yield từng câu hợp lệ từ raw dir (đã dedup).

    Args:
        max_sentences: Dừng sau khi yield đủ số câu này. 0 = không giới hạn.
    """
    txt_files = sorted(raw_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Không tìm thấy file .txt nào trong: {raw_dir}")
    logger.info("Tìm thấy %d file .txt trong %s", len(txt_files), raw_dir)

    seen: set[str] = set()
    count = 0

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
            count += 1
            if max_sentences > 0 and count >= max_sentences:
                return


# ──────────────────────────────────────────────────────────────────────────────
# Index builder — streaming vào SQLite
# ──────────────────────────────────────────────────────────────────────────────

def build_index(
    sentence_iter,
    num_perm: int,
    ngram_size: int,
    threshold: float,
    batch_size: int,
    db_path: Path,
    max_db_size_gb: float = 0.0,
) -> int:
    """
    Build MinHash LSH index streaming vào SQLite.

    Mỗi batch:
      - Tạo MinHash cho từng câu
      - Insert vào SqliteMinHashLSH (ghi xuống disk)
      - Giải phóng MinHash objects
      - gc.collect()

    RAM không tăng theo số câu — chỉ SQLite page cache.

    Returns:
        total_indexed — số câu đã index
    """
    logger.info(
        "Building MinHash LSH index (SQLite disk-based): "
        "num_perm=%d | ngram=%d | threshold=%.2f | batch_size=%d",
        num_perm, ngram_size, threshold, batch_size,
    )

    lsh = SqliteMinHashLSH.create(db_path, threshold=threshold, num_perm=num_perm)
    logger.info("  LSH params: b=%d bands, r=%d rows (b×r=%d ≤ num_perm=%d)",
                lsh.b, lsh.r, lsh.b * lsh.r, num_perm)

    total = 0
    errors = 0
    batch_count = 0

    for sent in sentence_iter:
        shingles = make_shingles(sent, ngram_size)
        m = make_minhash(shingles, num_perm)
        try:
            lsh.insert(str(total), m, sentence_text=sent)
            total += 1
        except Exception:
            errors += 1
        del m, shingles

        # Flush mỗi batch
        if total % batch_size == 0:
            lsh.flush()
            gc.collect()
            batch_count += 1
            mem = get_mem_mb()
            
            db_size_gb = db_path.stat().st_size / (1024**3)
            logger.info("  … %d câu đã index | RAM ≈ %.0f MB | DB size ≈ %.2f GB", total, mem, db_size_gb)
            
            if max_db_size_gb > 0 and db_size_gb >= max_db_size_gb:
                logger.warning("Đã đạt giới hạn dung lượng DB (%.2f GB >= %.2f GB). Dừng index sớm.", db_size_gb, max_db_size_gb)
                break

    # Flush phần còn lại
    lsh.flush()

    if errors:
        logger.warning("Bỏ qua %d câu do lỗi khi insert", errors)

    # Tạo index trên SQLite (chỉ 1 lần, sau khi insert xong)
    logger.info("  Đang tạo SQLite index (có thể mất vài phút cho corpus lớn)...")
    lsh.finalize()

    db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
    logger.info("  Index DB: %.1f MB | %d câu", db_size_mb, total)

    lsh.close()
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

def self_test(db_path: str, num_perm: int, ngram_size: int) -> None:
    """Thử query 3 câu đầu để verify index hoạt động."""
    logger.info("─── Self-test (3 câu đầu) ───")

    lsh = SqliteMinHashLSH.open(db_path)

    # Lấy 3 câu đầu
    cur = lsh.conn.execute("SELECT id, text FROM sentences ORDER BY id LIMIT 3")
    test_rows = cur.fetchall()

    for row_id, sent in test_rows:
        shingles = make_shingles(sent, ngram_size)
        m = make_minhash(shingles, num_perm)
        results = lsh.query(m)
        # Lấy text cho top 5 results
        top_keys = results[:5]
        texts = lsh.get_sentences_batch(top_keys)
        retrieved = [texts[k] for k in top_keys if k in texts]
        logger.info("Query: %s", sent[:60])
        logger.info("  → %d kết quả, ví dụ: %s", len(results), retrieved[:2])

    lsh.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build MinHash LSH index cho Sino-Nom RAG "
                    "(SQLite disk-based, không giới hạn RAM)"
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
        "--output_db",
        required=True,
        help="Đường dẫn lưu SQLite database (chứa cả LSH index + sentences)",
    )

    # MinHash parameters
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Số permutation MinHash. Default: 128")
    parser.add_argument("--ngram_size", type=int, default=2,
                        help="Kích thước character n-gram. Default: 2 (bigram)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Ngưỡng Jaccard similarity cho LSH bucket. Default: 0.3")

    parser.add_argument("--batch_size", type=int, default=50_000,
                        help="Số câu mỗi batch flush xuống SQLite. Default: 50000")
    parser.add_argument("--max_sentences", type=int, default=0,
                        help="Số câu tối đa để index. 0 = không giới hạn. Default: 0")
    parser.add_argument("--max_db_size_gb", type=float, default=20.0,
                        help="Giới hạn dung lượng tối đa của file SQLite (GB). 0 = không giới hạn. Default: 20.0")

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

    # ── Tạo sentence iterator ──
    max_s = args.max_sentences
    if args.sentences_file:
        sent_path = Path(args.sentences_file)
        logger.info("Đếm số câu trong %s …", sent_path)
        total_lines = count_lines(sent_path)
        logger.info("Tổng câu trong file: %d", total_lines)
        if max_s > 0:
            logger.info("Giới hạn: %d câu (--max_sentences)", max_s)
        sentence_iter = iter_sentences_from_file(
            sent_path, args.encoding, max_sentences=max_s
        )
    else:
        if max_s > 0:
            logger.info("Giới hạn: %d câu (--max_sentences)", max_s)
        sentence_iter = iter_sentences_from_raw_dir(
            Path(args.raw_data_dir), args.min_length, args.max_length,
            args.encoding, max_sentences=max_s,
        )

    # ── Build index ──
    db_path = Path(args.output_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    total_indexed = build_index(
        sentence_iter=sentence_iter,
        num_perm=args.num_perm,
        ngram_size=args.ngram_size,
        threshold=args.threshold,
        batch_size=args.batch_size,
        db_path=db_path,
        max_db_size_gb=args.max_db_size_gb,
    )

    if total_indexed == 0:
        logger.error("Không có câu nào được index. Kiểm tra lại input.")
        return

    # ── Self-test ──
    if not args.no_self_test:
        self_test(str(db_path), args.num_perm, args.ngram_size)

    mem = get_mem_mb()
    logger.info("✓ Hoàn thành! Tổng: %d câu | RAM ≈ %.0f MB", total_indexed, mem)


if __name__ == "__main__":
    main()
