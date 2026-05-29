#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_doc_dataset_parallel.py
────────────────────────────────
Phiên bản SONG SONG của generate_doc_dataset.py.

Tối ưu:
  1. multiprocessing.Pool — mỗi worker mở SQLite connection riêng (read-only)
  2. Batch chunk processing — giảm overhead IPC
  3. Resume support — nếu bị interrupt, chạy lại sẽ tiếp tục từ câu đã xử lý
  4. Tăng SQLite cache cho mỗi worker

Usage:
  python tools/generate_doc_dataset_parallel.py \
      --input_dir      data/sino_nom_punct \
      --output_dir     data/sino_nom_punct_doc \
      --index_db       data/index/minhash.db \
      --top_k          5 \
      --min_jaccard    0.3 \
      --max_jaccard    0.95 \
      --num_workers    4 \
      --chunk_size     500
"""

import argparse
import gc
import logging
import os
import tempfile
import shutil
from multiprocessing import Pool, cpu_count, current_process
from pathlib import Path
from functools import partial

try:
    from datasketch import MinHash
except ImportError:
    raise ImportError("Cài datasketch: pip install datasketch")

from sqlite_lsh import SqliteMinHashLSH

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CoNLL I/O
# ──────────────────────────────────────────────────────────────────────────────

Sentence = list[tuple[str, str]]


def read_all_conll(file_path: Path, text_col: int = 0, tag_col: int = 1) -> list[Sentence]:
    """Đọc TOÀN BỘ file CoNLL vào memory (cần cho parallel processing)."""
    sentences: list[Sentence] = []
    current: Sentence = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("-DOCSTART-"):
                continue

            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
                continue

            parts = line.split()
            if len(parts) <= max(text_col, tag_col):
                continue

            token = parts[text_col]
            tag = parts[tag_col]
            current.append((token, tag))

    if current:
        sentences.append(current)

    return sentences


def write_conll_doc_single(
    f,
    sent: Sentence,
    retrieved_sents: list[str],
    eos_tag: str = "S-X",
    retrieved_tag: str = "S-X",
) -> None:
    """Ghi 1 câu dataset *_doc."""
    f.write("-DOCSTART- O\n\n")

    for token, tag in sent:
        f.write(f"{token}\t{tag}\n")

    if retrieved_sents:
        f.write(f"<EOS>\t{eos_tag}\n")
        for ret_sent in retrieved_sents:
            for char_token in ret_sent.replace(" ", ""):
                f.write(f"{char_token}\t{retrieved_tag}\n")

    f.write("\n")


# ──────────────────────────────────────────────────────────────────────────────
# MinHash helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_shingles(text: str, n: int = 2) -> set[str]:
    text_ns = text.replace(" ", "")
    if len(text_ns) < n:
        return {text_ns}
    return {text_ns[i : i + n] for i in range(len(text_ns) - n + 1)}


def make_minhash(shingles: set[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def sentence_text(sent: Sentence) -> str:
    return "".join(tok for tok, _ in sent)


# ──────────────────────────────────────────────────────────────────────────────
# Worker init / process
# ──────────────────────────────────────────────────────────────────────────────

# Global per-worker state
_worker_lsh = None
_worker_params = None


def _worker_init(db_path: str, num_perm: int, ngram_size: int,
                 top_k: int, min_jaccard: float, max_jaccard: float,
                 cache_size_mb: int):
    """Khởi tạo SQLite connection cho mỗi worker process."""
    global _worker_lsh, _worker_params
    _worker_lsh = SqliteMinHashLSH.open(db_path, cache_size_mb=cache_size_mb)
    _worker_params = {
        "num_perm": num_perm,
        "ngram_size": ngram_size,
        "top_k": top_k,
        "min_jaccard": min_jaccard,
        "max_jaccard": max_jaccard,
    }


def _retrieve_for_sentence_worker(query_text: str) -> list[str]:
    """Query LSH cho 1 câu — chạy trong worker process."""
    global _worker_lsh, _worker_params

    p = _worker_params
    shingles_q = make_shingles(query_text, p["ngram_size"])
    mh_q = make_minhash(shingles_q, p["num_perm"])

    candidate_keys = _worker_lsh.query(mh_q)
    if not candidate_keys:
        return []

    cand_texts = _worker_lsh.get_sentences_batch(candidate_keys)

    scored: list[tuple[float, str]] = []
    for key in candidate_keys:
        cand_text = cand_texts.get(key)
        if cand_text is None or cand_text == query_text:
            continue

        shingles_c = make_shingles(cand_text, p["ngram_size"])
        mh_c = make_minhash(shingles_c, p["num_perm"])
        j = mh_q.jaccard(mh_c)

        if j < p["min_jaccard"] or j > p["max_jaccard"]:
            continue

        scored.append((j, cand_text))

    scored.sort(key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    result: list[str] = []
    for _, text in scored:
        if text not in seen:
            seen.add(text)
            result.append(text)
        if len(result) >= p["top_k"]:
            break

    return result


def _process_chunk(chunk: list[tuple[int, str]]) -> list[tuple[int, list[str]]]:
    """
    Xử lý 1 chunk gồm các (global_idx, query_text).
    Trả về list of (global_idx, retrieved_texts).
    """
    results = []
    for idx, q_text in chunk:
        retrieved = _retrieve_for_sentence_worker(q_text)
        results.append((idx, retrieved))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_split_parallel(
    input_file: Path,
    output_file: Path,
    db_path: str,
    num_perm: int,
    ngram_size: int,
    top_k: int,
    min_jaccard: float,
    max_jaccard: float,
    text_col: int,
    tag_col: int,
    num_workers: int,
    chunk_size: int,
    cache_size_mb: int,
) -> None:
    logger.info("Xử lý: %s → %s (workers=%d, chunk=%d)",
                input_file.name, output_file.name, num_workers, chunk_size)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Đọc toàn bộ câu vào memory ──
    logger.info("  Đang đọc file input...")
    all_sentences = read_all_conll(input_file, text_col, tag_col)
    total = len(all_sentences)
    logger.info("  Tổng: %d câu", total)

    # ── Resume support: kiểm tra file output đã có bao nhiêu câu ──
    resume_from = 0
    if output_file.exists():
        # Đếm số -DOCSTART- đã ghi = số câu đã xử lý
        with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("-DOCSTART-"):
                    resume_from += 1
        if resume_from > 0:
            logger.info("  ⚡ Resume: bỏ qua %d câu đã xử lý, tiếp tục từ câu %d",
                        resume_from, resume_from)

    if resume_from >= total:
        logger.info("  File đã hoàn thành, bỏ qua.")
        return

    # ── Chuẩn bị chunks cho parallel processing ──
    query_items: list[tuple[int, str]] = []
    for idx in range(resume_from, total):
        q_text = sentence_text(all_sentences[idx])
        query_items.append((idx, q_text))

    # Chia thành chunks
    chunks: list[list[tuple[int, str]]] = []
    for i in range(0, len(query_items), chunk_size):
        chunks.append(query_items[i : i + chunk_size])

    logger.info("  Cần xử lý: %d câu, chia thành %d chunks", len(query_items), len(chunks))

    # ── Xử lý song song ──
    # Mở file ở mode append (cho resume)
    open_mode = "a" if resume_from > 0 else "w"

    total_retrieved = 0
    no_result = 0
    processed = resume_from

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(db_path, num_perm, ngram_size, top_k, min_jaccard, max_jaccard, cache_size_mb),
    ) as pool:
        with open(output_file, open_mode, encoding="utf-8") as f_out:
            # imap_unordered cho throughput tốt hơn, nhưng ta cần giữ thứ tự
            # → dùng imap (ordered) để đảm bảo output đúng thứ tự
            for chunk_results in pool.imap(_process_chunk, chunks):
                # chunk_results: list of (global_idx, retrieved_texts)
                for global_idx, retrieved in chunk_results:
                    sent = all_sentences[global_idx]
                    write_conll_doc_single(f_out, sent, retrieved)

                    total_retrieved += len(retrieved)
                    if not retrieved:
                        no_result += 1
                    processed += 1

                # Flush sau mỗi chunk
                f_out.flush()

                if processed % 5000 < chunk_size:
                    avg = total_retrieved / max(processed - resume_from, 1)
                    logger.info(
                        "  … %d/%d câu đã xử lý (%.1f%%) | avg retrieved: %.1f",
                        processed, total, 100.0 * processed / total, avg,
                    )

    avg = total_retrieved / max(processed - resume_from, 1)
    logger.info(
        "  Xong: %d câu | avg retrieved=%.2f/câu | %d câu không có retrieved",
        processed, avg, no_result,
    )
    logger.info("  Đã lưu: %s", output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Sinh dataset *_doc dùng MinHash LSH retrieval — PARALLEL version"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Thư mục dataset gốc (CoNLL column format)")
    parser.add_argument("--output_dir", required=True,
                        help="Thư mục xuất dataset *_doc")
    parser.add_argument("--index_db", required=True,
                        help="SQLite database chứa LSH index (.db)")

    parser.add_argument("--top_k", type=int, default=5,
                        help="Số câu retrieved tối đa. Default: 5")
    parser.add_argument("--max_jaccard", type=float, default=0.95,
                        help="Loại câu có Jaccard > ngưỡng này. Default: 0.95")
    parser.add_argument("--min_jaccard", type=float, default=0.1,
                        help="Loại câu có Jaccard < ngưỡng này. Default: 0.1")
    parser.add_argument("--num_perm", type=int, default=0,
                        help="Override num_perm (0 = dùng từ DB metadata). Default: 0")
    parser.add_argument("--ngram_size", type=int, default=2,
                        help="Kích thước character n-gram. Default: 2")

    parser.add_argument("--text_col", type=int, default=0,
                        help="Cột text trong CoNLL file (0-indexed). Default: 0")
    parser.add_argument("--tag_col", type=int, default=1,
                        help="Cột NER tag trong CoNLL file (0-indexed). Default: 1")

    parser.add_argument("--splits", nargs="+",
                        default=["train.txt", "dev.txt", "test.txt"],
                        help="Tên các file split. Default: train.txt dev.txt test.txt")

    # ── Parallel-specific args ──
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Số worker processes (0 = auto = cpu_count). Default: 0")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Số câu mỗi chunk gửi cho worker. Default: 500")
    parser.add_argument("--cache_size_mb", type=int, default=128,
                        help="SQLite cache size cho mỗi worker (MB). Default: 128")

    args = parser.parse_args()

    # ── Determine worker count ──
    if args.num_workers <= 0:
        args.num_workers = max(1, cpu_count() - 1)
    logger.info("Parallel workers: %d", args.num_workers)

    # ── Đọc metadata từ DB ──
    logger.info("Đang đọc metadata từ: %s", args.index_db)
    lsh_tmp = SqliteMinHashLSH.open(args.index_db)
    total_in_index = lsh_tmp.total_sentences()
    num_perm = args.num_perm if args.num_perm > 0 else lsh_tmp.num_perm
    logger.info("Index: %d câu | num_perm=%d | b=%d | r=%d",
                total_in_index, num_perm, lsh_tmp.b, lsh_tmp.r)
    lsh_tmp.close()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        in_file = in_dir / split_name
        out_file = out_dir / split_name

        if not in_file.exists():
            logger.warning("Bỏ qua (không tồn tại): %s", in_file)
            continue

        process_split_parallel(
            input_file=in_file,
            output_file=out_file,
            db_path=args.index_db,
            num_perm=num_perm,
            ngram_size=args.ngram_size,
            top_k=args.top_k,
            min_jaccard=args.min_jaccard,
            max_jaccard=args.max_jaccard,
            text_col=args.text_col,
            tag_col=args.tag_col,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            cache_size_mb=args.cache_size_mb,
        )

    logger.info("✓ Hoàn thành! Dataset *_doc đã lưu tại: %s", out_dir)


if __name__ == "__main__":
    main()
