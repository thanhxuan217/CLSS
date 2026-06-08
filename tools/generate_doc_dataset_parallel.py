#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_doc_dataset_parallel.py
────────────────────────────────
Phiên bản SONG SONG + STREAMING của generate_doc_dataset.py.

Tối ưu:
  1. STREAMING — không load toàn bộ file vào RAM (fix OOM trên Kaggle 13GB)
  2. multiprocessing.Pool — mỗi worker mở SQLite connection riêng (read-only)
  3. Batch chunk processing — giảm overhead IPC
  4. Resume support — nếu bị interrupt, chạy lại sẽ tiếp tục từ câu đã xử lý

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
import json
import logging
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

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
# CoNLL I/O — STREAMING (không load toàn bộ vào RAM)
# ──────────────────────────────────────────────────────────────────────────────

Sentence = list[tuple[str, str]]


def iter_conll(file_path: Path, text_col: int = 0, tag_col: int = 1):
    """Đọc file CoNLL streaming, yield từng sentence."""
    current: Sentence = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("-DOCSTART-"):
                continue

            if line.strip() == "":
                if current:
                    yield current
                    current = []
                continue

            parts = line.split()
            if len(parts) <= max(text_col, tag_col):
                continue

            token = parts[text_col]
            tag = parts[tag_col]
            current.append((token, tag))

    if current:
        yield current


def iter_conll_chunks(
    file_path: Path,
    text_col: int,
    tag_col: int,
    chunk_size: int,
    skip: int = 0,
):
    """
    Stream file CoNLL theo chunks.
    Skip `skip` câu đầu (cho resume), rồi yield từng chunk.
    Mỗi chunk = list[Sentence], tối đa chunk_size câu.
    RAM chỉ giữ 1 chunk tại mỗi thời điểm.
    """
    chunk: list[Sentence] = []
    count = 0

    for sent in iter_conll(file_path, text_col, tag_col):
        if count < skip:
            count += 1
            continue

        chunk.append(sent)
        count += 1

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def count_sentences_in_file(file_path: Path, text_col: int = 0, tag_col: int = 1) -> int:
    """Đếm nhanh số câu trong file CoNLL (streaming, không giữ data)."""
    count = 0
    for _ in iter_conll(file_path, text_col, tag_col):
        count += 1
    return count


def _format_conll_doc_single(
    sent: Sentence,
    retrieved_sents: list[str],
    max_seq_len: int = 500,
    eos_tag: str = "S-X",
    retrieved_tag: str = "S-X",
) -> str:
    """Format 1 câu dataset *_doc thành string."""
    parts: list[str] = ["-DOCSTART- O\n\n"]

    current_len = 0
    for token, tag in sent:
        parts.append(f"{token}\t{tag}\n")
        current_len += 1

    if retrieved_sents:
        if current_len < max_seq_len:
            parts.append(f"<EOS>\t{eos_tag}\n")
            current_len += 1
            for ret_sent in retrieved_sents:
                for char_token in ret_sent.replace(" ", ""):
                    if current_len >= max_seq_len:
                        break
                    parts.append(f"{char_token}\t{retrieved_tag}\n")
                    current_len += 1
                if current_len >= max_seq_len:
                    break

    parts.append("\n")
    return "".join(parts)


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
# Worker init / process — mỗi worker mở SQLite connection riêng
# ──────────────────────────────────────────────────────────────────────────────

# Global per-worker state (set by _worker_init)
_worker_lsh = None
_worker_params = None


def _worker_init(
    db_path: str, num_perm: int, ngram_size: int,
    top_k: int, min_jaccard: float, max_jaccard: float,
    cache_size_mb: int,
):
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


def _retrieve_single(query_text: str) -> list[str]:
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


def _process_chunk(chunk: list[Sentence]) -> list[tuple[Sentence, list[str]]]:
    """
    Worker xử lý 1 chunk câu.
    Input:  list[Sentence]  (sentence = list of (token, tag))
    Output: list[(Sentence, retrieved_texts)]
    """
    results = []
    for sent in chunk:
        q_text = sentence_text(sent)
        retrieved = _retrieve_single(q_text)
        results.append((sent, retrieved))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint — robust resume khi bị timeout / crash
# ──────────────────────────────────────────────────────────────────────────────

def _checkpoint_path(output_file: Path, checkpoint_dir: Path | None = None) -> Path:
    """Trả về path checkpoint tương ứng với output file."""
    name = output_file.name + ".checkpoint"
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / name
    # Fallback: try output_file.parent; if not writable, use /tmp
    parent = output_file.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        # Quick write-access check
        test = parent / ".write_test"
        test.touch()
        test.unlink()
        return parent / name
    except OSError:
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        logger.warning(
            "  ⚠ Thư mục output '%s' chỉ đọc, lưu checkpoint vào: %s",
            parent, tmp_dir,
        )
        return tmp_dir / name


def _load_checkpoint(output_file: Path, checkpoint_dir: Path | None = None) -> tuple[int, int]:
    """
    Load checkpoint. Trả về (processed_count, byte_offset).
    Nếu không có checkpoint hoặc lỗi → (0, 0).
    """
    ckpt = _checkpoint_path(output_file, checkpoint_dir)
    if not ckpt.exists():
        return 0, 0
    try:
        with open(ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = int(data["processed"])
        offset = int(data["byte_offset"])
        # Validate: output file phải tồn tại và đủ lớn
        if not output_file.exists():
            return 0, 0
        file_size = output_file.stat().st_size
        if offset > file_size:
            logger.warning(
                "  ⚠ Checkpoint byte_offset (%d) > file size (%d). Reset.",
                offset, file_size,
            )
            return 0, 0
        return count, offset
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        logger.warning("  ⚠ Checkpoint lỗi (%s). Reset.", e)
        return 0, 0


def _save_checkpoint(output_file: Path, processed: int, byte_offset: int,
                     checkpoint_dir: Path | None = None) -> None:
    """Ghi checkpoint ra file (atomic trên hầu hết OS)."""
    ckpt = _checkpoint_path(output_file, checkpoint_dir)
    tmp = ckpt.with_suffix(".tmp")
    data = {"processed": processed, "byte_offset": byte_offset}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    # Atomic rename (Windows: replace nếu đã tồn tại)
    try:
        tmp.replace(ckpt)
    except OSError:
        # Fallback cho Windows cũ
        if ckpt.exists():
            ckpt.unlink()
        tmp.rename(ckpt)


def _delete_checkpoint(output_file: Path, checkpoint_dir: Path | None = None) -> None:
    """Xoá checkpoint khi đã hoàn thành."""
    ckpt = _checkpoint_path(output_file, checkpoint_dir)
    if ckpt.exists():
        ckpt.unlink()


def _truncate_output(output_file: Path, byte_offset: int) -> None:
    """
    Truncate output file về đúng byte_offset.
    Fix trường hợp crash giữa chừng khi đang ghi → dữ liệu dở.
    """
    if not output_file.exists():
        return
    file_size = output_file.stat().st_size
    if file_size > byte_offset:
        logger.info(
            "  🔧 Truncate output: %d → %d bytes (xoá %d bytes dở)",
            file_size, byte_offset, file_size - byte_offset,
        )
        with open(output_file, "r+b") as f:
            f.truncate(byte_offset)


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
    max_seq_len: int,
    num_workers: int,
    chunk_size: int,
    cache_size_mb: int,
    checkpoint_dir: Path | None = None,
) -> None:
    logger.info("Xử lý: %s → %s (workers=%d, chunk=%d)",
                input_file.name, output_file.name, num_workers, chunk_size)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Đếm tổng câu (streaming, không giữ data) ──
    logger.info("  Đang đếm câu trong file input (streaming)...")
    total = count_sentences_in_file(input_file, text_col, tag_col)
    logger.info("  Tổng: %d câu", total)

    # ── Resume: load checkpoint ──
    resume_from, resume_offset = _load_checkpoint(output_file, checkpoint_dir)
    if resume_from > 0:
        logger.info(
            "  ⚡ Resume: bỏ qua %d câu đã xử lý (offset=%d bytes), tiếp tục từ câu %d",
            resume_from, resume_offset, resume_from,
        )
        # Truncate output file về đúng checkpoint offset
        # (xoá dữ liệu dở nếu crash giữa chừng)
        _truncate_output(output_file, resume_offset)

    if resume_from >= total:
        logger.info("  File đã hoàn thành, bỏ qua.")
        _delete_checkpoint(output_file, checkpoint_dir)
        return

    remaining = total - resume_from
    logger.info("  Cần xử lý: %d câu còn lại", remaining)

    # ── File mode ──
    open_mode = "ab" if resume_from > 0 else "wb"

    total_retrieved = 0
    no_result = 0
    processed = resume_from
    t0 = time.time()

    # ── Stream chunks → worker pool → write output ──
    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(db_path, num_perm, ngram_size, top_k,
                  min_jaccard, max_jaccard, cache_size_mb),
    ) as pool:
        # Generator: stream chunks từ file, skip resume_from câu đầu
        chunk_gen = iter_conll_chunks(
            input_file, text_col, tag_col, chunk_size, skip=resume_from,
        )

        with open(output_file, open_mode) as f_out:
            # pool.imap giữ thứ tự, lazy consume generator
            # → chỉ vài chunk trong RAM tại mỗi thời điểm
            for chunk_results in pool.imap(_process_chunk, chunk_gen):
                for sent, retrieved in chunk_results:
                    # Ghi bằng bytes để byte_offset chính xác
                    line = _format_conll_doc_single(sent, retrieved, max_seq_len=max_seq_len)
                    f_out.write(line.encode("utf-8"))
                    total_retrieved += len(retrieved)
                    if not retrieved:
                        no_result += 1
                    processed += 1

                # Flush + fsync + save checkpoint sau mỗi chunk
                f_out.flush()
                os.fsync(f_out.fileno())
                _save_checkpoint(output_file, processed, f_out.tell(), checkpoint_dir)

                # Log progress
                if processed % 5000 < chunk_size:
                    elapsed = time.time() - t0
                    speed = (processed - resume_from) / max(elapsed, 0.1)
                    eta_sec = (total - processed) / max(speed, 0.01)
                    eta_min = eta_sec / 60
                    avg = total_retrieved / max(processed - resume_from, 1)
                    logger.info(
                        "  … %d/%d (%.1f%%) | %.0f câu/s | ETA %.0f phút | avg ret: %.1f",
                        processed, total,
                        100.0 * processed / total,
                        speed, eta_min, avg,
                    )

    # ── Hoàn thành → xoá checkpoint ──
    _delete_checkpoint(output_file, checkpoint_dir)

    elapsed = time.time() - t0
    avg = total_retrieved / max(processed - resume_from, 1)
    logger.info(
        "  Xong: %d câu trong %.0f giây (%.0f câu/s) | avg ret=%.2f | %d ko ret",
        processed, elapsed, (processed - resume_from) / max(elapsed, 0.1),
        avg, no_result,
    )
    logger.info("  Đã lưu: %s", output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Sinh dataset *_doc — PARALLEL + STREAMING (low memory)"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--index_db", required=True)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_jaccard", type=float, default=0.95)
    parser.add_argument("--min_jaccard", type=float, default=0.1)
    parser.add_argument("--num_perm", type=int, default=0,
                        help="0 = đọc từ DB metadata")
    parser.add_argument("--ngram_size", type=int, default=2)

    parser.add_argument("--text_col", type=int, default=0)
    parser.add_argument("--tag_col", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=500,
                        help="Độ dài tối đa của câu ghép để tránh quá giới hạn model. Default: 500")

    parser.add_argument("--splits", nargs="+",
                        default=["train.txt", "dev.txt", "test.txt"])

    # ── Parallel args ──
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 = auto (cpu_count - 1)")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Số câu mỗi chunk gửi cho worker. Default: 500")
    parser.add_argument("--cache_size_mb", type=int, default=128,
                        help="SQLite cache/worker (MB). Default: 128")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Thư mục lưu checkpoint resume. Default: cùng thư mục output")

    args = parser.parse_args()

    # ── Workers ──
    if args.num_workers <= 0:
        args.num_workers = max(1, cpu_count() - 1)
    logger.info("Workers: %d | chunk_size: %d | cache: %d MB/worker",
                args.num_workers, args.chunk_size, args.cache_size_mb)

    # ── Đọc metadata từ DB (1 connection tạm) ──
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
    # Default checkpoint_dir to output_dir so checkpoints always land in a
    # writable location (avoids OSError on read-only mounts like /kaggle/input)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoint dir: %s", ckpt_dir)

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
            max_seq_len=args.max_seq_len,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            cache_size_mb=args.cache_size_mb,
            checkpoint_dir=ckpt_dir,
        )

    logger.info("✓ Hoàn thành! Dataset *_doc đã lưu tại: %s", out_dir)


if __name__ == "__main__":
    main()
