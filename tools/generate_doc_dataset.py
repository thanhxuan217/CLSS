#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_doc_dataset.py
───────────────────────
Query MinHash LSH index (SQLite) để tìm câu tương tự và sinh dataset *_doc
(CoNLL column format) cho CLSS multi-view training.

Quy trình:
  1. Đọc dataset CoNLL column format (train/dev/test)
  2. Mở SQLite index (nhẹ, chỉ dùng ~64 MB RAM)
  3. Với mỗi câu:
     a. Tạo MinHash signature
     b. Query SQLite LSH → candidates
     c. Tính Jaccard similarity thực tế, rank kết quả
     d. Lọc: loại câu quá giống (> max_jaccard) và quá khác (< min_jaccard)
  4. Ghép: câu gốc + <EOS> S-X + retrieved sentences (tagged S-X)
  5. Xuất dataset *_doc ở CoNLL column format

Usage:
  python tools/generate_doc_dataset.py \\
      --input_dir      data/sino_nom_punct \\
      --output_dir     data/sino_nom_punct_doc \\
      --index_db       data/index/minhash.db \\
      --top_k          5 \\
      --min_jaccard    0.1 \\
      --max_jaccard    0.95
"""

import argparse
import gc
import logging
from pathlib import Path

try:
    from datasketch import MinHash
except ImportError:
    raise ImportError("Cài datasketch: pip install datasketch")

from sqlite_lsh import SqliteMinHashLSH

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CoNLL I/O
# ──────────────────────────────────────────────────────────────────────────────

# Mỗi Sentence được biểu diễn là list of (token, tag) tuples
Sentence = list[tuple[str, str]]


def read_conll(file_path: Path, text_col: int = 0, tag_col: int = 1) -> list[Sentence]:
    """
    Đọc file CoNLL column format.
    Mỗi sentence phân cách bằng dòng trống.
    Dòng bắt đầu bằng -DOCSTART- được bỏ qua.
    """
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


def write_conll_doc(
    sentences: list[Sentence],
    retrieved_groups: list[list[str]],
    out_file: Path,
    eos_tag: str = "S-X",
    retrieved_tag: str = "S-X",
) -> None:
    """
    Ghi dataset *_doc theo format CLSS:
      -DOCSTART- O
      (blank)
      <original tokens with tags>
      <EOS>  S-X
      <retrieved tokens with S-X tag>
      (blank)
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for sent, retrieved_sents in zip(sentences, retrieved_groups):
            f.write("-DOCSTART- O\n\n")

            # Câu gốc
            for token, tag in sent:
                f.write(f"{token}\t{tag}\n")

            if retrieved_sents:
                # EOS separator
                f.write(f"<EOS>\t{eos_tag}\n")

                # Các câu retrieved
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


def jaccard_estimate(m1: MinHash, m2: MinHash) -> float:
    return m1.jaccard(m2)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def sentence_text(sent: Sentence) -> str:
    """Nối các token thành chuỗi (không khoảng trắng cho CJK)."""
    return "".join(tok for tok, _ in sent)


def retrieve_for_sentence(
    query_text: str,
    lsh: SqliteMinHashLSH,
    num_perm: int,
    ngram_size: int,
    top_k: int,
    min_jaccard: float,
    max_jaccard: float,
) -> list[str]:
    """
    Query SQLite LSH, tính Jaccard thực, lọc và trả về top-K câu tương tự.
    Loại bỏ câu trùng hoàn toàn với query.
    """
    shingles_q = make_shingles(query_text, ngram_size)
    mh_q = make_minhash(shingles_q, num_perm)

    candidate_keys = lsh.query(mh_q)
    if not candidate_keys:
        return []

    # Batch lấy sentences từ SQLite (1 query thay vì N queries)
    cand_texts = lsh.get_sentences_batch(candidate_keys)

    scored: list[tuple[float, str]] = []
    for key in candidate_keys:
        cand_text = cand_texts.get(key)
        if cand_text is None or cand_text == query_text:
            continue

        shingles_c = make_shingles(cand_text, ngram_size)
        mh_c = make_minhash(shingles_c, num_perm)
        j = jaccard_estimate(mh_q, mh_c)

        if j < min_jaccard or j > max_jaccard:
            continue

        scored.append((j, cand_text))

    # Sort giảm dần theo Jaccard
    scored.sort(key=lambda x: x[0], reverse=True)

    # Dedup (lấy câu unique)
    seen: set[str] = set()
    result: list[str] = []
    for _, text in scored:
        if text not in seen:
            seen.add(text)
            result.append(text)
        if len(result) >= top_k:
            break

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_split(
    input_file: Path,
    output_file: Path,
    lsh: SqliteMinHashLSH,
    num_perm: int,
    ngram_size: int,
    top_k: int,
    min_jaccard: float,
    max_jaccard: float,
    text_col: int,
    tag_col: int,
) -> None:
    logger.info("Xử lý: %s → %s", input_file.name, output_file.name)
    sentences = read_conll(input_file, text_col, tag_col)
    logger.info("  Số câu: %d", len(sentences))

    retrieved_groups: list[list[str]] = []
    total_retrieved = 0
    no_result = 0

    for i, sent in enumerate(sentences):
        q_text = sentence_text(sent)
        retrieved = retrieve_for_sentence(
            q_text, lsh,
            num_perm, ngram_size, top_k, min_jaccard, max_jaccard,
        )
        retrieved_groups.append(retrieved)
        total_retrieved += len(retrieved)
        if not retrieved:
            no_result += 1

        if (i + 1) % 500 == 0:
            logger.info(
                "  … %d / %d câu (avg retrieved: %.1f)",
                i + 1,
                len(sentences),
                total_retrieved / (i + 1),
            )

    avg = total_retrieved / max(len(sentences), 1)
    logger.info(
        "  Xong: avg retrieved=%.2f/câu | %d câu không có retrieved",
        avg, no_result,
    )

    write_conll_doc(sentences, retrieved_groups, output_file)
    logger.info("  Đã lưu: %s", output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Sinh dataset *_doc dùng MinHash LSH retrieval (SQLite)"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Thư mục dataset gốc (CoNLL column format)")
    parser.add_argument("--output_dir", required=True,
                        help="Thư mục xuất dataset *_doc")
    parser.add_argument("--index_db", required=True,
                        help="SQLite database chứa LSH index (.db)")

    parser.add_argument("--top_k", type=int, default=5,
                        help="Số câu retrieved tối đa mỗi câu gốc. Default: 5")
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

    args = parser.parse_args()

    # ── Mở SQLite index ──
    logger.info("Đang mở LSH index từ: %s", args.index_db)
    lsh = SqliteMinHashLSH.open(args.index_db)

    total = lsh.total_sentences()
    num_perm = args.num_perm if args.num_perm > 0 else lsh.num_perm
    logger.info("Index: %d câu | num_perm=%d | b=%d | r=%d",
                total, num_perm, lsh.b, lsh.r)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Xử lý từng split ──
    for split_name in args.splits:
        in_file = in_dir / split_name
        out_file = out_dir / split_name

        if not in_file.exists():
            logger.warning("Bỏ qua (không tồn tại): %s", in_file)
            continue

        process_split(
            input_file=in_file,
            output_file=out_file,
            lsh=lsh,
            num_perm=num_perm,
            ngram_size=args.ngram_size,
            top_k=args.top_k,
            min_jaccard=args.min_jaccard,
            max_jaccard=args.max_jaccard,
            text_col=args.text_col,
            tag_col=args.tag_col,
        )

    lsh.close()
    logger.info("✓ Hoàn thành! Dataset *_doc đã lưu tại: %s", out_dir)


if __name__ == "__main__":
    main()
