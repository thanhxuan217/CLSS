#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_punct_conll.py
────────────────────
Chuyển đổi raw .txt corpus (có dấu câu Sino-Nom) sang CoNLL column format
cho task Punctuation Restoration.

Quy tắc chuyển đổi:
  - Mỗi ký tự không phải dấu câu → token với label O
  - Khi gặp dấu câu (，。：、；？！), gán làm label cho token TRƯỚC nó
  - Nếu có nhiều dấu câu liên tiếp, chỉ lấy dấu câu đầu tiên
  - Câu được tách theo dấu câu kết thúc câu (。？！) và newline

Output format (tab-separated, 2 cột):
  天\tO
  下\tO
  太\tO
  平\t，
  萬\tO
  民\tO
  安\tO
  樂\t。
  (blank line between sentences)

Usage:
  python tools/build_punct_conll.py \\
      --raw_data_dir  data/raw \\
      --output_dir    data/sino_nom_punct \\
      --train_ratio   0.8 \\
      --dev_ratio     0.1 \\
      [--min_sent_len 3] [--max_sent_len 150] [--encoding utf-8] [--seed 42]
"""

import argparse
import logging
import random
import re
import unicodedata
from pathlib import Path
from collections import Counter

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Dấu câu Sino-Nom cần predict
PUNCT_LABELS = {"，", "。", "：", "、", "；", "？", "！"}

# Dấu câu kết thúc câu (dùng để tách câu)
SENTENCE_END = {"。", "？", "！"}

# Dấu câu và ký tự cần bỏ qua (không làm token)
SKIP_CHARS = set("　 \t\r\n")

def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def process_file(
    file_path: Path,
    min_len: int,
    max_len: int,
    encoding: str,
) -> list[str]:
    """Xử lý một file txt, trả về list các câu dưới dạng chuỗi CoNLL."""
    all_sentences = []
    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            current_sent_tokens = []
            
            for line in f:
                if not line.strip():
                    if current_sent_tokens:
                        if min_len <= len(current_sent_tokens) <= max_len:
                            all_sentences.append("\n".join(f"{t}\t{l}" for t, l in current_sent_tokens))
                        current_sent_tokens = []
                    continue
                
                text = normalize_text(line.strip())
                chars = list(text)
                i = 0
                n = len(chars)
                
                while i < n:
                    ch = chars[i]
                    if ch in SKIP_CHARS:
                        i += 1
                        continue
                    
                    if ch in PUNCT_LABELS:
                        if current_sent_tokens:
                            prev_tok, prev_lab = current_sent_tokens[-1]
                            if prev_lab == "O":
                                current_sent_tokens[-1] = (prev_tok, ch)
                                # Kiểm tra xem đây có phải là dấu kết thúc câu không
                                if ch in SENTENCE_END:
                                    if min_len <= len(current_sent_tokens) <= max_len:
                                        all_sentences.append("\n".join(f"{t}\t{l}" for t, l in current_sent_tokens))
                                    current_sent_tokens = []
                        i += 1
                        continue
                        
                    current_sent_tokens.append((ch, "O"))
                    i += 1
                    
            if current_sent_tokens and min_len <= len(current_sent_tokens) <= max_len:
                all_sentences.append("\n".join(f"{t}\t{l}" for t, l in current_sent_tokens))
                
    except Exception as exc:
        logger.warning("Không đọc được %s: %s", file_path, exc)
        return []

    return all_sentences


def write_conll(sentences: list[str], out_file: Path) -> None:
    """Ghi danh sách câu ra file CoNLL 2-cột."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent)
            f.write("\n\n")
    logger.info("  Đã ghi %d câu → %s", len(sentences), out_file)


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển raw .txt Sino-Nom → CoNLL format cho Punctuation Restoration"
    )
    parser.add_argument(
        "--raw_data_dir", required=True,
        help="Thư mục gốc chứa file .txt raw (duyệt đệ quy)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Thư mục xuất: sẽ tạo train.txt / dev.txt / test.txt"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Tỷ lệ train split. Default: 0.8"
    )
    parser.add_argument(
        "--dev_ratio", type=float, default=0.1,
        help="Tỷ lệ dev split. Default: 0.1 (test = 1 - train - dev)"
    )
    parser.add_argument(
        "--min_sent_len", type=int, default=3,
        help="Số token tối thiểu mỗi câu. Default: 3"
    )
    parser.add_argument(
        "--max_sent_len", type=int, default=150,
        help="Số token tối đa mỗi câu. Default: 150"
    )
    parser.add_argument(
        "--encoding", default="utf-8",
        help="Encoding file input. Default: utf-8"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed cho shuffle. Default: 42"
    )
    args = parser.parse_args()

    assert args.train_ratio + args.dev_ratio < 1.0, \
        "train_ratio + dev_ratio phải < 1.0 để còn phần test"

    raw_dir = Path(args.raw_data_dir)
    out_dir = Path(args.output_dir)
    txt_files = sorted(raw_dir.rglob("*.txt"))

    if not txt_files:
        logger.error("Không tìm thấy file .txt nào trong: %s", raw_dir)
        return

    logger.info("Tìm thấy %d file .txt", len(txt_files))

    # Thu thập tất cả câu
    all_sentences: list[str] = []
    for fp in txt_files:
        logger.info("Xử lý: %s", fp.name)
        sents = process_file(fp, args.min_sent_len, args.max_sent_len, args.encoding)
        all_sentences.extend(sents)

    if not all_sentences:
        logger.error("Không có câu nào sau khi xử lý. Kiểm tra lại file input.")
        return

    # Shuffle và split
    random.seed(args.seed)
    random.shuffle(all_sentences)

    n = len(all_sentences)
    n_train = int(n * args.train_ratio)
    n_dev = int(n * args.dev_ratio)

    train_sents = all_sentences[:n_train]
    dev_sents = all_sentences[n_train:n_train + n_dev]
    test_sents = all_sentences[n_train + n_dev:]

    # Ghi output
    write_conll(train_sents, out_dir / "train.txt")
    write_conll(dev_sents,   out_dir / "dev.txt")
    write_conll(test_sents,  out_dir / "test.txt")

    # Thống kê
    label_counter: Counter = Counter()
    for sent in all_sentences:
        for line in sent.split("\n"):
            if line:
                tok, lab = line.split("\t")
                label_counter[lab] += 1

    total_tokens = sum(label_counter.values())
    logger.info("=" * 60)
    logger.info("THỐNG KÊ")
    logger.info("=" * 60)
    logger.info("Tổng câu       : %d", n)
    logger.info("  Train         : %d", len(train_sents))
    logger.info("  Dev           : %d", len(dev_sents))
    logger.info("  Test          : %d", len(test_sents))
    logger.info("Tổng token     : %d", total_tokens)
    logger.info("Phân phối nhãn:")
    for lab, cnt in sorted(label_counter.items()):
        pct = 100.0 * cnt / total_tokens
        logger.info("  %-6s  %8d  (%.2f%%)", lab, cnt, pct)
    logger.info("Output → %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
