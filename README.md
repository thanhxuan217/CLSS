
# CLNER / CLSS — Named Entity Recognition with External Context & Cooperative Learning

The code is for our ACL-IJCNLP 2021 paper: [Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://arxiv.org/abs/2105.03654)

CLNER is a framework for improving NER accuracy by retrieving external context sentences, then applying **cooperative learning** (multi-view training with KL divergence) between the original view and the context-augmented view.

This fork extends the original framework with a **MinHash LSH RAG pipeline** tailored for **Sino-Nom / Classical Chinese** — replacing the Google Search retrieval with efficient local approximate-nearest-neighbor retrieval over your own raw corpus.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-named-entity-recognition-by/named-entity-recognition-on-wnut-2016)](https://paperswithcode.com/sota/named-entity-recognition-on-wnut-2016?p=improving-named-entity-recognition-by)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-named-entity-recognition-by/named-entity-recognition-on-wnut-2017)](https://paperswithcode.com/sota/named-entity-recognition-on-wnut-2017?p=improving-named-entity-recognition-by)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-named-entity-recognition-by/named-entity-recognition-ner-on-bc5cdr)](https://paperswithcode.com/sota/named-entity-recognition-ner-on-bc5cdr?p=improving-named-entity-recognition-by)

---

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Sino-Nom Pipeline (MinHash RAG)](#sino-nom-pipeline-minhash-rag)
  - [Tổng quan luồng dữ liệu](#tổng-quan-luồng-dữ-liệu)
  - [Bước 0 — Chuẩn bị dữ liệu đầu vào](#bước-0--chuẩn-bị-dữ-liệu-đầu-vào)
  - [Bước 1 — Tiền xử lý raw corpus](#bước-1--tiền-xử-lý-raw-corpus)
  - [Bước 2 — Build MinHash LSH Index](#bước-2--build-minhash-lsh-index)
  - [Bước 3 — Sinh dataset \*\_doc](#bước-3--sinh-dataset-_doc)
  - [Bước 4 — Cấu hình và Training](#bước-4--cấu-hình-và-training)
- [Original English Pipeline](#original-english-pipeline)
- [Config File Reference](#config-file-reference)
- [Parse / Inference](#parse--inference)
- [Citing Us](#citing-us)
- [Contact](#contact)

---

## Requirements

Python 3.10+ và PyTorch 2.0+.

```bash
pip install -r requirements.txt
```

> **Lưu ý quan trọng:** `requirements.txt` đã được làm sạch so với bản gốc. Các package deprecated (`pytorch-transformers`, `pytorch-pretrained-bert`, `tensorflow`, `mxnet`, v.v.) đã bị xóa. Package mới được thêm: `datasketch>=1.6.0` cho MinHash LSH.

---

## Project Structure

```
CLSS/
├── flair/                   # Framework core (NER model, embeddings, trainers)
│   ├── models/
│   ├── trainers/
│   └── embeddings.py
├── tools/                   # Sino-Nom data pipeline scripts
│   ├── preprocess_raw_corpus.py   # Bước 1: Tiền xử lý raw corpus
│   ├── build_minhash_index.py     # Bước 2: Build MinHash LSH index
│   ├── generate_doc_dataset.py    # Bước 3: Sinh dataset *_doc
│   └── bert_scoring.py            # (Legacy) BertScore ranking cho English
├── config/
│   ├── sino_nom_doc_cl.yaml       # Config training Sino-Nom (MinHash RAG)
│   └── wnut17_doc_cl_kl.yaml      # Config gốc cho WNUT17 (English)
├── data/                    # Thư mục dữ liệu (không track bởi git)
│   ├── raw/                 # Raw corpus .txt files
│   ├── processed/           # Corpus đã tiền xử lý
│   ├── index/               # MinHash LSH index (pickle)
│   ├── sino_nom/            # NER dataset gốc (CoNLL format)
│   └── sino_nom_doc/        # NER dataset *_doc (output của pipeline)
├── resources/taggers/       # Model checkpoints và tag dictionaries
├── train.py                 # Entry point training
└── requirements.txt
```

---

## Sino-Nom Pipeline — Punctuation Restoration (MinHash RAG)

Task: Dự đoán dấu câu cần chèn **sau mỗi ký tự** trong văn bản Sino-Nom / Hán Nôm.

**Tag set**: `O` (không có dấu câu) · `，` · `。` · `：` · `、` · `；` · `？` · `！`

**Ví dụ**:
```
天  O
下  O
太  O
平  ，     ← cần chèn dấu phẩy sau "平"
萬  O
民  O
安  O
樂  。     ← cần chèn dấu chấm sau "樂"
```

### Tổng quan luồng dữ liệu

```
Raw .txt corpus (có dấu câu)
        │
        ▼
[Bước 1] build_punct_conll.py  →  data/sino_nom_punct/{train,dev,test}.txt
        │
        ▼
[Bước 2a] preprocess_raw_corpus.py  →  data/processed/corpus_sentences.txt
        │
        ▼
[Bước 2b] build_minhash_index.py  →  data/index/{minhash.pkl, sentences.txt}
        │
        ▼
[Bước 3] generate_doc_dataset.py  →  data/sino_nom_punct_doc/{train,dev,test}.txt
        │
        ▼
[Bước 4] train.py --config config/sino_nom_doc_cl.yaml
        │
        ▼
  resources/taggers/<model_name>/best-model.pt
```

---

### Bước 0 — Chuẩn bị dữ liệu raw

Raw corpus là các file `.txt` **đã có dấu câu** Sino-Nom. Script sẽ tự động:
- Tách thành từng câu dựa trên dấu câu kết thúc (。？！)
- Gán nhãn punctuation cho từng ký tự
- Split train/dev/test

**Cấu trúc thư mục raw:**
```
data/raw/
├── book1/chapter1.txt
├── book1/chapter2.txt
└── inscriptions/stele_A.txt
```

**Nội dung file mẫu:**
```
天下太平，萬民安樂。永曆帝御駕親征？
```

---

### Bước 1 — Sinh dataset CoNLL từ raw corpus

```bash
python tools/build_punct_conll.py \
    --raw_data_dir  data/raw \
    --output_dir    data/sino_nom_punct \
    --train_ratio   0.8 \
    --dev_ratio     0.1 \
    --min_sent_len  3 \
    --max_sent_len  150
```

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--raw_data_dir` | Thư mục raw `.txt` **(bắt buộc)** | — |
| `--output_dir` | Thư mục xuất CoNLL **(bắt buộc)** | — |
| `--train_ratio` | Tỷ lệ train | `0.8` |
| `--dev_ratio` | Tỷ lệ dev (test = 1 - train - dev) | `0.1` |
| `--min_sent_len` | Số token tối thiểu/câu | `3` |
| `--max_sent_len` | Số token tối đa/câu | `150` |
| `--seed` | Random seed cho shuffle | `42` |

**Output mẫu (log thống kê):**
```
Tổng câu       : 125,430
  Train         : 100,344
  Dev           : 12,543
  Test          : 12,543
Tổng token     : 4,821,500
Phân phối nhãn:
  O        4,234,200  (87.82%)
  ，          312,450   (6.48%)
  。          198,230   (4.11%)
  ：           43,120   (0.89%)
  ...
```

---

### Bước 2 — Build MinHash LSH Index (cho retrieved context)

```bash
# Tiền xử lý corpus (tách câu, lọc, dedup)
python tools/preprocess_raw_corpus.py \
    --raw_data_dir  data/raw \
    --output_file   data/processed/corpus_sentences.txt

# Build index (streaming, memory-efficient)
python tools/build_minhash_index.py \
    --sentences_file    data/processed/corpus_sentences.txt \
    --output_index      data/index/minhash.pkl \
    --output_sentences  data/index/sentences.txt \
    --num_perm          128 \
    --ngram_size        2 \
    --threshold         0.3 \
    --batch_size        50000
```

---

### Bước 3 — Sinh dataset \*\_doc (với retrieved context)

```bash
python tools/generate_doc_dataset.py \
    --input_dir      data/sino_nom_punct \
    --output_dir     data/sino_nom_punct_doc \
    --index_path     data/index/minhash.pkl \
    --sentences_path data/index/sentences.txt \
    --top_k          5 \
    --min_jaccard    0.1 \
    --max_jaccard    0.95 \
    --num_perm       128 \
    --ngram_size     2
```

Output: mỗi câu gốc được nối thêm các câu retrieved, đánh dấu tag `S-X`:
```
天	O
下	O
平	，
<EOS>	S-X
萬	S-X
民	S-X
```

---

### Bước 4 — Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/sino_nom_doc_cl.yaml
```

**Model lưu tại:**
```
resources/taggers/<model_name>/
├── best-model.pt
└── training.log
```

**Metric evaluate** (per-class token-level F1):
```
MICRO_AVG: acc 0.9124 - f1-score 0.8234
MACRO_AVG: acc 0.8901 - f1-score 0.7856
O          tp: 421200 - fp: 12300 - ...  f1-score: 0.9712
，         tp:  28100 - fp:  2100 - ...  f1-score: 0.8134
。         tp:  19200 - fp:  1800 - ...  f1-score: 0.8921
...
```

**Các tham số training quan trọng trong config:**
```yaml
train:
  learning_rate: 5.0e-5        # Giảm xuống 2e-5 nếu dùng model lớn hơn
  mini_batch_size: 4           # Tăng nếu VRAM > 16GB
  gradient_accumulation_steps: 4
  max_epochs: 10
embeddings:
  TransformerWordEmbeddings-0:
    model: bert-base-chinese   # Hoặc hfl/chinese-roberta-wwm-ext
```







Bạn cần **2 loại dữ liệu** trước khi bắt đầu:

#### A. Raw corpus (để build retrieval index)

Tập hợp các file văn bản Sino-Nom / Hán Nôm, **không cần** có nhãn NER.

**Yêu cầu định dạng:**
- Các file `.txt`, đặt trong một thư mục (có thể nhiều thư mục con — script duyệt đệ quy)
- Encoding: **UTF-8** (hoặc chỉ định bằng `--encoding`)
- Mỗi file chứa văn bản liên tục **có dấu câu** (`。！？；` hoặc newline để tách câu)
- Không cần mỗi dòng là một câu riêng — script tự tách

**Ví dụ cấu trúc:**
```
data/raw/
├── book1/
│   ├── chapter1.txt
│   └── chapter2.txt
├── book2.txt
└── inscriptions/
    └── stele_A.txt
```

**Ví dụ nội dung file:**
```
天地玄黃宇宙洪荒。日月盈昃辰宿列張。寒來暑往秋收冬藏。
閏餘成歲律呂調陽。雲騰致雨露結為霜。
```

---

#### B. NER dataset (dataset có nhãn, để train model)

Dataset ở **CoNLL column format** — mỗi token một dòng, câu phân cách bằng dòng trống.

**Cấu trúc thư mục bắt buộc:**
```
data/sino_nom/
├── train.txt
├── dev.txt
└── test.txt
```

**Format mỗi file (2 cột, tab-separated):**
```
-DOCSTART- O

天	O
下	O
太	B-LOC
平	I-LOC
，	O
萬	O
民	O
安	O
樂	O

-DOCSTART- O

永	B-PER
曆	I-PER
帝	I-PER
御	O
駕	O
親	O
征	O
```

> **Lưu ý về tag scheme:** Script hỗ trợ cả BIO (`B-`, `I-`, `O`) và BIOES. Config `tag_to_bioes: ner` sẽ tự động chuyển đổi sang BIOES khi load.

**Nếu có nhiều hơn 2 cột** (ví dụ có thêm cột POS), chỉ định cột khi chạy script:
```bash
# Ví dụ: text ở cột 0, NER ở cột 2
python tools/generate_doc_dataset.py --text_col 0 --tag_col 2 ...
```

---


## Original English Pipeline

Để chạy pipeline gốc với Google Search (English NER), xem hướng dẫn tại [link dataset](https://1drv.ms/u/s!Am53YNAPSsodg9ce3ovPukuFtSj6NQ?e=tpCvf8).

```bash
# Training with external contexts
CUDA_VISIBLE_DEVICES=0 python train.py --config config/wnut17_doc.yaml

# Training with cooperative learning
CUDA_VISIBLE_DEVICES=0 python train.py --config config/wnut17_doc_cl_kl.yaml
```

---

## Config File Reference

Config files ở YAML format. Các key chính:

| Key | Mô tả |
|-----|-------|
| `targets` | Task type: `ner`, `upos`, `chunk`, `dependency`, ... |
| `ner.Corpus` | Kết hợp corpus: `CorpusA:CorpusB` |
| `ner.ColumnCorpus-X.data_folder` | Đường dẫn thư mục dataset |
| `ner.ColumnCorpus-X.column_format` | Mapping cột → field (`0: text`, `1: ner`) |
| `ner.tag_dictionary` | Path đến tag dictionary `.pkl` (tự tạo nếu chưa có) |
| `model.FastSequenceTagger.multi_view_training` | Bật cooperative learning |
| `model.FastSequenceTagger.distill_posterior` | Dùng KL divergence loss giữa 2 views |
| `model.FastSequenceTagger.remove_x` | Bỏ token `S-X` khỏi prediction |
| `embeddings.TransformerWordEmbeddings-0.model` | HuggingFace model name hoặc local path |
| `train.learning_rate` | Learning rate |
| `train.mini_batch_size` | Batch size |
| `train.gradient_accumulation_steps` | Accumulation steps |
| `train.max_epochs` | Số epochs |
| `trainer` | `ModelFinetuner` (fine-tune) hoặc `ReinforcementTrainer` (ACE) |
| `target_dir` | Thư mục lưu model |
| `model_name` | Tên subfolder trong `target_dir` |

---

## Parse / Inference

Để predict trên file mới:

```bash
# File cần predict phải có tên chứa 'train' và đặt trong $dir
# Ví dụ: parse_dir/train.myfile.txt

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/sino_nom_doc_cl.yaml \
    --parse \
    --target_dir parse_dir \
    --keep_order
```

Kết quả xuất ra thư mục `outputs/`. Format file input: CoNLL column với dummy tags (xem [issue #12](https://github.com/Alibaba-NLP/ACE/issues/12)).

---

## Citing Us

```bibtex
@inproceedings{wang2021improving,
    title     = {{Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning}},
    author    = {Wang, Xinyu and Jiang, Yong and Bach, Nguyen and Wang, Tao and Huang, Zhongqiang and Huang, Fei and Tu, Kewei},
    booktitle = {Proceedings of ACL-IJCNLP 2021},
    month     = aug,
    year      = {2021},
    publisher = {Association for Computational Linguistics},
}
```

---

## Contact

Feel free to open an issue or email questions to [Xinyu Wang](http://wangxinyu0922.github.io/).
