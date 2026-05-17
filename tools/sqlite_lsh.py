#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sqlite_lsh.py
─────────────
SQLite-backed MinHash LSH — thay thế datasketch.MinHashLSH cho corpus lớn.

Toàn bộ band hash tables + sentences nằm trên DISK (SQLite).
RAM chỉ dùng SQLite page cache (~50-100 MB) bất kể corpus size.

Usage (build):
    from sqlite_lsh import SqliteMinHashLSH

    lsh = SqliteMinHashLSH.create("index.db", threshold=0.3, num_perm=128)
    lsh.insert("0", minhash_obj, sentence_text="天下太平")
    lsh.finalize()   # tạo index sau khi insert xong
    lsh.close()

Usage (query):
    lsh = SqliteMinHashLSH.open("index.db")
    keys = lsh.query(minhash_obj)
    text = lsh.get_sentence(keys[0])
    lsh.close()
"""

import hashlib
import sqlite3
from pathlib import Path

try:
    from datasketch import MinHash
except ImportError:
    raise ImportError("Cài datasketch: pip install datasketch")


# ──────────────────────────────────────────────────────────────────────────────
# Optimal (b, r) parameter computation — same algorithm as datasketch
# ──────────────────────────────────────────────────────────────────────────────

def _integrate_trapezoidal(f, a, b, n=50):
    """Simple trapezoidal numerical integration."""
    if b <= a:
        return 0.0
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def _optimal_param(threshold, num_perm):
    """
    Compute optimal (b, r) for LSH.
    Same false-positive/false-negative minimization as datasketch.
    b = number of bands, r = rows per band, b*r <= num_perm.
    """
    min_error = float("inf")
    opt = (1, num_perm)

    for b in range(1, num_perm + 1):
        max_r = num_perm // b
        for r in range(1, max_r + 1):
            # False positive: integral of P(candidate) from 0 to threshold
            fp = _integrate_trapezoidal(
                lambda s: 1.0 - (1.0 - s ** float(r)) ** float(b),
                0.0, threshold,
            )
            # False negative: integral of P(not candidate) from threshold to 1
            fn = _integrate_trapezoidal(
                lambda s: 1.0 - (1.0 - (1.0 - s ** float(r)) ** float(b)),
                threshold, 1.0,
            )
            error = 0.5 * fp + 0.5 * fn
            if error < min_error:
                min_error = error
                opt = (b, r)

    return opt


# ──────────────────────────────────────────────────────────────────────────────
# SqliteMinHashLSH
# ──────────────────────────────────────────────────────────────────────────────

class SqliteMinHashLSH:
    """
    MinHash LSH backed by SQLite.

    Tất cả band hash tables + sentences nằm trên disk.
    Peak RAM ≈ SQLite page cache (tuỳ chỉnh, mặc định 64 MB).

    Schema:
        meta(key, value)           — threshold, num_perm, b, r
        sentences(id, text)        — câu theo global index
        bands(band_id, bucket, key) — band hash → sentence key
        + index trên bands(band_id, bucket) sau khi build xong
    """

    def __init__(self, conn, num_perm, threshold, b, r):
        self.conn = conn
        self.num_perm = num_perm
        self.threshold = threshold
        self.b = b
        self.r = r
        self._insert_buf = []
        self._sent_buf = []
        self._buf_size = 0

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def create(cls, db_path, threshold, num_perm, cache_size_mb=64):
        """Tạo database mới cho build phase."""
        db_path = str(db_path)
        # Xoá file cũ nếu có
        p = Path(db_path)
        if p.exists():
            p.unlink()

        conn = sqlite3.connect(db_path)
        # Performance pragmas cho write-heavy workload
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA cache_size=-{cache_size_mb * 1024}")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256 MB mmap

        # Compute optimal bands/rows
        b, r = _optimal_param(threshold, num_perm)

        # Create tables (NO indexes yet — thêm sau khi insert xong)
        conn.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE sentences (id INTEGER PRIMARY KEY, text TEXT NOT NULL);
            CREATE TABLE bands (band_id INTEGER NOT NULL,
                                bucket  BLOB    NOT NULL,
                                key     INTEGER NOT NULL);
        """)

        # Save metadata
        conn.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [
                ("threshold", str(threshold)),
                ("num_perm", str(num_perm)),
                ("b", str(b)),
                ("r", str(r)),
            ],
        )
        conn.commit()

        obj = cls(conn, num_perm, threshold, b, r)
        return obj

    @classmethod
    def open(cls, db_path, cache_size_mb=64):
        """Mở database đã build (read-only query)."""
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.execute(f"PRAGMA cache_size=-{cache_size_mb * 1024}")
        conn.execute("PRAGMA mmap_size=268435456")

        # Read metadata
        rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        num_perm = int(rows["num_perm"])
        threshold = float(rows["threshold"])
        b = int(rows["b"])
        r = int(rows["r"])

        return cls(conn, num_perm, threshold, b, r)

    # ── Insert (build phase) ─────────────────────────────────────────────────

    def _hash_band(self, hashvalues, band_id):
        """Hash 1 band of the MinHash signature."""
        start = band_id * self.r
        end = min(start + self.r, self.num_perm)
        band_bytes = hashvalues[start:end].tobytes()
        return hashlib.sha1(band_bytes).digest()

    def insert(self, key, minhash, sentence_text=None):
        """
        Insert 1 MinHash vào buffer.
        Gọi flush() mỗi batch_size để ghi xuống disk.

        Args:
            key: string hoặc int key (global index)
            minhash: datasketch.MinHash object
            sentence_text: (optional) câu tương ứng để lưu kèm
        """
        int_key = int(key)
        hashvalues = minhash.hashvalues

        for band_id in range(self.b):
            bucket = self._hash_band(hashvalues, band_id)
            self._insert_buf.append((band_id, bucket, int_key))

        if sentence_text is not None:
            self._sent_buf.append((int_key, sentence_text))

        self._buf_size += 1

    def flush(self):
        """Ghi buffer xuống SQLite."""
        if self._insert_buf:
            self.conn.executemany(
                "INSERT INTO bands VALUES (?, ?, ?)", self._insert_buf
            )
            self._insert_buf.clear()
        if self._sent_buf:
            self.conn.executemany(
                "INSERT INTO sentences VALUES (?, ?)", self._sent_buf
            )
            self._sent_buf.clear()
        self.conn.commit()
        self._buf_size = 0

    def finalize(self):
        """
        Gọi SAU KHI insert xong tất cả.
        Flush buffer còn lại + tạo index (tốn thời gian nhưng chỉ 1 lần).
        """
        self.flush()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bands "
            "ON bands(band_id, bucket)"
        )
        self.conn.commit()
        # Compact database
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ── Query (read phase) ───────────────────────────────────────────────────

    def query(self, minhash):
        """
        Query LSH index. Trả về list of keys (string) giống datasketch API.
        """
        hashvalues = minhash.hashvalues
        candidates = set()
        cur = self.conn.cursor()

        for band_id in range(self.b):
            bucket = self._hash_band(hashvalues, band_id)
            cur.execute(
                "SELECT key FROM bands WHERE band_id = ? AND bucket = ?",
                (band_id, bucket),
            )
            for (k,) in cur:
                candidates.add(str(k))

        return list(candidates)

    def get_sentence(self, key):
        """Lấy câu theo key (string hoặc int)."""
        cur = self.conn.execute(
            "SELECT text FROM sentences WHERE id = ?", (int(key),)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def get_sentences_batch(self, keys):
        """Lấy nhiều câu cùng lúc. Returns dict {key: text}."""
        if not keys:
            return {}
        int_keys = [int(k) for k in keys]
        placeholders = ",".join("?" * len(int_keys))
        cur = self.conn.execute(
            f"SELECT id, text FROM sentences WHERE id IN ({placeholders})",
            int_keys,
        )
        return {str(row[0]): row[1] for row in cur}

    def total_sentences(self):
        """Số câu trong index."""
        cur = self.conn.execute("SELECT COUNT(*) FROM sentences")
        return cur.fetchone()[0]

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
