"""
Optimized and robust entity resolution pipeline using vectorized pandas joins.

Key speed & memory optimizations:
  - Fast single-pass regex normalization with ASCII fast-paths
  - Fast stream reading for TSV and ground truth (seconds instead of minutes)
  - Selective candidate filtering during S2/S3 streaming based on S1 blocking keys
  - Vectorized pandas merge blocking without row-by-row DataFrame iterrows overhead
  - Correct 10-parameter featurize unpack matching record lookups
  - Full CLI support (--train-sample, --val-frac, --max-cands, --max-tok1)
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT       = Path(__file__).resolve().parents[3]
DATA_TRAIN = ROOT / "student_resource" / "dataset" / "train"
DATA_TEST  = ROOT / "student_resource" / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
SCRATCH    = ROOT / "data" / "_pipeline_scratch"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import macro_f05, tune_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
TRAIN_SAMPLE  = 200_000
VAL_FRAC      = 0.15
MAX_CANDS     = 60
MAX_TOK1      = 10      # cap single-token candidates
STREAM_CHUNK  = 200_000
FEAT_BATCH    = 50_000
SEED          = 42

FEATURE_NAMES = [
    "name_tok_jac", "name_c3_jac", "name_c4_jac", "name_lcp",
    "name_exact", "name_len_r", "name_first_eq", "name_last_eq",
    "addr_tok_jac", "addr_c3_jac", "addr_exact",
    "addr_num_jac", "addr_len_r",
    "country_eq", "both_exact",
]

# ═════════════════════════════════════════════════════════════════════════════
# Fast single-pass normalization
# ═════════════════════════════════════════════════════════════════════════════

_LEGAL_DICT = {
    "incorporated": "inc", "corporation": "corp", "corporations": "corp",
    "limited liability company": "llc", "limited liability partnership": "llp",
    "private limited": "pvt ltd", "private ltd": "pvt ltd", "private": "pvt",
    "limited": "ltd", "company": "co", "brothers": "bros", "and": "&",
    "enterprises": "ent", "enterprise": "ent", "trading": "trdg",
    "international": "intl", "national": "natl", "associates": "assoc",
    "services": "svcs", "solutions": "soln", "industries": "ind",
    "group": "grp", "management": "mgmt",
}
_LEGAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_LEGAL_DICT.keys(), key=len, reverse=True)) + r")\b",
    re.I,
)

_ADDR_DICT = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "court": "ct", "circle": "cir", "lane": "ln",
    "place": "pl", "suite": "ste", "apartment": "apt", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "highway": "hwy", "freeway": "fwy", "parkway": "pkwy", "nagar": "ngr",
    "colony": "col", "sector": "sec",
}
_ADDR_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ADDR_DICT.keys(), key=len, reverse=True)) + r")\b",
    re.I,
)
_PUNC = re.compile(r"[^\w\s&]")


def _to_ascii(t: str) -> str:
    n = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return n if n.strip() else t


def _clean_norm_name(s: str) -> str:
    if not s:
        return ""
    if not s.isascii():
        s = _to_ascii(s)
    t = _PUNC.sub(" ", s.lower())
    t = _LEGAL_RE.sub(lambda m: _LEGAL_DICT[m.group(0).lower()], t)
    return " ".join(t.split())


def _clean_norm_addr(s: str) -> str:
    if not s:
        return ""
    if not s.isascii():
        s = _to_ascii(s)
    t = _PUNC.sub(" ", s.lower())
    t = _ADDR_RE.sub(lambda m: _ADDR_DICT[m.group(0).lower()], t)
    return " ".join(t.split())


def vec_norm_name(series: pd.Series) -> pd.Series:
    vals = series.fillna("").astype(str).tolist()
    return pd.Series([_clean_norm_name(x) for x in vals], index=series.index)


def vec_norm_addr(series: pd.Series) -> pd.Series:
    vals = series.fillna("").astype(str).tolist()
    return pd.Series([_clean_norm_addr(x) for x in vals], index=series.index)


def tok1(s: str) -> str:
    toks = s.split()
    return toks[0] if toks else ""


def tok2(s: str) -> str:
    toks = s.split()
    return " ".join(toks[:2]) if len(toks) >= 2 else ""


# ═════════════════════════════════════════════════════════════════════════════
# Vectorized Blocking via Pandas Merges
# ═════════════════════════════════════════════════════════════════════════════

def block_by_join(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    max_cands: int = MAX_CANDS,
    max_tok1: int = MAX_TOK1,
) -> Dict[str, Set[str]]:
    """
    Vectorized blocking via pandas merge on (key, country).
    Returns {s1_id: set of s23_ids}.
    """
    cands: Dict[str, Set[str]] = {eid: set() for eid in s1["entity_id"]}

    # Pass 1: exact norm_name
    s1_nn = s1[s1["norm_name"] != ""]
    s23_nn = s23[s23["norm_name"] != ""]
    if not s1_nn.empty and not s23_nn.empty:
        m = pd.merge(
            s1_nn[["entity_id", "norm_name", "country"]],
            s23_nn[["entity_id", "norm_name", "country"]],
            on=["norm_name", "country"],
            suffixes=("_1", "_23"),
        )
        for s1_id, s23_id in zip(m["entity_id_1"], m["entity_id_23"]):
            s = cands[s1_id]
            if len(s) < max_cands:
                s.add(s23_id)

    # Pass 2: exact norm_addr
    s1_na = s1[s1["norm_addr"] != ""]
    s23_na = s23[s23["norm_addr"] != ""]
    if not s1_na.empty and not s23_na.empty:
        m = pd.merge(
            s1_na[["entity_id", "norm_addr", "country"]],
            s23_na[["entity_id", "norm_addr", "country"]],
            on=["norm_addr", "country"],
            suffixes=("_1", "_23"),
        )
        for s1_id, s23_id in zip(m["entity_id_1"], m["entity_id_23"]):
            s = cands[s1_id]
            if len(s) < max_cands:
                s.add(s23_id)

    # Pass 3: tok2 join
    s1_t2 = s1[s1["tok2_name"] != ""]
    s23_t2 = s23[s23["tok2_name"] != ""]
    if not s1_t2.empty and not s23_t2.empty:
        m = pd.merge(
            s1_t2[["entity_id", "tok2_name", "country"]],
            s23_t2[["entity_id", "tok2_name", "country"]],
            on=["tok2_name", "country"],
            suffixes=("_1", "_23"),
        )
        for s1_id, s23_id in zip(m["entity_id_1"], m["entity_id_23"]):
            s = cands[s1_id]
            if len(s) < max_cands:
                s.add(s23_id)

    # Pass 4: tok1 join (capped at max_tok1 per entity)
    sparse_s1 = s1[s1["entity_id"].map(lambda e: len(cands[e]) < 5) & (s1["tok1_name"] != "")]
    s23_t1 = s23[s23["tok1_name"] != ""]
    if not sparse_s1.empty and not s23_t1.empty:
        m = pd.merge(
            sparse_s1[["entity_id", "tok1_name", "country"]],
            s23_t1[["entity_id", "tok1_name", "country"]],
            on=["tok1_name", "country"],
            suffixes=("_1", "_23"),
        )
        counts: Dict[str, int] = {}
        for s1_id, s23_id in zip(m["entity_id_1"], m["entity_id_23"]):
            n = counts.get(s1_id, 0)
            if n < max_tok1:
                s = cands[s1_id]
                if len(s) < max_cands and s23_id not in s:
                    s.add(s23_id)
                    counts[s1_id] = n + 1

    return cands


# ═════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═════════════════════════════════════════════════════════════════════════════

def _jac(a: frozenset, b: frozenset) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)


def _tok_jac(a: str, b: str) -> float:
    return _jac(frozenset(a.split()), frozenset(b.split()))


def _char_jac(a: str, b: str, n: int) -> float:
    if len(a) < n or len(b) < n: return 0.0
    sa = frozenset(a[i:i+n] for i in range(len(a)-n+1))
    sb = frozenset(b[i:i+n] for i in range(len(b)-n+1))
    return _jac(sa, sb)


def _lcp(a: str, b: str) -> float:
    if not a or not b: return 0.0
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]: i += 1
    return i / max(len(a), len(b))


def featurize(nn1: str, na1: str, c1: str, on1: str, oa1: str,
              nn2: str, na2: str, c2: str, on2: str, oa2: str) -> List[float]:
    """Pairwise features for an (S1, S2/S3) pair."""
    t1 = nn1.split()
    t2 = nn2.split()
    ne = float(nn1 == nn2 and bool(nn1))
    ae = float(na1 == na2 and bool(na1))
    return [
        _tok_jac(nn1, nn2),
        _char_jac(nn1, nn2, 3),
        _char_jac(nn1, nn2, 4),
        _lcp(nn1, nn2),
        ne,
        (min(len(nn1), len(nn2)) / max(len(nn1), len(nn2))) if nn1 and nn2 else 0.0,
        float(bool(t1) and bool(t2) and t1[0] == t2[0]),
        float(bool(t1) and bool(t2) and t1[-1] == t2[-1]),
        _tok_jac(na1, na2),
        _char_jac(na1, na2, 3),
        ae,
        _jac(frozenset(re.findall(r"\d+", oa1)), frozenset(re.findall(r"\d+", oa2))),
        (min(len(na1), len(na2)) / max(len(na1), len(na2))) if na1 and na2 else 0.0,
        float(c1 == c2),
        float(ne and ae),
    ]


RecordLookup = Dict[str, Tuple]  # eid → (norm_name, norm_addr, country, orig_name, orig_addr)


def df_to_lookup(df: pd.DataFrame) -> RecordLookup:
    """Fast conversion of DataFrame to lookup dict without iterrows."""
    return {
        eid: (norm_n, norm_a, cty, b_name, b_addr)
        for eid, norm_n, norm_a, cty, b_name, b_addr in zip(
            df["entity_id"], df["norm_name"], df["norm_addr"],
            df["country"], df["business_name"], df["business_address"]
        )
    }


def compute_features(
    pairs: List[Tuple[str, str]],
    lk1: RecordLookup,
    lk23: RecordLookup,
) -> np.ndarray:
    rows = []
    for s1_id, s23_id in pairs:
        r1  = lk1.get(s1_id)
        r23 = lk23.get(s23_id)
        if r1 is None or r23 is None:
            rows.append([0.0] * len(FEATURE_NAMES))
        else:
            rows.append(featurize(*r1, *r23))
    return np.array(rows, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_gt(path: Path) -> Dict[str, Set[str]]:
    """Fast ground truth loader using standard file streaming."""
    gt_map: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for i, line in enumerate(f):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            ids = parts[1] if len(parts) > 1 else ""
            gt_map[s1_id] = set(ids.split(",")) if ids else set()
            if (i + 1) % 500_000 == 0:
                log.info("  Loaded %d GT records...", i + 1)
    log.info("  Total ground truth S1 records: %d", len(gt_map))
    return gt_map


def recall_at_k(cands: Dict[str, Set[str]], gt: Dict[str, Set[str]]) -> float:
    total = hit = 0
    for s1_id, actual in gt.items():
        if not actual: continue
        total += len(actual)
        hit   += len(actual & cands.get(s1_id, set()))
    return hit / max(total, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    train_sample: int = TRAIN_SAMPLE,
    val_frac: float = VAL_FRAC,
    max_cands: int = MAX_CANDS,
    max_tok1: int = MAX_TOK1,
):
    OUTPUT_DIR.mkdir(exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    # ── 1. Load GT ─────────────────────────────────────────────────────────
    log.info("=== 1/7: Loading ground truth ===")
    t0 = time.time()
    gt_map = load_gt(DATA_TRAIN / "train_ground_truth.tsv")
    all_s1_ids = list(gt_map.keys())
    log.info("  Loaded %d S1 entities in %.1fs", len(all_s1_ids), time.time() - t0)

    # ── 2. Sample Train / Val S1 ───────────────────────────────────────────
    log.info("=== 2/7: Sampling S1 entities ===")
    sample_n  = min(train_sample, len(all_s1_ids))
    sampled   = set(random.sample(all_s1_ids, sample_n))
    val_n     = int(sample_n * val_frac)
    val_ids   = set(random.sample(sorted(sampled), val_n))
    tr_ids    = sampled - val_ids
    log.info("  Sample size: %d total (%d train, %d val)", sample_n, len(tr_ids), len(val_ids))

    # ── 3. Load & preprocess sample S1 ─────────────────────────────────────
    log.info("=== 3/7: Loading and preprocessing sample S1 ===")
    t0 = time.time()
    tr_s1_chunks = []
    chunk_idx = 0
    for chunk in pd.read_csv(DATA_TRAIN / "train_source1.tsv", sep="\t", dtype=str, chunksize=STREAM_CHUNK):
        chunk = chunk.fillna("")
        sub = chunk[chunk["entity_id"].isin(sampled)].copy()
        if not sub.empty:
            sub["norm_name"] = vec_norm_name(sub["business_name"])
            sub["norm_addr"] = vec_norm_addr(sub["business_address"])
            sub["tok1_name"] = sub["norm_name"].apply(tok1)
            sub["tok2_name"] = sub["norm_name"].apply(tok2)
            tr_s1_chunks.append(sub[["entity_id", "country", "business_name", "business_address",
                                     "norm_name", "norm_addr", "tok1_name", "tok2_name"]])
        chunk_idx += 1
        if chunk_idx % 3 == 0:
            log.info("  Processed %d train_source1 chunks...", chunk_idx)

    sample_s1 = pd.concat(tr_s1_chunks, ignore_index=True)
    del tr_s1_chunks
    gc.collect()
    log.info("  Preprocessed %d sample S1 records in %.1fs", len(sample_s1), time.time() - t0)

    tr_s1  = sample_s1[sample_s1["entity_id"].isin(tr_ids)].reset_index(drop=True)
    val_s1 = sample_s1[sample_s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    # Collect blocking keys for fast filtering of train S2/S3
    s1_countries = set(sample_s1["country"])
    s1_names     = set(sample_s1["norm_name"]) - {""}
    s1_tok2      = set(sample_s1["tok2_name"]) - {""}
    s1_tok1      = set(sample_s1["tok1_name"]) - {""}
    s1_addrs     = set(sample_s1["norm_addr"]) - {""}
    s1_pos_ids   = {cid for s1_id in sampled for cid in gt_map.get(s1_id, set())}
    del sample_s1
    gc.collect()

    # ── 4. Load & preprocess train S2/S3 (filtered to candidate pool) ───────
    log.info("=== 4/7: Streaming train S2/S3 candidates ===")
    t0 = time.time()
    tr_s23_chunks = []
    for path in [DATA_TRAIN / "train_source2.tsv", DATA_TRAIN / "train_source3.tsv"]:
        log.info("  Reading %s …", path.name)
        c_num = 0
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=STREAM_CHUNK):
            chunk = chunk.fillna("")
            # Fast filter: must match country or be an explicit positive GT match
            c_mask = chunk["entity_id"].isin(s1_pos_ids) | chunk["country"].isin(s1_countries)
            chunk = chunk[c_mask].copy()
            if chunk.empty:
                continue

            chunk["norm_name"] = vec_norm_name(chunk["business_name"])
            chunk["norm_addr"] = vec_norm_addr(chunk["business_address"])
            chunk["tok1_name"] = chunk["norm_name"].apply(tok1)
            chunk["tok2_name"] = chunk["norm_name"].apply(tok2)

            # Filter: only keep if it is a positive GT ID or matches one of the blocking keys
            keep_mask = (
                chunk["entity_id"].isin(s1_pos_ids)
                | chunk["norm_name"].isin(s1_names)
                | chunk["tok2_name"].isin(s1_tok2)
                | chunk["norm_addr"].isin(s1_addrs)
                | chunk["tok1_name"].isin(s1_tok1)
            )
            filtered = chunk[keep_mask]
            if not filtered.empty:
                tr_s23_chunks.append(filtered[["entity_id", "country", "business_name", "business_address",
                                               "norm_name", "norm_addr", "tok1_name", "tok2_name"]])
            c_num += 1
            if c_num % 3 == 0:
                log.info("    %s: chunk %d processed...", path.name, c_num)

    tr_s23 = pd.concat(tr_s23_chunks, ignore_index=True)
    del tr_s23_chunks
    gc.collect()
    log.info("  Filtered train S2+S3: %d candidate rows in %.1fs (~%.1f MB)",
             len(tr_s23), time.time() - t0, tr_s23.memory_usage(deep=True).sum() / 1e6)

    # ── 5. Blocking on Train & Val ─────────────────────────────────────────
    log.info("=== 5/7: Blocking train and validation sets ===")
    tr_cands  = block_by_join(tr_s1,  tr_s23, max_cands, max_tok1)
    val_cands = block_by_join(val_s1, tr_s23, max_cands, max_tok1)

    tr_gt  = {k: gt_map[k] for k in tr_ids  if k in gt_map}
    val_gt = {k: gt_map[k] for k in val_ids if k in gt_map}
    tr_rec  = recall_at_k(tr_cands,  tr_gt)
    val_rec = recall_at_k(val_cands, val_gt)
    log.info("  Train blocking recall @ %d: %.4f", max_cands, tr_rec)
    log.info("  Val   blocking recall @ %d: %.4f", max_cands, val_rec)

    # ── 6. Feature computation & Model training ───────────────────────────
    log.info("=== 6/7: Feature extraction and model training ===")
    s1_lk: RecordLookup = df_to_lookup(pd.concat([tr_s1, val_s1], ignore_index=True))

    needed_s23: Set[str] = set()
    for v in tr_cands.values():  needed_s23.update(v)
    for v in val_cands.values(): needed_s23.update(v)
    for s1_id in sampled:        needed_s23.update(gt_map.get(s1_id, set()))

    sub_tr_s23 = tr_s23[tr_s23["entity_id"].isin(needed_s23)]
    s23_lk: RecordLookup = {
        eid: (norm_n, norm_a, cty, b_name, b_addr)
        for eid, norm_n, norm_a, cty, b_name, b_addr in zip(
            sub_tr_s23["entity_id"], sub_tr_s23["norm_name"], sub_tr_s23["norm_addr"],
            sub_tr_s23["country"], sub_tr_s23["business_name"], sub_tr_s23["business_address"]
        )
    }
    del tr_s23, sub_tr_s23, needed_s23
    gc.collect()

    tr_pairs: List[Tuple] = []
    tr_labels: List[int]  = []
    for s1_id, cset in tr_cands.items():
        pos = gt_map.get(s1_id, set())
        for cid in cset:
            tr_pairs.append((s1_id, cid))
            tr_labels.append(1 if cid in pos else 0)
        for pos_id in pos:
            if pos_id not in cset and pos_id in s23_lk:
                tr_pairs.append((s1_id, pos_id))
                tr_labels.append(1)

    y_tr = np.array(tr_labels, dtype=np.int32)
    log.info("  Generated %d training pairs (%d positive, %.1f%%)",
             len(y_tr), y_tr.sum(), 100 * y_tr.sum() / max(len(y_tr), 1))

    log.info("  Computing training features...")
    X_tr_parts = []
    for start in range(0, len(tr_pairs), FEAT_BATCH):
        X_tr_parts.append(compute_features(tr_pairs[start:start+FEAT_BATCH], s1_lk, s23_lk))
    X_tr = np.vstack(X_tr_parts)
    del X_tr_parts, tr_pairs, tr_labels
    gc.collect()

    log.info("  Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=7, min_samples_leaf=20,
        class_weight="balanced", random_state=SEED, n_iter_no_change=25,
        validation_fraction=0.05,
    )
    clf.fit(X_tr, y_tr)
    log.info("  Classifier fitted (%d trees)", clf.n_iter_)
    del X_tr, y_tr
    gc.collect()

    # Threshold tuning
    log.info("  Tuning decision threshold on validation set...")
    val_pairs = [(s1_id, cid) for s1_id, cset in val_cands.items() for cid in cset]
    X_val_parts = []
    for start in range(0, len(val_pairs), FEAT_BATCH):
        X_val_parts.append(compute_features(val_pairs[start:start+FEAT_BATCH], s1_lk, s23_lk))
    X_val = np.vstack(X_val_parts) if X_val_parts else np.zeros((0, len(FEATURE_NAMES)))
    del X_val_parts

    val_sc = clf.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    del X_val

    best_t = tune_threshold(val_pairs, val_sc, val_gt)

    pred_map: Dict[str, Set[str]] = {}
    for (s1_id, cid), sc in zip(val_pairs, val_sc):
        if sc >= best_t:
            pred_map.setdefault(s1_id, set()).add(cid)
    for s1_id in val_ids:
        pred_map.setdefault(s1_id, set())
    val_f05 = macro_f05(pred_map, val_gt)
    log.info("  Optimal threshold: %.3f (Validation Macro F0.5 = %.4f)", best_t, val_f05)

    del val_pairs, val_sc, pred_map, s1_lk, s23_lk, tr_s1, val_s1, tr_cands, val_cands, gt_map
    gc.collect()

    # ── 7. Preprocess Test, Score, and Output ──────────────────────────────
    log.info("=== 7/7: Scoring test set and generating submissions ===")
    log.info("  Preprocessing test_source1.tsv...")
    t0 = time.time()
    test_s1_chunks = []
    for chunk in pd.read_csv(DATA_TEST / "test_source1.tsv", sep="\t", dtype=str, chunksize=STREAM_CHUNK):
        chunk = chunk.fillna("")
        chunk["norm_name"] = vec_norm_name(chunk["business_name"])
        chunk["norm_addr"] = vec_norm_addr(chunk["business_address"])
        chunk["tok1_name"] = chunk["norm_name"].apply(tok1)
        chunk["tok2_name"] = chunk["norm_name"].apply(tok2)
        test_s1_chunks.append(chunk[["entity_id", "country", "business_name", "business_address",
                                     "norm_name", "norm_addr", "tok1_name", "tok2_name"]])
    test_s1 = pd.concat(test_s1_chunks, ignore_index=True)
    del test_s1_chunks
    gc.collect()
    log.info("  Test S1: %d entities in %.1fs", len(test_s1), time.time() - t0)

    test_s1_ids_ordered = test_s1["entity_id"].tolist()
    test_countries = set(test_s1["country"])
    test_names     = set(test_s1["norm_name"]) - {""}
    test_tok2      = set(test_s1["tok2_name"]) - {""}
    test_tok1      = set(test_s1["tok1_name"]) - {""}
    test_addrs     = set(test_s1["norm_addr"]) - {""}

    log.info("  Preprocessing and filtering test S2/S3 against test S1 blocking keys...")
    test_s23_chunks = []
    for path in [DATA_TEST / "test_source2.tsv", DATA_TEST / "test_source3.tsv"]:
        log.info("  Reading %s …", path.name)
        c_num = 0
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=STREAM_CHUNK):
            chunk = chunk.fillna("")
            chunk = chunk[chunk["country"].isin(test_countries)].copy()
            if chunk.empty:
                continue

            chunk["norm_name"] = vec_norm_name(chunk["business_name"])
            chunk["norm_addr"] = vec_norm_addr(chunk["business_address"])
            chunk["tok1_name"] = chunk["norm_name"].apply(tok1)
            chunk["tok2_name"] = chunk["norm_name"].apply(tok2)

            keep_mask = (
                chunk["norm_name"].isin(test_names)
                | chunk["tok2_name"].isin(test_tok2)
                | chunk["norm_addr"].isin(test_addrs)
                | chunk["tok1_name"].isin(test_tok1)
            )
            filtered = chunk[keep_mask]
            if not filtered.empty:
                test_s23_chunks.append(filtered[["entity_id", "country", "business_name", "business_address",
                                                 "norm_name", "norm_addr", "tok1_name", "tok2_name"]])
            c_num += 1
            if c_num % 5 == 0:
                log.info("    %s: chunk %d processed...", path.name, c_num)

    test_s23 = pd.concat(test_s23_chunks, ignore_index=True)
    del test_s23_chunks, test_countries, test_names, test_tok2, test_tok1, test_addrs
    gc.collect()
    log.info("  Filtered test S2+S3: %d candidate rows (~%.0f MB)",
             len(test_s23), test_s23.memory_usage(deep=True).sum() / 1e6)

    # Block test
    log.info("  Blocking test set...")
    test_cands = block_by_join(test_s1, test_s23, max_cands, max_tok1)
    n_with_cands = sum(1 for v in test_cands.values() if v)
    log.info("  %d / %d test S1 have candidates (%.1f%%)",
             n_with_cands, len(test_s1), 100 * n_with_cands / max(len(test_s1), 1))

    # Build test lookups
    test_s1_lk: RecordLookup = df_to_lookup(test_s1)
    del test_s1
    gc.collect()

    needed_test_s23: Set[str] = set()
    for v in test_cands.values():
        needed_test_s23.update(v)

    sub_test_s23 = test_s23[test_s23["entity_id"].isin(needed_test_s23)]
    test_s23_lk: RecordLookup = {
        eid: (norm_n, norm_a, cty, b_name, b_addr)
        for eid, norm_n, norm_a, cty, b_name, b_addr in zip(
            sub_test_s23["entity_id"], sub_test_s23["norm_name"], sub_test_s23["norm_addr"],
            sub_test_s23["country"], sub_test_s23["business_name"], sub_test_s23["business_address"]
        )
    }
    del test_s23, sub_test_s23, needed_test_s23
    gc.collect()

    # Score test pairs
    all_test_pairs = [
        (s1_id, cid)
        for s1_id, cset in test_cands.items()
        for cid in cset
    ]
    log.info("  Scoring %d candidate pairs...", len(all_test_pairs))

    match_map: Dict[str, Set[str]] = {}
    t0 = time.time()
    for start in range(0, len(all_test_pairs), FEAT_BATCH):
        batch = all_test_pairs[start:start+FEAT_BATCH]
        X_b = compute_features(batch, test_s1_lk, test_s23_lk)
        sc_b = clf.predict_proba(X_b)[:, 1]
        for (s1_id, cid), sc in zip(batch, sc_b):
            if sc >= best_t:
                match_map.setdefault(s1_id, set()).add(cid)
        if start > 0 and (start % (FEAT_BATCH * 10) == 0 or start + FEAT_BATCH >= len(all_test_pairs)):
            log.info("  Scored %d / %d pairs (%.1fs)", min(start + FEAT_BATCH, len(all_test_pairs)),
                     len(all_test_pairs), time.time() - t0)

    log.info("  Scoring complete in %.1fs", time.time() - t0)
    del test_s1_lk, test_s23_lk, all_test_pairs
    gc.collect()

    # Write output TSVs
    log.info("  Writing submission files...")
    matching_rows, cand_rows = [], []
    for s1_id in test_s1_ids_ordered:
        matched = sorted(match_map.get(s1_id, set()))
        cands   = sorted(test_cands.get(s1_id, set()))
        matching_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matched)})
        cand_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(cands)})

    matching_path = OUTPUT_DIR / "matching_results.tsv"
    cand_path     = OUTPUT_DIR / "candidate_pairs.tsv"

    pd.DataFrame(matching_rows).to_csv(matching_path, sep="\t", index=False)
    pd.DataFrame(cand_rows).to_csv(cand_path, sep="\t", index=False)

    n_matched   = sum(1 for v in match_map.values() if v)
    n_singleton = len(test_s1_ids_ordered) - n_matched

    log.info("=" * 60)
    log.info("PIPELINE SUMMARY:")
    log.info("  Train blocking recall: %.4f", tr_rec)
    log.info("  Val   blocking recall: %.4f", val_rec)
    log.info("  Val Macro F0.5:        %.4f (threshold = %.3f)", val_f05, best_t)
    log.info("  Test S1 with matches:  %d / %d (%.1f%%)", n_matched, len(test_s1_ids_ordered),
             100 * n_matched / max(len(test_s1_ids_ordered), 1))
    log.info("  Test singletons (0 m): %d", n_singleton)
    log.info("  Output saved to:")
    log.info("    %s", matching_path)
    log.info("    %s", cand_path)
    log.info("=" * 60)
    return val_f05, best_t, tr_rec, val_rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument(
        "--train-sample",
        type=int,
        default=TRAIN_SAMPLE,
        help=f"Number of S1 train entities to sample (default: {TRAIN_SAMPLE:,})",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=VAL_FRAC,
        help=f"Fraction of sampled S1 used for validation (default: {VAL_FRAC})",
    )
    parser.add_argument(
        "--max-cands",
        type=int,
        default=MAX_CANDS,
        help=f"Maximum candidates per S1 entity (default: {MAX_CANDS})",
    )
    parser.add_argument(
        "--max-tok1",
        type=int,
        default=MAX_TOK1,
        help=f"Maximum single-token candidates (default: {MAX_TOK1})",
    )
    args = parser.parse_args()

    run_pipeline(
        train_sample=args.train_sample,
        val_frac=args.val_frac,
        max_cands=args.max_cands,
        max_tok1=args.max_tok1,
    )
