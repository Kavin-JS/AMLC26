"""
Memory-efficient entity resolution pipeline using vectorized pandas joins.

Architecture:
  - Build compact DataFrames with (block_key, country, entity_id) for S2/S3
  - Join against S1 block keys using pd.merge (vectorized)
  - Process in chunks to cap memory
  - Train on 200K S1 sample; use swap (15GB available) for index storage

Memory strategy:
  - Index DataFrames: ~200 bytes/record × 10M = ~2GB — processed in chunks
  - Use swap liberally since 15GB swap is available
  - Vectorized norm via str operations (no Python loops per row)
"""

from __future__ import annotations

import gc
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT       = Path(__file__).resolve().parents[4]
DATA_TRAIN = ROOT / "student_resource" / "dataset" / "train"
DATA_TEST  = ROOT / "student_resource" / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
SCRATCH    = ROOT / "data" / "_pipeline_scratch"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize_name, normalize_address, numeric_tokens
from metrics   import macro_f05, tune_threshold

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
MAX_TOK1      = 10      # cap single-token candidates (noisy)
STREAM_CHUNK  = 500_000
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
# Vectorized normalization (pandas str ops — much faster than iterrows)
# ═════════════════════════════════════════════════════════════════════════════

import re, unicodedata

_LEGAL_MAP = [
    (re.compile(r, re.I), s) for r, s in [
        (r"\bincorporated\b", "inc"), (r"\bcorporation\b", "corp"),
        (r"\blimited liability company\b", "llc"), (r"\blimited liability partnership\b", "llp"),
        (r"\bprivate limited\b", "pvt ltd"), (r"\bprivate ltd\b", "pvt ltd"),
        (r"\bprivate\b", "pvt"), (r"\blimited\b", "ltd"), (r"\bcompany\b", "co"),
        (r"\bbrothers\b", "bros"), (r"\band\b", "&"), (r"\benterprises?\b", "ent"),
        (r"\binternational\b", "intl"), (r"\bnational\b", "natl"),
        (r"\bassociates\b", "assoc"), (r"\bservices\b", "svcs"),
        (r"\bsolutions\b", "soln"), (r"\bindustries\b", "ind"),
        (r"\bgroup\b", "grp"), (r"\bmanagement\b", "mgmt"),
    ]
]
_ADDR_MAP = [
    (re.compile(r, re.I), s) for r, s in [
        (r"\bstreet\b", "st"), (r"\broad\b", "rd"), (r"\bavenue\b", "ave"),
        (r"\bboulevard\b", "blvd"), (r"\bdrive\b", "dr"), (r"\bcourt\b", "ct"),
        (r"\blane\b", "ln"), (r"\bplace\b", "pl"), (r"\bsuite\b", "ste"),
        (r"\bapartment\b", "apt"), (r"\bbuilding\b", "bldg"),
        (r"\bnorth\b", "n"), (r"\bsouth\b", "s"), (r"\beast\b", "e"), (r"\bwest\b", "w"),
        (r"\bhighway\b", "hwy"), (r"\bparkway\b", "pkwy"), (r"\bnagar\b", "ngr"),
        (r"\bcolony\b", "col"),
    ]
]
_PUNC = re.compile(r"[^\w\s&]")
_WS   = re.compile(r"\s+")


def _to_ascii(t: str) -> str:
    n = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return n if n.strip() else t


def vec_norm_name(series: pd.Series) -> pd.Series:
    """Vectorized name normalization (much faster than row-by-row)."""
    s = series.fillna("").str.strip().str.lower()
    # ASCII transliteration (apply per row, but str is already fast)
    s = s.apply(lambda t: _to_ascii(t) if t else t)
    s = s.str.replace(_PUNC, " ", regex=True)
    for pat, rep in _LEGAL_MAP:
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(_WS, " ", regex=True).str.strip()
    return s


def vec_norm_addr(series: pd.Series) -> pd.Series:
    s = series.fillna("").str.strip().str.lower()
    s = s.apply(lambda t: _to_ascii(t) if t else t)
    s = s.str.replace(_PUNC, " ", regex=True)
    for pat, rep in _ADDR_MAP:
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(_WS, " ", regex=True).str.strip()
    return s


def tok1(s: str) -> str:
    toks = s.split()
    return toks[0] if toks else ""


def tok2(s: str) -> str:
    toks = s.split()
    return " ".join(toks[:2]) if len(toks) >= 2 else ""


# ═════════════════════════════════════════════════════════════════════════════
# Index building and blocking (join-based)
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_source(path: Path, chunk_size: int = STREAM_CHUNK) -> pd.DataFrame:
    """Load and preprocess a source TSV: add norm_name, norm_addr, tok1_name, tok2_name."""
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunk_size):
        chunk = chunk.fillna("")
        chunk["norm_name"] = vec_norm_name(chunk["business_name"])
        chunk["norm_addr"] = vec_norm_addr(chunk["business_address"])
        chunk["tok1_name"] = chunk["norm_name"].apply(tok1)
        chunk["tok2_name"] = chunk["norm_name"].apply(tok2)
        chunks.append(chunk[["entity_id", "country", "business_name", "business_address",
                              "norm_name", "norm_addr", "tok1_name", "tok2_name"]])
    return pd.concat(chunks, ignore_index=True)


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

    def _add_matches(merged: pd.DataFrame) -> None:
        for _, row in merged.iterrows():
            s1_id = row["entity_id_x"]
            s23_id = row["entity_id_y"]
            if len(cands.get(s1_id, set())) < max_cands:
                cands.setdefault(s1_id, set()).add(s23_id)

    def _merge(key: str) -> pd.DataFrame:
        return pd.merge(
            s1[["entity_id", key, "country"]].rename(columns={"entity_id": "entity_id_x"}),
            s23[["entity_id", key, "country"]].rename(columns={"entity_id": "entity_id_y"}),
            on=[key, "country"],
        )[["entity_id_x", "entity_id_y"]]

    # Pass 1: exact norm_name
    log.debug("  exact name join")
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
            cands.setdefault(s1_id, set()).add(s23_id)

    # Pass 2: exact norm_addr
    log.debug("  exact addr join")
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
            if s23_id not in cands.get(s1_id, set()):
                cands.setdefault(s1_id, set()).add(s23_id)

    # Pass 3: tok2 join
    log.debug("  tok2 join")
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
            if s23_id not in cands.get(s1_id, set()) and len(cands.get(s1_id, set())) < max_cands:
                cands.setdefault(s1_id, set()).add(s23_id)

    # Pass 4: tok1 join (capped at max_tok1 per entity)
    # Only for S1 entities that still have < MIN_CANDS candidates
    log.debug("  tok1 join")
    sparse_s1 = s1[s1["entity_id"].map(lambda e: len(cands.get(e, set())) < 5) & (s1["tok1_name"] != "")]
    s23_t1 = s23[s23["tok1_name"] != ""]
    if not sparse_s1.empty and not s23_t1.empty:
        m = pd.merge(
            sparse_s1[["entity_id", "tok1_name", "country"]],
            s23_t1[["entity_id", "tok1_name", "country"]],
            on=["tok1_name", "country"],
            suffixes=("_1", "_23"),
        )
        # Count and cap
        counts: Dict[str, int] = {}
        for s1_id, s23_id in zip(m["entity_id_1"], m["entity_id_23"]):
            n = counts.get(s1_id, 0)
            if n < max_tok1 and s23_id not in cands.get(s1_id, set()):
                cands.setdefault(s1_id, set()).add(s23_id)
                counts[s1_id] = n + 1

    return cands


# ═════════════════════════════════════════════════════════════════════════════
# Feature engineering — vectorized where possible
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


def featurize(nn1, na1, c1, oa1, nn2, na2, c2, oa2) -> List[float]:
    t1 = nn1.split(); t2 = nn2.split()
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
    return {
        row["entity_id"]: (
            row["norm_name"], row["norm_addr"], row["country"],
            row["business_name"], row["business_address"],
        )
        for _, row in df.iterrows()
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
    gt_map: Dict[str, Set[str]] = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=STREAM_CHUNK):
        chunk = chunk.fillna("")
        for _, row in chunk.iterrows():
            ids = row["matched_entity_ids"]
            gt_map[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()
    return gt_map


def recall_at_k(cands: Dict[str, Set[str]], gt: Dict[str, Set[str]]) -> float:
    total = hit = 0
    for s1_id, actual in gt.items():
        if not actual: continue
        total += len(actual)
        hit   += len(actual & cands.get(s1_id, set()))
    return hit / max(total, 1)


def load_subset_lookup(paths: List[Path], needed: Set[str]) -> RecordLookup:
    """Load only the needed entity rows and return a RecordLookup."""
    lk: RecordLookup = {}
    for path in paths:
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=STREAM_CHUNK):
            chunk = chunk.fillna("")
            sub = chunk[chunk["entity_id"].isin(needed)].copy()
            if not sub.empty:
                sub["norm_name"] = vec_norm_name(sub["business_name"])
                sub["norm_addr"] = vec_norm_addr(sub["business_address"])
                for _, row in sub.iterrows():
                    lk[row["entity_id"]] = (
                        row["norm_name"], row["norm_addr"], row["country"],
                        row["business_name"], row["business_address"],
                    )
            if len(lk) >= len(needed):
                break
    return lk


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    OUTPUT_DIR.mkdir(exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    # ── 1. Load GT (streaming) ─────────────────────────────────────────────
    log.info("=== Loading ground truth ===")
    gt_map = load_gt(DATA_TRAIN / "train_ground_truth.tsv")
    all_s1_ids = list(gt_map.keys())
    log.info("  %d S1 entities", len(all_s1_ids))

    # ── 2. Sample ──────────────────────────────────────────────────────────
    sample_n  = min(TRAIN_SAMPLE, len(all_s1_ids))
    sampled   = set(random.sample(all_s1_ids, sample_n))
    val_n     = int(sample_n * VAL_FRAC)
    val_ids   = set(random.sample(sorted(sampled), val_n))
    tr_ids    = sampled - val_ids
    log.info("  Sample: %d train, %d val", len(tr_ids), len(val_ids))

    # ── 3. Load & preprocess S2/S3 for training ────────────────────────────
    log.info("=== Preprocessing train S2/S3 ===")
    t0 = time.time()
    tr_s23_chunks = []
    for path in [DATA_TRAIN/"train_source2.tsv", DATA_TRAIN/"train_source3.tsv"]:
        log.info("  %s …", path.name)
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=STREAM_CHUNK):
            chunk = chunk.fillna("")
            chunk["norm_name"] = vec_norm_name(chunk["business_name"])
            chunk["norm_addr"] = vec_norm_addr(chunk["business_address"])
            chunk["tok1_name"] = chunk["norm_name"].apply(tok1)
            chunk["tok2_name"] = chunk["norm_name"].apply(tok2)
            tr_s23_chunks.append(chunk[["entity_id","country","business_name","business_address",
                                        "norm_name","norm_addr","tok1_name","tok2_name"]])
    tr_s23 = pd.concat(tr_s23_chunks, ignore_index=True)
    del tr_s23_chunks
    gc.collect()
    log.info("  S2+S3: %d rows in %.1fs, ~%.0f MB", len(tr_s23), time.time()-t0,
             tr_s23.memory_usage(deep=True).sum()/1e6)

    # ── 4. Load & preprocess sample S1 ────────────────────────────────────
    log.info("=== Loading sample S1 ===")
    tr_s1_chunks = []
    for chunk in pd.read_csv(DATA_TRAIN/"train_source1.tsv", sep="\t", dtype=str, chunksize=STREAM_CHUNK):
        chunk = chunk.fillna("")
        sub = chunk[chunk["entity_id"].isin(sampled)].copy()
        if not sub.empty:
            sub["norm_name"] = vec_norm_name(sub["business_name"])
            sub["norm_addr"] = vec_norm_addr(sub["business_address"])
            sub["tok1_name"] = sub["norm_name"].apply(tok1)
            sub["tok2_name"] = sub["norm_name"].apply(tok2)
            tr_s1_chunks.append(sub[["entity_id","country","business_name","business_address",
                                     "norm_name","norm_addr","tok1_name","tok2_name"]])
    sample_s1 = pd.concat(tr_s1_chunks, ignore_index=True)
    del tr_s1_chunks
    log.info("  %d sample S1 rows", len(sample_s1))

    tr_s1  = sample_s1[sample_s1["entity_id"].isin(tr_ids)].reset_index(drop=True)
    val_s1 = sample_s1[sample_s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    del sample_s1
    gc.collect()

    # ── 5. Block by join ──────────────────────────────────────────────────
    log.info("=== Blocking (train join) ===")
    tr_cands  = block_by_join(tr_s1,  tr_s23, MAX_CANDS, MAX_TOK1)
    log.info("=== Blocking (val join) ===")
    val_cands = block_by_join(val_s1, tr_s23, MAX_CANDS, MAX_TOK1)

    tr_gt  = {k: gt_map[k] for k in tr_ids  if k in gt_map}
    val_gt = {k: gt_map[k] for k in val_ids if k in gt_map}
    tr_rec  = recall_at_k(tr_cands,  tr_gt)
    val_rec = recall_at_k(val_cands, val_gt)
    log.info("  Train blocking recall: %.4f", tr_rec)
    log.info("  Val   blocking recall: %.4f", val_rec)

    # ── 6. Lookups for feature computation ────────────────────────────────
    log.info("=== Building record lookups ===")
    s1_lk: RecordLookup = df_to_lookup(pd.concat([tr_s1, val_s1], ignore_index=True))

    # Needed S23 IDs
    needed_s23: Set[str] = set()
    for v in tr_cands.values():  needed_s23.update(v)
    for v in val_cands.values(): needed_s23.update(v)
    for s1_id in sampled:        needed_s23.update(gt_map.get(s1_id, set()))
    log.info("  Need %d S23 records for features", len(needed_s23))

    # Build lookup from the already-loaded tr_s23
    s23_lk: RecordLookup = {
        row["entity_id"]: (
            row["norm_name"], row["norm_addr"], row["country"],
            row["business_name"], row["business_address"],
        )
        for _, row in tr_s23[tr_s23["entity_id"].isin(needed_s23)].iterrows()
    }

    del tr_s23
    gc.collect()

    # ── 7. Build training pairs ────────────────────────────────────────────
    log.info("=== Building training pairs ===")
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
    log.info("  %d pairs, %d pos (%.1f%%)", len(y_tr), y_tr.sum(), 100*y_tr.sum()/max(len(y_tr),1))

    log.info("  Computing features …")
    X_tr_parts = []
    for start in range(0, len(tr_pairs), FEAT_BATCH):
        X_tr_parts.append(compute_features(tr_pairs[start:start+FEAT_BATCH], s1_lk, s23_lk))
    X_tr = np.vstack(X_tr_parts)
    del X_tr_parts, tr_pairs, tr_labels
    gc.collect()

    # ── 8. Train ──────────────────────────────────────────────────────────
    log.info("=== Training HistGradientBoostingClassifier ===")
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=7, min_samples_leaf=20,
        class_weight="balanced", random_state=SEED, n_iter_no_change=30,
        validation_fraction=0.05,
    )
    clf.fit(X_tr, y_tr)
    log.info("  Trained (%d iters)", clf.n_iter_)
    del X_tr, y_tr
    gc.collect()

    # ── 9. Threshold tuning ───────────────────────────────────────────────
    log.info("=== Threshold tuning ===")
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
    log.info("  Val macro F0.5 @ %.3f → %.4f", best_t, val_f05)

    del val_pairs, val_sc, pred_map, s1_lk, s23_lk
    del tr_s1, val_s1, tr_cands, val_cands
    gc.collect()

    # ── 10. Build test S2/S3 ──────────────────────────────────────────────
    log.info("=== Preprocessing test S2/S3 ===")
    test_s23_chunks = []
    for path in [DATA_TEST/"test_source2.tsv", DATA_TEST/"test_source3.tsv"]:
        log.info("  %s …", path.name)
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=STREAM_CHUNK):
            chunk = chunk.fillna("")
            chunk["norm_name"] = vec_norm_name(chunk["business_name"])
            chunk["norm_addr"] = vec_norm_addr(chunk["business_address"])
            chunk["tok1_name"] = chunk["norm_name"].apply(tok1)
            chunk["tok2_name"] = chunk["norm_name"].apply(tok2)
            test_s23_chunks.append(chunk[["entity_id","country","business_name","business_address",
                                          "norm_name","norm_addr","tok1_name","tok2_name"]])
    test_s23 = pd.concat(test_s23_chunks, ignore_index=True)
    del test_s23_chunks
    gc.collect()
    log.info("  test S2+S3: %d rows, ~%.0f MB", len(test_s23), test_s23.memory_usage(deep=True).sum()/1e6)

    # ── 11. Load & preprocess test S1 ────────────────────────────────────
    log.info("=== Preprocessing test S1 ===")
    test_s1_chunks = []
    for chunk in pd.read_csv(DATA_TEST/"test_source1.tsv", sep="\t", dtype=str, chunksize=STREAM_CHUNK):
        chunk = chunk.fillna("")
        chunk["norm_name"] = vec_norm_name(chunk["business_name"])
        chunk["norm_addr"] = vec_norm_addr(chunk["business_address"])
        chunk["tok1_name"] = chunk["norm_name"].apply(tok1)
        chunk["tok2_name"] = chunk["norm_name"].apply(tok2)
        test_s1_chunks.append(chunk[["entity_id","country","business_name","business_address",
                                     "norm_name","norm_addr","tok1_name","tok2_name"]])
    test_s1 = pd.concat(test_s1_chunks, ignore_index=True)
    del test_s1_chunks
    gc.collect()
    log.info("  test S1: %d rows, ~%.0f MB", len(test_s1), test_s1.memory_usage(deep=True).sum()/1e6)

    test_s1_ids_ordered = test_s1["entity_id"].tolist()

    # ── 12. Block test by join ─────────────────────────────────────────────
    log.info("=== Blocking test ===")
    test_cands = block_by_join(test_s1, test_s23, MAX_CANDS, MAX_TOK1)
    n_with = sum(1 for v in test_cands.values() if v)
    log.info("  %d / %d test S1 have candidates", n_with, len(test_s1))

    # ── 13. Build test lookups ────────────────────────────────────────────
    log.info("=== Building test lookups ===")
    test_s1_lk: RecordLookup = df_to_lookup(test_s1)
    del test_s1
    gc.collect()

    needed_test_s23: Set[str] = set()
    for v in test_cands.values():
        needed_test_s23.update(v)
    test_s23_lk: RecordLookup = {
        row["entity_id"]: (
            row["norm_name"], row["norm_addr"], row["country"],
            row["business_name"], row["business_address"],
        )
        for _, row in test_s23[test_s23["entity_id"].isin(needed_test_s23)].iterrows()
    }
    del test_s23, needed_test_s23
    gc.collect()

    # ── 14. Score test pairs ──────────────────────────────────────────────
    log.info("=== Scoring test pairs ===")
    all_test_pairs = [
        (s1_id, cid)
        for s1_id, cset in test_cands.items()
        for cid in cset
    ]
    log.info("  %d test pairs", len(all_test_pairs))

    match_map: Dict[str, Set[str]] = {}
    t0 = time.time()
    for start in range(0, len(all_test_pairs), FEAT_BATCH):
        batch = all_test_pairs[start:start+FEAT_BATCH]
        X_b = compute_features(batch, test_s1_lk, test_s23_lk)
        sc_b = clf.predict_proba(X_b)[:, 1]
        for (s1_id, cid), sc in zip(batch, sc_b):
            if sc >= best_t:
                match_map.setdefault(s1_id, set()).add(cid)
        if start % (FEAT_BATCH * 20) == 0 and start > 0:
            log.info("  scored %d / %d", start + FEAT_BATCH, len(all_test_pairs))

    log.info("  Scoring done in %.1fs", time.time() - t0)

    # ── 15. Write output ──────────────────────────────────────────────────
    log.info("=== Writing output ===")
    matching_rows, cand_rows = [], []
    for s1_id in test_s1_ids_ordered:
        matched = sorted(match_map.get(s1_id, set()))
        cands   = sorted(test_cands.get(s1_id, set()))
        matching_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matched)})
        cand_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(cands)})

    pd.DataFrame(matching_rows).to_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False)
    pd.DataFrame(cand_rows).to_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)

    n_matched   = sum(1 for v in match_map.values() if v)
    n_singleton = len(test_s1_ids_ordered) - n_matched

    log.info("=== Summary ===")
    log.info("  Train blocking recall: %.4f", tr_rec)
    log.info("  Val   blocking recall: %.4f", val_rec)
    log.info("  Val macro F0.5 @ %.3f: %.4f", best_t, val_f05)
    log.info("  Test S1 with matches:  %d / %d", n_matched, len(test_s1_ids_ordered))
    log.info("  Test singletons:       %d", n_singleton)
    log.info("Done!")
    return val_f05, best_t, tr_rec, val_rec


if __name__ == "__main__":
    run_pipeline()
