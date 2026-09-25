"""
Pairwise feature engineering for the matching classifier.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from .normalize import normalize_name, normalize_address, numeric_tokens


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_overlap(a: str, b: str) -> float:
    ta = frozenset(a.split())
    tb = frozenset(b.split())
    return _jaccard(ta, tb)


def _lcp_ratio(a: str, b: str) -> float:
    """Longest common prefix ratio."""
    if not a or not b:
        return 0.0
    i = 0
    min_len = min(len(a), len(b))
    while i < min_len and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b))


def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard."""
    if len(a) < n or len(b) < n:
        return 0.0
    sa = frozenset(a[i : i + n] for i in range(len(a) - n + 1))
    sb = frozenset(b[i : i + n] for i in range(len(b) - n + 1))
    return _jaccard(sa, sb)


FEATURE_NAMES: List[str] = [
    # Name features
    "name_token_overlap",
    "name_char3_jaccard",
    "name_char4_jaccard",
    "name_lcp_ratio",
    "name_exact_match",
    "name_len_ratio",
    "name_first_token_match",
    "name_last_token_match",
    # Address features
    "addr_token_overlap",
    "addr_char3_jaccard",
    "addr_exact_match",
    "addr_numeric_overlap",
    "addr_len_ratio",
    # Country
    "country_match",
    # Combined
    "name_addr_both_exact",
]


def compute_features(
    pairs: List[Tuple[str, str]],
    df1: pd.DataFrame,
    df23: pd.DataFrame,
) -> np.ndarray:
    """
    Compute feature matrix for a list of (s1_id, s23_id) pairs.

    Returns shape (n_pairs, n_features).
    """
    # Build lookup dicts
    rec1 = df1.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    rec23 = df23.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    rows = []
    for s1_id, s23_id in pairs:
        r1 = rec1.get(s1_id, {})
        r2 = rec23.get(s23_id, {})

        n1 = normalize_name(r1.get("business_name", ""))
        n2 = normalize_name(r2.get("business_name", ""))
        a1 = normalize_address(r1.get("business_address", ""))
        a2 = normalize_address(r2.get("business_address", ""))
        c1 = str(r1.get("country", ""))
        c2 = str(r2.get("country", ""))

        tok1 = n1.split()
        tok2 = n2.split()

        name_tok_ov = _token_overlap(n1, n2)
        name_c3 = _char_ngram_jaccard(n1, n2, 3)
        name_c4 = _char_ngram_jaccard(n1, n2, 4)
        name_lcp = _lcp_ratio(n1, n2)
        name_exact = float(n1 == n2 and bool(n1))
        name_len_ratio = (
            min(len(n1), len(n2)) / max(len(n1), len(n2))
            if n1 and n2
            else 0.0
        )
        name_first = float(bool(tok1) and bool(tok2) and tok1[0] == tok2[0])
        name_last = float(bool(tok1) and bool(tok2) and tok1[-1] == tok2[-1])

        addr_tok_ov = _token_overlap(a1, a2)
        addr_c3 = _char_ngram_jaccard(a1, a2, 3)
        addr_exact = float(a1 == a2 and bool(a1))
        num1 = numeric_tokens(r1.get("business_address", ""))
        num2 = numeric_tokens(r2.get("business_address", ""))
        addr_num_ov = _jaccard(num1, num2)
        addr_len_ratio = (
            min(len(a1), len(a2)) / max(len(a1), len(a2))
            if a1 and a2
            else 0.0
        )

        country_match = float(c1 == c2)
        both_exact = float(name_exact and addr_exact)

        rows.append([
            name_tok_ov, name_c3, name_c4, name_lcp, name_exact,
            name_len_ratio, name_first, name_last,
            addr_tok_ov, addr_c3, addr_exact, addr_num_ov, addr_len_ratio,
            country_match,
            both_exact,
        ])

    return np.array(rows, dtype=np.float32)


def build_training_pairs(
    df1: pd.DataFrame,
    df23: pd.DataFrame,
    gt: pd.DataFrame,
    candidates: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from candidates, using ground truth for labels.

    Positive pairs: S1 ↔ S2/S3 pairs where a match is known.
    Negative pairs: candidates that are NOT in ground truth.

    Returns X (n_pairs, n_features), y (n_pairs,)
    """
    # Build gt lookup: s1_id → set of matched ids
    gt_map = {}
    for _, row in gt.iterrows():
        ids_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        gt_map[row["source1_entity_id"]] = (
            set(ids_str.split(",")) if ids_str else set()
        )

    pairs = []
    labels = []

    for s1_id, cand_set in candidates.items():
        positives = gt_map.get(s1_id, set())
        for cid in cand_set:
            pairs.append((s1_id, cid))
            labels.append(1 if cid in positives else 0)

        # Add any positive pairs NOT in candidates (to fix recall during training)
        for pos_id in positives:
            if pos_id not in cand_set:
                pairs.append((s1_id, pos_id))
                labels.append(1)

    X = compute_features(pairs, df1, df23)
    y = np.array(labels, dtype=np.int32)
    return X, y, pairs
