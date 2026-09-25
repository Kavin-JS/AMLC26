"""
Candidate generation (blocking) for entity resolution.

Strategy (multi-pass):
1. Exact normalized name match within same country
2. Exact normalized address match within same country
3. First-token (primary name word) + country blocking
4. TF-IDF character n-gram approximate name matching

Returns Dict[s1_id, Set[s2_or_s3_id]]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Set

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .normalize import normalize_name, normalize_address

logger = logging.getLogger(__name__)


def _build_inverted_index(series: pd.Series, key_fn) -> dict:
    """Build key → list of (idx, entity_id) inverted index."""
    idx = defaultdict(list)
    for row_id, eid in series.items():
        k = key_fn(row_id)
        if k:
            idx[k].append(eid)
    return idx


def _exact_block(
    df1: pd.DataFrame,
    df23: pd.DataFrame,
    col: str,
    norm_fn,
    max_candidates: int = 50,
) -> Dict[str, Set[str]]:
    """Block on exact normalized value of `col` within same country."""
    cands: Dict[str, Set[str]] = defaultdict(set)

    # Build index: (norm_val, country) → list of s2/s3 ids
    index: dict = defaultdict(list)
    for _, row in df23.iterrows():
        nv = norm_fn(row[col])
        if nv:
            index[(nv, str(row["country"]))].append(row["entity_id"])

    for _, row in df1.iterrows():
        nv = norm_fn(row[col])
        if not nv:
            continue
        key = (nv, str(row["country"]))
        matches = index.get(key, [])
        if matches:
            cands[row["entity_id"]].update(matches[:max_candidates])

    return cands


def _token_block(
    df1: pd.DataFrame,
    df23: pd.DataFrame,
    col: str,
    norm_fn,
    n_tokens: int = 2,
    max_candidates: int = 50,
) -> Dict[str, Set[str]]:
    """Block on first n tokens of normalized field."""
    cands: Dict[str, Set[str]] = defaultdict(set)

    index: dict = defaultdict(list)
    for _, row in df23.iterrows():
        nv = norm_fn(row[col])
        tokens = nv.split()
        if len(tokens) >= 1:
            key = (" ".join(tokens[:n_tokens]), str(row["country"]))
            index[key].append(row["entity_id"])

    for _, row in df1.iterrows():
        nv = norm_fn(row[col])
        tokens = nv.split()
        if not tokens:
            continue
        key = (" ".join(tokens[:n_tokens]), str(row["country"]))
        matches = index.get(key, [])
        if matches:
            cands[row["entity_id"]].update(matches[:max_candidates])

    return cands


def _tfidf_block(
    df1: pd.DataFrame,
    df23: pd.DataFrame,
    col: str,
    norm_fn,
    threshold: float = 0.45,
    max_candidates: int = 30,
    batch_size: int = 2000,
) -> Dict[str, Set[str]]:
    """TF-IDF character 3-gram approximate blocking within each country."""
    # pyrefly: ignore [missing-import]
    from scipy.sparse import csr_matrix

    cands: Dict[str, Set[str]] = defaultdict(set)
    countries = df1["country"].unique()

    for country in countries:
        mask1 = df1["country"] == country
        mask23 = df23["country"] == country
        sub1 = df1[mask1].copy()
        sub23 = df23[mask23].copy()

        if sub1.empty or sub23.empty:
            continue

        texts1 = sub1[col].apply(norm_fn).tolist()
        texts23 = sub23[col].apply(norm_fn).tolist()
        ids1 = sub1["entity_id"].tolist()
        ids23 = sub23["entity_id"].tolist()

        # Filter empties
        valid23 = [(t, eid) for t, eid in zip(texts23, ids23) if t]
        if not valid23:
            continue
        texts23_f, ids23_f = zip(*valid23)

        try:
            vect = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True
            )
            mat23 = vect.fit_transform(texts23_f)
            mat23_n = normalize(mat23, norm="l2", copy=False)

            for start in range(0, len(texts1), batch_size):
                batch_texts = texts1[start : start + batch_size]
                batch_ids = ids1[start : start + batch_size]
                valid_mask = [bool(t) for t in batch_texts]
                if not any(valid_mask):
                    continue
                bt = [t for t, v in zip(batch_texts, valid_mask) if v]
                bi = [i for i, v in zip(batch_ids, valid_mask) if v]
                mat1 = vect.transform(bt)
                mat1_n = normalize(mat1, norm="l2", copy=False)
                sim = (mat1_n @ mat23_n.T).toarray()
                for row_idx, s1_id in enumerate(bi):
                    above = np.where(sim[row_idx] >= threshold)[0]
                    if above.size:
                        picks = above[np.argsort(-sim[row_idx][above])[:max_candidates]]
                        cands[s1_id].update(ids23_f[p] for p in picks)
        except Exception as exc:
            logger.warning("TF-IDF block failed for country=%s: %s", country, exc)
            continue

    return cands


def _merge_dicts(*dicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    merged: Dict[str, Set[str]] = defaultdict(set)
    for d in dicts:
        for k, v in d.items():
            merged[k].update(v)
    return merged


def generate_candidates(
    df1: pd.DataFrame,
    df23: pd.DataFrame,
    tfidf_threshold: float = 0.45,
    max_per_entity: int = 100,
) -> Dict[str, Set[str]]:
    """
    Multi-pass blocking: returns {s1_id: set of candidate s2/s3 ids}.

    Args:
        df1:   Source-1 records (entity_id, business_name, business_address, country)
        df23:  Combined Source-2 and Source-3 records
        tfidf_threshold: Cosine similarity threshold for character-ngram blocking
        max_per_entity:  Cap per S1 entity
    """
    logger.info("Pass 1: exact name blocking")
    c1 = _exact_block(df1, df23, "business_name", normalize_name)

    logger.info("Pass 2: exact address blocking")
    c2 = _exact_block(df1, df23, "business_address", normalize_address)

    logger.info("Pass 3: first-2-token name blocking")
    c3 = _token_block(df1, df23, "business_name", normalize_name, n_tokens=2)

    logger.info("Pass 4: first-1-token name blocking")
    c4 = _token_block(df1, df23, "business_name", normalize_name, n_tokens=1)

    logger.info("Pass 5: TF-IDF name blocking")
    c5 = _tfidf_block(df1, df23, "business_name", normalize_name, threshold=tfidf_threshold)

    merged = _merge_dicts(c1, c2, c3, c4, c5)

    # Cap per entity
    for k in merged:
        if len(merged[k]) > max_per_entity:
            merged[k] = set(list(merged[k])[:max_per_entity])

    total = sum(len(v) for v in merged.values())
    logger.info(
        "Candidates: %d S1 entities, %.1f avg candidates, %d total pairs",
        len(merged), total / max(len(merged), 1), total,
    )
    return dict(merged)
