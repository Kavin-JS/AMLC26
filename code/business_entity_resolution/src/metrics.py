"""
Evaluation: entity-level macro F0.5.
"""

from __future__ import annotations

from typing import Dict, Set
import numpy as np


def f05(precision: float, recall: float) -> float:
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return 1.25 * precision * recall / denom


def entity_f05(predicted: Set[str], actual: Set[str]) -> float:
    """F0.5 for a single S1 entity."""
    if not actual and not predicted:
        return 1.0
    if not actual:
        return 0.0  # predicted something but nothing is true → precision 0
    if not predicted:
        # recall 0, precision undefined → treat as 0
        # F0.5 = 1.25 * p * r / (0.25*p + r) with r=0 → 0
        return 0.0

    tp = len(predicted & actual)
    prec = tp / len(predicted)
    rec = tp / len(actual)
    return f05(prec, rec)


def macro_f05(
    pred_map: Dict[str, Set[str]],
    gt_map: Dict[str, Set[str]],
) -> float:
    """
    Macro-average F0.5 over all S1 entities in gt_map.
    Entities missing from pred_map are treated as empty predictions.
    """
    scores = []
    for s1_id, actual in gt_map.items():
        predicted = pred_map.get(s1_id, set())
        scores.append(entity_f05(predicted, actual))
    return float(np.mean(scores)) if scores else 0.0


def tune_threshold(
    pairs: list,
    scores: np.ndarray,
    gt_map: Dict[str, Set[str]],
    thresholds=None,
) -> float:
    """
    Find the probability threshold that maximizes macro-F0.5 on val split.

    Args:
        pairs:      list of (s1_id, s23_id)
        scores:     model probability scores (same length as pairs)
        gt_map:     ground truth {s1_id: set of matched ids}
        thresholds: iterable of thresholds to try; defaults to np.linspace(0.1, 0.9, 17)

    Returns best threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        pred_map: Dict[str, Set[str]] = {}
        for (s1_id, s23_id), sc in zip(pairs, scores):
            if sc >= t:
                pred_map.setdefault(s1_id, set()).add(s23_id)
        mf = macro_f05(pred_map, gt_map)
        if mf > best_f:
            best_f, best_t = mf, t

    return best_t
