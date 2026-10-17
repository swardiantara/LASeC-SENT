"""Grouping metrics and forensic triage metrics.

All grouping metrics are computed over UNIQUE sentences (each distinct string is one item).
'weighted' variants weight each unique sentence by its number of occurrences in the logs,
i.e. they answer "what fraction of the log lines the analyst sees are grouped correctly".
"""
import warnings
from collections import Counter, defaultdict
from math import comb
from typing import Dict, Hashable, List, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _groups(labels):
    g = defaultdict(set)
    for i, lab in enumerate(labels):
        g[lab].add(i)
    return g


def ga_correct(gold: Sequence[Hashable], pred: Sequence[Hashable]) -> np.ndarray:
    """Per-item Group Accuracy indicator (Zhu et al., 2019): an item is correct iff its predicted
    group contains exactly the items of its gold group."""
    gg, pg = _groups(gold), _groups(pred)
    return np.array([gg[g] == pg[p] for g, p in zip(gold, pred)], dtype=float)


def pairwise_prf(gold, pred):
    """Pairwise precision (penalises over-merging) and recall (penalises over-splitting)."""
    cont = Counter(zip(gold, pred))
    tp = sum(comb(n, 2) for n in cont.values())
    pp = sum(comb(n, 2) for n in Counter(pred).values())
    gp = sum(comb(n, 2) for n in Counter(gold).values())
    p = tp / pp if pp else 1.0
    r = tp / gp if gp else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def merge_errors(gold, pred, sentences, weights=None):
    """Over-merges: predicted clusters holding >1 gold event. Over-splits: gold events spread over >1 cluster."""
    w = weights if weights is not None else [1] * len(gold)
    by_pred, by_gold = defaultdict(list), defaultdict(set)
    for g, p, s, wi in zip(gold, pred, sentences, w):
        by_pred[p].append((g, s, wi))
        by_gold[g].add(p)
    over = []
    for p, members in by_pred.items():
        events = sorted({g for g, _, _ in members})
        if len(events) > 1:
            over.append(dict(cluster=p, events=events, sentences=[s for _, s, _ in members],
                             occurrences=int(sum(wi for _, _, wi in members))))
    split = [dict(event=g, n_clusters=len(ps)) for g, ps in by_gold.items() if len(ps) > 1]
    return over, split


def _quiet(fn, gold, pred):
    """sklearn warns that many unique labels look 'continuous'; harmless for clustering scores."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(fn([str(g) for g in gold], [str(p) for p in pred]))


def grouping_report(gold, pred, sentences, weights=None) -> dict:
    gold, pred = list(gold), list(pred)
    w = np.asarray(weights if weights is not None else np.ones(len(gold)), dtype=float)
    c = ga_correct(gold, pred)
    p, r, f = pairwise_prf(gold, pred)
    over, split = merge_errors(gold, pred, sentences, w)
    return dict(n_items=len(gold), n_gold_events=len(set(gold)), n_clusters=len(set(pred)),
                ga=float(c.mean()), ga_weighted=float((c * w).sum() / w.sum()),
                pair_precision=p, pair_recall=r, pair_f1=f,
                ari=_quiet(adjusted_rand_score, gold, pred), nmi=_quiet(normalized_mutual_info_score, gold, pred),
                n_overmerged_clusters=len(over), n_oversplit_events=len(split),
                overmerged_occurrences=int(sum(o["occurrences"] for o in over)),
                overmerges=over)


# ----------------------------------------------------------------------------- forensic triage
def representative(members: List[str], counts: Dict[str, int]) -> str:
    """The sentence shown to the analyst for a cluster: most frequent, ties broken by length then text."""
    return sorted(members, key=lambda s: (-counts.get(s, 0), -len(s), s))[0]


def triage_report(occ_sentences: Sequence[str], gold: Dict[str, str], pred: Dict[str, Hashable],
                  key_events: Sequence[str]) -> dict:
    """Triage view of ONE flight.

    occ_sentences: sentence occurrences of the flight, in log order.
    key_events: expert key-event sentences of the flight; they are compared at the level of
      their gold event (two key sentences with the same gold event count once).
    A key event is *recovered* if some predicted cluster of this flight has a representative
    whose gold event equals it. It is *hidden* if it occurs in the flight but every cluster
    containing it is represented by a sentence of a different gold event (over-merge).
    """
    counts = Counter(occ_sentences)
    clusters = defaultdict(list)
    for s in counts:
        clusters[pred[s]].append(s)
    reps = {c: representative(m, counts) for c, m in clusters.items()}
    rep_events = {gold[r] for r in reps.values()}
    key_gold = sorted({gold[k] for k in key_events})
    present = {gold[s] for s in counts}
    recovered = [e for e in key_gold if e in rep_events]
    hidden = [e for e in key_gold if e in present and e not in rep_events]
    n = len(occ_sentences)
    return dict(n_sentences=n, n_unique_sentences=len(counts), n_gold_events=len(present),
                n_triage_events=len(clusters),
                workload_reduction=1 - len(clusters) / n if n else 0.0,
                key_events=len(key_gold), key_recovered=len(recovered),
                key_recall=len(recovered) / len(key_gold) if key_gold else float("nan"),
                hidden_key_events=hidden,
                gold_events_hidden=sorted(present - rep_events))
