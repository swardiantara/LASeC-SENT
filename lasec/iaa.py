"""Inter-annotator agreement for event-grouping labels.

Event IDs are arbitrary names, so agreement is measured on the pairwise decision
"do these two sentences describe the same event?" over all unordered sentence pairs
(Cohen's kappa), complemented by ARI between the two partitions.
"""
import itertools

import numpy as np
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score


def pairwise_agreement(labels_a, labels_b) -> dict:
    assert len(labels_a) == len(labels_b)
    same_a, same_b = [], []
    for i, j in itertools.combinations(range(len(labels_a)), 2):
        same_a.append(labels_a[i] == labels_a[j])
        same_b.append(labels_b[i] == labels_b[j])
    same_a, same_b = np.array(same_a), np.array(same_b)
    return dict(n_items=len(labels_a), n_pairs=len(same_a),
                kappa=float(cohen_kappa_score(same_a, same_b)),
                pair_agreement=float((same_a == same_b).mean()),
                same_event_pairs_a=int(same_a.sum()), same_event_pairs_b=int(same_b.sum()),
                same_event_pairs_both=int((same_a & same_b).sum()),
                ari=float(adjusted_rand_score(labels_a, labels_b)),
                exact_partition_match=bool(adjusted_rand_score(labels_a, labels_b) == 1.0))
