"""Step 8: repeatability of LASeC on FPV.

Re-runs the clustering of each LASeC model with N random input orders, for the main algorithm and each
ablation algorithm. Agglomerative clustering should be order-invariant; BIRCH builds its CF-tree
incrementally, so its clusters can change with the order. Reported results use alphabetical order.
LLM run-to-run variance is reported by evaluate_fpv.py (llm_summary.csv).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.cluster import ALGORITHM_LABEL, cluster_sentences, embed, load_encoder
from lasec.experiment import cluster_kwargs, load_fpv, slug
from lasec.metrics import grouping_report
from lasec.utils import load_config, path, read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--orders", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    fpv, _ = load_fpv(cfg)
    sents, gold, w = fpv.sentence.tolist(), fpv.event.tolist(), fpv.occurrences.to_numpy()
    rows, summary = [], {}
    for meta_f in sorted(glob.glob(str(path(cfg, "models") / "lasec-*" / "lasec_meta.json"))):
        meta = read_json(meta_f)
        enc = load_encoder(os.path.dirname(meta_f))
        X1 = embed(enc, sents, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
        X2 = embed(enc, sents, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
        main = meta.get("algorithm", cfg["clustering"]["algorithm"])
        thr = meta.get("thresholds_by_algorithm", {main: meta["threshold"]})
        for alg, t in thr.items():
            name = f"LASeC ({slug(meta['base_model'])})" + ("" if alg == main else f" + {ALGORITHM_LABEL[alg]}")
            for seed in range(a.orders):
                pred = cluster_sentences(sents, X1, t, **cluster_kwargs(cfg, alg, order="shuffle", seed=seed))
                r = grouping_report(gold, [pred[s] for s in sents], sents, w)
                rows.append(dict(method=name, algorithm=alg, order_seed=seed, ga=r["ga"], ga_weighted=r["ga_weighted"],
                                 pair_f1=r["pair_f1"], n_clusters=r["n_clusters"], n_overmerged=r["n_overmerged_clusters"]))
            d = pd.DataFrame([x for x in rows if x["method"] == name])
            summary[name] = dict(algorithm=alg, embedding_max_abs_diff_between_two_passes=float(np.abs(X1 - X2).max()),
                                 **{f"{c}_{s}": float(getattr(d[c], s)()) for c in ("ga", "ga_weighted", "n_clusters")
                                    for s in ("mean", "std", "min", "max")})
    out = path(cfg, "results") / "repeatability"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "order_runs.csv", index=False)
    write_json(summary, out / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
