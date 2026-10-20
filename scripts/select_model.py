"""Step 2: model selection on the catalogue with PRE-TRAINED encoders (no fine-tuning, no FPV data).

Stage A (clustering algorithm, cf. old Fig. 4): encode the catalogue with pre-trained SBERT
  (selection.embedding) and score every algorithm in selection.algorithms over a threshold grid.
Stage B (encoder, cf. old Fig. 5): with the configured clustering algorithm, score every pre-trained model in
  embedding.base_model_variants over the grid and measure encoding throughput (CPU and, if present, GPU).
The configured choices (clustering.algorithm, embedding.base_model) are the ones used downstream; this script
records the evidence and prints a NOTE when the evidence favours something else.
Outputs in results/selection/: algorithm_grid.csv, algorithm_summary.csv, embedding_grid.csv,
embedding_summary.csv, selection.json
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.cluster import ALGORITHMS, embed, load_encoder, threshold_grid
from lasec.experiment import load_catalogue, slug, threshold_curve
from lasec.utils import environment_info, load_config, path, set_seed, write_json


def summarise(curve: pd.DataFrame, key: str, value: str) -> dict:
    best = curve.loc[curve.ga.idxmax()]
    return {key: value, "best_threshold": float(best.threshold), "peak_ga": float(best.ga),
            "peak_pair_f1": float(best.pair_f1), "clusters_at_peak": int(best.n_clusters),
            "mean_ga_over_grid": float(curve.ga.mean())}


def throughput(encoder, sentences, cfg, repeats=3):
    """Sentences per second for encoding (warm-up excluded; best of `repeats`)."""
    embed(encoder, sentences[:8], cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        embed(encoder, sentences, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
        best = min(best, time.perf_counter() - t)
    return len(sentences) / best


def full_grid(cfg, s, e, grid, out):
    """Every clustering algorithm with every pre-trained encoder (checks the encoder x algorithm interaction)."""
    sel = cfg["selection"]
    curves, summ = [], []
    for m in cfg["embedding"]["base_model_variants"]:
        X = embed(load_encoder(m), s, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
        for alg in sel["algorithms"]:
            c = threshold_curve(X, s, e, cfg, algorithm=alg, grid=grid)
            c.insert(0, "algorithm", alg)
            c.insert(0, "embedding", slug(m))
            curves.append(c)
            row = summarise(c, "algorithm", alg)
            row["embedding"] = slug(m)
            summ.append(row)
            print(f"  {slug(m):24s} {alg:14s} peak GA {row['peak_ga']:.3f} at t={row['best_threshold']}")
    pd.concat(curves).to_csv(out / "full_grid.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "full_grid_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--full-grid", action="store_true",
                    help="only run every algorithm x every pre-trained encoder (interaction check) -> full_grid*.csv")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    set_seed(cfg["seed"])
    sel = cfg["selection"]
    cat = load_catalogue(cfg)
    s, e = cat.sentence.tolist(), cat.event.tolist()
    grid = threshold_grid(sel["threshold_grid"])
    out = path(cfg, "results") / "selection"
    out.mkdir(parents=True, exist_ok=True)
    if a.full_grid:
        return full_grid(cfg, s, e, grid, out)

    # ---------------- Stage A: clustering algorithms with pre-trained SBERT
    print(f"Stage A: {sel['embedding']} x {sel['algorithms']}")
    X = embed(load_encoder(sel["embedding"]), s, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
    curves, summ = [], []
    for alg in sel["algorithms"]:
        if alg not in ALGORITHMS:
            sys.exit(f"unknown algorithm {alg}")
        c = threshold_curve(X, s, e, cfg, algorithm=alg, grid=grid)
        c.insert(0, "algorithm", alg)
        curves.append(c)
        summ.append(summarise(c, "algorithm", alg))
        print(f"  {alg:14s} peak GA {summ[-1]['peak_ga']:.3f} at t={summ[-1]['best_threshold']}")
    pd.concat(curves).to_csv(out / "algorithm_grid.csv", index=False)
    alg_sum = pd.DataFrame(summ).sort_values("peak_ga", ascending=False)
    alg_sum.to_csv(out / "algorithm_summary.csv", index=False)
    best_alg = alg_sum.iloc[0].algorithm
    chosen_alg = cfg["clustering"]["algorithm"]

    # ---------------- Stage B: pre-trained encoders with the chosen algorithm
    print(f"Stage B: {chosen_alg} x {len(cfg['embedding']['base_model_variants'])} pre-trained encoders")
    curves, summ = [], []
    for m in cfg["embedding"]["base_model_variants"]:
        enc = load_encoder(m)
        Xm = embed(enc, s, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
        c = threshold_curve(Xm, s, e, cfg, algorithm=chosen_alg, grid=grid)
        c.insert(0, "embedding", slug(m))
        curves.append(c)
        row = summarise(c, "embedding", slug(m))
        row["params_m"] = round(sum(p.numel() for p in enc.parameters()) / 1e6, 1)
        row["dim"] = int(Xm.shape[1])
        row["sentences_per_s"] = throughput(enc, s, cfg)
        summ.append(row)
        print(f"  {slug(m):24s} peak GA {row['peak_ga']:.3f} at t={row['best_threshold']}  "
              f"{row['sentences_per_s']:.0f} sent/s  {row['params_m']}M params")
    pd.concat(curves).to_csv(out / "embedding_grid.csv", index=False)
    emb_sum = pd.DataFrame(summ)
    emb_sum.to_csv(out / "embedding_summary.csv", index=False)
    best_emb = emb_sum.loc[emb_sum.peak_ga.idxmax()]
    ok = emb_sum[emb_sum.peak_ga >= best_emb.peak_ga - sel["efficiency_tolerance"]]
    efficient = ok.loc[ok.sentences_per_s.idxmax()]
    chosen_emb = slug(cfg["embedding"]["base_model"])

    result = dict(stage_a=dict(embedding=sel["embedding"], best_algorithm=best_alg, configured_algorithm=chosen_alg,
                               consistent=bool(best_alg == chosen_alg)),
                  stage_b=dict(algorithm=chosen_alg, most_accurate=best_emb.embedding,
                               efficient_within_tolerance=efficient.embedding, tolerance=sel["efficiency_tolerance"],
                               configured_embedding=chosen_emb, consistent=bool(efficient.embedding == chosen_emb)),
                  environment=environment_info())
    write_json(result, out / "selection.json")
    if best_alg != chosen_alg:
        print(f"NOTE: Stage A favours {best_alg}, but clustering.algorithm = {chosen_alg}. Revisit before reporting.")
    if efficient.embedding != chosen_emb:
        print(f"NOTE: Stage B's efficient choice is {efficient.embedding} (most accurate: {best_emb.embedding}), "
              f"but embedding.base_model = {chosen_emb}. Revisit before reporting.")
    print(f"configured: {chosen_alg} + {chosen_emb}")


if __name__ == "__main__":
    main()
