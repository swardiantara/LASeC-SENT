"""Step 6: evaluate every method on the held-out FPV sentences.

Methods: exact dedup, regex-masked dedup, Drain3 (tuned on the catalogue), pre-trained embedding + the main
clustering algorithm (threshold tuned on the catalogue), LASeC (threshold from cross-validation), LASeC with each
ablation clustering algorithm (its own CV threshold), and every cached LLM run.
Each method groups ALL FPV sentences; scores are reported on two subsets:
  all     every unique sentence
  events  event sentences only (sentence_type event/compound); groups are restricted to these sentences
Outputs in results/fpv/:
  metrics.csv        one row per (method, subset) with CIs, seen/unseen GA, runtime, LLM cost
  llm_summary.csv    mean/sd over repeated LLM runs per (model, mode, subset)
  significance.csv   primary LASeC vs each baseline (paired bootstrap, exact McNemar, Holm) per subset
  sensitivity.csv    post-hoc FPV score vs clustering threshold (analysis only; thresholds are chosen on the catalogue)
  overmerges.json, predictions/*.csv
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.baselines import drain3_groups, exact_dedup, regex_masked_dedup
from lasec.cluster import ALGORITHM_LABEL, cluster_sentences, embed, load_encoder
from lasec.experiment import (EVENT_TYPES, algorithm_grid, cluster_kwargs, load_fpv, method_file, primary_method,
                              slug, subset_ga)
from lasec.metrics import ga_correct, grouping_report
from lasec.stats import bootstrap_ci, cluster_bootstrap_ci, holm, mcnemar_exact, paired_bootstrap
from lasec.utils import load_config, path, read_json, write_json


OPTIONAL_COLUMNS = ("threshold", "base_model", "algorithm", "cost_usd", "llm_config", "llm_run")


def timed(fn, *args, **kw):
    t = time.perf_counter()
    r = fn(*args, **kw)
    return r, time.perf_counter() - t


def embedding_method(model, sents, threshold, cfg, algorithm):
    """Encode + cluster; returns (prediction, seconds, X) - model loading is excluded from the timing."""
    enc = load_encoder(model)
    embed(enc, sents[:2], cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])  # warm-up
    t = time.perf_counter()
    X = embed(enc, sents, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
    pred = cluster_sentences(sents, X, threshold, **cluster_kwargs(cfg, algorithm))
    return pred, time.perf_counter() - t, X


def llm_run_cost(cache_dir, meta, run_idx):
    """Latency and cost of one LLM run from its cached responses."""
    rec = meta["runs"][run_idx] if run_idx < len(meta["runs"]) else {}
    lat, cost = 0.0, 0.0
    for b in rec.get("batches", []):
        f = cache_dir / b["file"]
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            lat += r.get("latency_s") or 0.0
            cost += (r.get("usage") or {}).get("cost") or 0.0
    return lat, cost


def collect_methods(cfg, sents):
    res = path(cfg, "results")
    m, info, sweeps = {}, {}, []
    for name, fn, fam in (("Exact dedup", exact_dedup, "string"), ("Regex-masked dedup", regex_masked_dedup, "string")):
        m[name], t = timed(fn, sents)
        info[name] = dict(family=fam, runtime_s=t)
    sel = read_json(res / "baselines" / "selection.json")
    d = sel["drain3"]
    m["Drain3"], t = timed(drain3_groups, sents, d["sim_th"], d["depth"])
    info["Drain3"] = dict(family="parser", runtime_s=t, threshold=None)
    main_alg = cfg["clustering"]["algorithm"]
    models = [(f"Pre-trained {slug(mdl)} + {ALGORITHM_LABEL[s.get('algorithm', main_alg)]}", mdl, s["threshold"],
               "pretrained", mdl, s.get("algorithm", main_alg)) for mdl, s in sel["pretrained"].items()]
    for meta_f in sorted(glob.glob(str(path(cfg, "models") / "lasec-*" / "lasec_meta.json"))):
        meta = read_json(meta_f)
        alg = meta.get("algorithm", main_alg)
        models.append((f"LASeC ({slug(meta['base_model'])})", os.path.dirname(meta_f), meta["threshold"], "lasec",
                       meta["base_model"], alg))
        for abl, thr in meta.get("thresholds_by_algorithm", {}).items():  # clustering-algorithm ablation
            if abl != alg:
                models.append((f"LASeC ({slug(meta['base_model'])}) + {ALGORITHM_LABEL[abl]}", os.path.dirname(meta_f),
                               thr, "lasec", meta["base_model"], abl))
    for name, mdl, thr, fam, base, alg in models:
        m[name], t, X = embedding_method(mdl, sents, thr, cfg, alg)
        info[name] = dict(family=fam, runtime_s=t, threshold=thr, base_model=slug(base), algorithm=alg)
        for tt in algorithm_grid(cfg, alg):
            sweeps.append((name, fam, tt, cluster_sentences(sents, X, tt, **cluster_kwargs(cfg, alg))))
    cache = res / "llm" / "cache"
    for run_f in sorted(glob.glob(str(res / "llm" / "*" / "run*.csv"))):
        cfg_name, run = os.path.basename(os.path.dirname(run_f)), os.path.basename(run_f)[:-4]
        df = pd.read_csv(run_f)
        meta_f = os.path.join(os.path.dirname(run_f), "meta.json")
        meta = read_json(meta_f) if os.path.exists(meta_f) else {"runs": []}
        lat, cost = llm_run_cost(cache, meta, int(run[3:]))
        name = f"LLM {cfg_name} {run}"
        m[name] = dict(zip(df.sentence, df.pred))
        mode = cfg_name.rsplit("_", 1)[-1]
        info[name] = dict(family=f"llm-{mode}", runtime_s=lat, cost_usd=cost, llm_config=cfg_name, llm_run=int(run[3:]))
    return m, info, sweeps


def score(name, pred, gold, sents, w, occ, fpv, mask, B):
    """Grouping report restricted to `mask` (groups are intersected with the subset)."""
    g = [x for x, k in zip(gold, mask) if k]
    p = [x for x, k in zip(pred, mask) if k]
    s = [x for x, k in zip(sents, mask) if k]
    ww = w[mask]
    r = grouping_report(g, p, s, ww)
    c = ga_correct(g, p)
    by_s = dict(zip(s, c))
    flights = [occ[(occ.flight == f) & occ.sentence.isin(by_s)].sentence.map(by_s).to_numpy()
               for f in sorted(occ.flight.unique())]
    lo, hi = bootstrap_ci(c, None, B)
    wlo, whi = bootstrap_ci(c, ww, B)
    flo, fhi = cluster_bootstrap_ci([f for f in flights if len(f)], B)
    row = dict(**{k: v for k, v in r.items() if k != "overmerges"}, ga_ci_low=lo, ga_ci_high=hi,
               ga_weighted_ci_low=wlo, ga_weighted_ci_high=whi, ga_weighted_flight_ci_low=flo,
               ga_weighted_flight_ci_high=fhi)
    if "seen_sentence" in fpv:
        seen = fpv.seen_sentence.to_numpy()[mask]
        row["ga_seen_sentence"] = subset_ga(g, p, seen)
        row["ga_unseen_sentence"] = subset_ga(g, p, ~seen)
        row["n_unseen_sentence"] = int((~seen).sum())
    return row, c, r["overmerges"]


def significance(primary, correct, info, subsets, w, B):
    """Primary LASeC vs each baseline (first LLM run only): paired bootstrap, exact McNemar, Holm within subset."""
    tests = []
    for sub in subsets:
        fam_tests = []
        for name in info:
            fam = info[name]["family"]
            if name == primary or fam == "lasec" or (fam.startswith("llm") and info[name]["llm_run"] != 0):
                continue
            a_, b_ = correct[(primary, sub)], correct[(name, sub)]
            ww = w[subsets[sub]]
            pb, pw, mc = paired_bootstrap(a_, b_, None, B), paired_bootstrap(a_, b_, ww, B), mcnemar_exact(a_, b_)
            fam_tests.append(dict(subset=sub, baseline=name, family=fam, diff_ga=pb["diff"], ci_low=pb["ci"][0],
                                  ci_high=pb["ci"][1], p_bootstrap=pb["p"], diff_ga_weighted=pw["diff"],
                                  p_bootstrap_weighted=pw["p"], lasec_only_correct=mc["a_only"],
                                  baseline_only_correct=mc["b_only"], p_mcnemar=mc["p"]))
        t = pd.DataFrame(fam_tests)
        for col in ("p_bootstrap", "p_bootstrap_weighted", "p_mcnemar"):  # Holm within each subset's family
            adj = holm(dict(zip(t.baseline, t[col])))
            t[col + "_holm"] = t.baseline.map(adj)
        tests.append(t)
    sig = pd.concat(tests, ignore_index=True)
    sig.insert(0, "lasec", primary)
    return sig


def stats_only(cfg, fpv, subsets, w, B):
    """Recompute significance.csv for the configured primary model from saved predictions (no models needed)."""
    out = path(cfg, "results") / "fpv"
    metrics = pd.read_csv(out / "metrics.csv")
    info = {r.method: dict(family=r.family, llm_run=r.llm_run) for r in metrics[metrics.subset == "all"].itertuples()}
    gold = fpv.event.tolist()
    correct = {}
    for name in info:
        p = pd.read_csv(out / "predictions" / (method_file(name) + ".csv")).set_index("sentence").loc[fpv.sentence]
        for sub, mask in subsets.items():
            correct[(name, sub)] = ga_correct([g for g, k in zip(gold, mask) if k], p.pred.to_numpy()[mask].tolist())
    primary = primary_method(cfg)
    if primary not in info:
        sys.exit(f"{primary} not found in {out / 'metrics.csv'}")
    sig = significance(primary, correct, info, subsets, w, B)
    sig.to_csv(out / "significance.csv", index=False)
    print(sig[["subset", "baseline", "diff_ga", "ci_low", "ci_high", "p_mcnemar_holm"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stats-only", action="store_true",
                    help="only recompute significance.csv for the configured primary model from saved predictions")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    B = cfg["stats"]["bootstrap"]
    fpv, occ = load_fpv(cfg)
    fpv = fpv.sort_values("sentence").reset_index(drop=True)
    sents, gold, w = fpv.sentence.tolist(), fpv.event.tolist(), fpv.occurrences.to_numpy()
    subsets = {"all": np.ones(len(fpv), bool)}
    if "sentence_type" in fpv:
        subsets["events"] = fpv.sentence_type.isin(EVENT_TYPES).to_numpy()
    if a.stats_only:
        return stats_only(cfg, fpv, subsets, w, B)
    methods, info, sweeps = collect_methods(cfg, sents)
    primary = primary_method(cfg) if primary_method(cfg) in methods else None
    out = path(cfg, "results") / "fpv"
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    rows, correct, overmerges = [], {}, {}
    for name, pred_map in methods.items():
        pred = [pred_map[s] for s in sents]
        for sub, mask in subsets.items():
            row, c, om = score(name, pred, gold, sents, w, occ, fpv, mask, B)
            rows.append(dict(method=name, subset=sub, **info[name], **row))
            correct[(name, sub)] = c
            overmerges[f"{name}|{sub}"] = om
        pd.DataFrame({"sentence": sents, "occurrences": w, "gold": gold, "pred": pred,
                      "sentence_type": fpv.get("sentence_type", pd.Series([""] * len(fpv))),
                      "correct": ga_correct(gold, pred)}).to_csv(out / "predictions" / (method_file(name) + ".csv"), index=False)
    metrics = pd.DataFrame(rows)
    for c in OPTIONAL_COLUMNS:  # stable schema, whether or not LLM runs / embedding models exist
        if c not in metrics:
            metrics[c] = np.nan
    metrics.to_csv(out / "metrics.csv", index=False)
    write_json(overmerges, out / "overmerges.json")

    llm = metrics[metrics.family.str.startswith("llm")].copy()
    if len(llm):
        agg = llm.groupby(["llm_config", "subset"]).agg(
            runs=("ga", "size"), ga_mean=("ga", "mean"), ga_sd=("ga", "std"), ga_weighted_mean=("ga_weighted", "mean"),
            ga_weighted_sd=("ga_weighted", "std"), pair_f1_mean=("pair_f1", "mean"), n_clusters_mean=("n_clusters", "mean"),
            overmerged_mean=("n_overmerged_clusters", "mean"), latency_s_mean=("runtime_s", "mean"),
            cost_usd_mean=("cost_usd", "mean")).reset_index()
        agg.to_csv(out / "llm_summary.csv", index=False)

    if primary:
        significance(primary, correct, info, subsets, w, B).to_csv(out / "significance.csv", index=False)

    srows = []
    for name, fam, t, pred_map in sweeps:
        pred = [pred_map[s] for s in sents]
        for sub, mask in subsets.items():
            g = [x for x, k in zip(gold, mask) if k]
            p = [x for x, k in zip(pred, mask) if k]
            r = grouping_report(g, p, [x for x, k in zip(sents, mask) if k], w[mask])
            srows.append(dict(method=name, family=fam, subset=sub, threshold=t, ga=r["ga"], ga_weighted=r["ga_weighted"],
                              pair_f1=r["pair_f1"], n_clusters=r["n_clusters"]))
    pd.DataFrame(srows).to_csv(out / "sensitivity.csv", index=False)

    view = metrics[metrics.subset == "all"][["method", "n_clusters", "ga", "ga_ci_low", "ga_ci_high", "ga_weighted",
                                             "pair_precision", "pair_recall", "n_overmerged_clusters", "runtime_s"]]
    print(view.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
