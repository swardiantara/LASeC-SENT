"""Step 8b: robustness analyses requested in review (all from saved results unless --recluster is given).

  group_ci.csv              GA and weighted GA with CIs from a bootstrap over TRUE EVENT GROUPS (not sentences)
  group_significance.csv    primary LASeC vs each baseline: group-bootstrap CI of the difference and a
                            group-level sign-flip permutation test, Holm-adjusted within each subset (GA and weighted GA)
  per_flight_pooled.csv     GA / pairwise P, R of the pooled (case-level) grouping restricted to each flight
  cv_nested.csv             CV GA with the threshold chosen on the OTHER folds (no selection bias)
  cv_singletons.csv         share of singleton groups among seen- and novel-event sentences of each CV fold
  cv_by_group_size.csv      CV GA by seen/novel x singleton/multi-sentence group (needs fold_predictions.csv)
  llm_variance.csv          GA and run-to-run agreement (ARI between runs) of the LLMs with shuffled vs fixed
                            sentence order and default vs zero temperature (needs run_llm.py --fixed-order runs)
  per_flight_recluster.csv  (--recluster, needs the trained models) each flight clustered on its own
All files go to results/robustness/.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.data import load_key_events
from lasec.experiment import EVENT_TYPES, cluster_kwargs, load_catalogue, load_fpv, method_file, primary_method, slug
from lasec.metrics import ga_correct, pairwise_prf, triage_report
from lasec.report import reported_bases
from lasec.stats import group_bootstrap_ci, group_paired_test, holm
from lasec.utils import load_config, path

B = 10000


def load_pred(res, method, sentences):
    p = pd.read_csv(res / "fpv" / "predictions" / f"{method_file(method)}.csv").set_index("sentence")
    return p.loc[sentences, "pred"].tolist()


def group_analyses(cfg, res, out):
    fpv, _ = load_fpv(cfg)
    fpv = fpv.sort_values("sentence").reset_index(drop=True)
    sents, gold, w = fpv.sentence.tolist(), np.array(fpv.event), fpv.occurrences.to_numpy(float)
    subsets = {"all": np.ones(len(fpv), bool), "events": fpv.sentence_type.isin(EVENT_TYPES).to_numpy()}
    metrics = pd.read_csv(res / "fpv" / "metrics.csv")
    info = metrics[metrics.subset == "all"].set_index("method")
    correct = {}
    for m in info.index:
        pred = np.array(load_pred(res, m, sents), dtype=object)
        for sub, mask in subsets.items():
            correct[(m, sub)] = ga_correct(gold[mask].tolist(), pred[mask].tolist())
    rows = []
    for m in info.index:
        for sub, mask in subsets.items():
            c = correct[(m, sub)]
            lo, hi = group_bootstrap_ci(c, gold[mask], B=B)
            wlo, whi = group_bootstrap_ci(c, gold[mask], w[mask], B=B)
            rows.append(dict(method=m, family=info.loc[m, "family"], subset=sub, ga=c.mean(), ga_ci_low=lo, ga_ci_high=hi,
                             ga_weighted=(c * w[mask]).sum() / w[mask].sum(), ga_weighted_ci_low=wlo,
                             ga_weighted_ci_high=whi))
    pd.DataFrame(rows).to_csv(out / "group_ci.csv", index=False)

    primary = primary_method(cfg)
    lite = [f"LASeC ({b})" for b in reported_bases(cfg)[1:]]
    tests = []
    for sub, mask in subsets.items():
        fam = []
        for m in info.index:
            f = info.loc[m, "family"]
            run = info.loc[m, "llm_run"]
            if m == primary or (f == "lasec" and m not in lite) or (str(f).startswith("llm") and run != 0):
                continue
            a, b = correct[(primary, sub)], correct[(m, sub)]
            r = group_paired_test(a, b, gold[mask], B=B)
            rw = group_paired_test(a, b, gold[mask], w[mask], B=B)
            fam.append(dict(subset=sub, baseline=m, family=f, in_family=m not in lite, diff_ga=r["diff"],
                            ci_low=r["ci"][0], ci_high=r["ci"][1], p_perm=r["p"], diff_ga_weighted=rw["diff"],
                            ci_weighted_low=rw["ci"][0], ci_weighted_high=rw["ci"][1], p_perm_weighted=rw["p"],
                            groups_differing=r["groups_differing"]))
        t = pd.DataFrame(fam)
        inf = t[t.in_family]
        for col in ("p_perm", "p_perm_weighted"):
            adj = holm(dict(zip(inf.baseline, inf[col])))
            t[col + "_holm"] = t.baseline.map(adj)  # LASeC-lite is reported outside the Holm family
        tests.append(t)
    sig = pd.concat(tests, ignore_index=True)
    sig.insert(0, "lasec", primary)
    sig.to_csv(out / "group_significance.csv", index=False)
    print(sig[sig.subset == "all"][["baseline", "diff_ga", "ci_low", "ci_high", "p_perm", "p_perm_holm",
                                    "diff_ga_weighted", "p_perm_weighted_holm"]].round(3).to_string(index=False))


def per_flight_pooled(cfg, res, out):
    fpv, occ = load_fpv(cfg)
    gold = dict(zip(fpv.sentence, fpv.event))
    metrics = pd.read_csv(res / "fpv" / "metrics.csv")
    rows = []
    for m in metrics[metrics.subset == "all"].method:
        p = pd.read_csv(res / "fpv" / "predictions" / f"{method_file(m)}.csv").set_index("sentence").pred.to_dict()
        for flight, sub in occ.groupby("flight"):
            s = sorted(set(sub.sentence))
            g, pr = [gold[x] for x in s], [p[x] for x in s]
            c = ga_correct(g, pr)
            pp, rr, _ = pairwise_prf(g, pr)
            rows.append(dict(method=m, flight=flight, n_unique=len(s), n_correct=int(c.sum()), ga=c.mean(),
                             pair_precision=pp, pair_recall=rr, n_events=len(set(pr))))
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_flight_pooled.csv", index=False)
    s = df.groupby("method").agg(ga_mean=("ga", "mean"), ga_min=("ga", "min"), ga_max=("ga", "max"),
                                 n_correct=("n_correct", "sum"), n=("n_unique", "sum"))
    s["ga_micro"] = s.n_correct / s.n
    s.to_csv(out / "per_flight_pooled_summary.csv")


def cv_analyses(cfg, res, out):
    rows = []
    for d in sorted((res / "cv").iterdir()):
        f = d / "fold_curves.csv"
        if not f.exists():
            continue
        fc = pd.read_csv(f)
        fc = fc[fc.algorithm == cfg["clustering"]["algorithm"]]
        for enc in ("lasec", "pretrained"):
            x = fc[fc.encoder == enc]
            per = []
            for k in sorted(x.fold.unique()):
                other = x[x.fold != k].groupby("threshold").ga.mean()
                t = float(other[other == other.max()].index.min())  # ties -> smaller threshold, as in selection
                per.append((k, t, float(x[(x.fold == k) & (x.threshold == t)].ga.iloc[0])))
            rows.append(dict(base_model=d.name, encoder=enc, ga_nested_mean=np.mean([p[2] for p in per]),
                             ga_nested_sd=np.std([p[2] for p in per], ddof=1),
                             thresholds=";".join(f"{p[1]:.2f}" for p in per)))
    pd.DataFrame(rows).to_csv(out / "cv_nested.csv", index=False)

    cat = load_catalogue(cfg)
    s, e = cat.sentence.to_numpy(), cat.event.to_numpy()
    rows = []
    for k, (tr, te) in enumerate(KFold(cfg["cv"]["n_folds"], shuffle=True, random_state=cfg["seed"]).split(s)):
        novel = ~np.isin(e[te], e[tr])
        size = pd.Series(e[te]).map(pd.Series(e[te]).value_counts()).to_numpy()
        rows.append(dict(fold=k, n_test=len(te), n_novel=int(novel.sum()), n_seen=int((~novel).sum()),
                         singleton_share_novel=float((size[novel] == 1).mean()),
                         singleton_share_seen=float((size[~novel] == 1).mean()),
                         singleton_share_fold=float((size == 1).mean())))
    pd.DataFrame(rows).to_csv(out / "cv_singletons.csv", index=False)
    full = pd.Series(e).map(pd.Series(e).value_counts())
    print("singleton share: full catalogue %.3f | fold %.3f | novel %.3f | seen %.3f" % (
        (full == 1).mean(), np.mean([r["singleton_share_fold"] for r in rows]),
        np.mean([r["singleton_share_novel"] for r in rows]), np.mean([r["singleton_share_seen"] for r in rows])))

    strat = []
    for d in sorted((res / "cv").iterdir()):
        f = d / "fold_predictions.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        p["group"] = np.where(p.group_size_in_fold == 1, "singleton", "multi-sentence")
        p["event_status"] = np.where(p.novel_event, "novel", "seen")
        g = p.groupby(["event_status", "group"]).agg(n=("correct", "size"), ga=("correct", "mean")).reset_index()
        g.insert(0, "base_model", d.name)
        strat.append(g)
    if strat:
        pd.concat(strat).to_csv(out / "cv_by_group_size.csv", index=False)
    else:
        print("cv_by_group_size: SKIPPED (re-run cross_validate.py to write fold_predictions.csv)")


def llm_variance(cfg, res, out):
    """Separate order sensitivity from sampling: compare runs with shuffled order (main runs), fixed order at the
    provider default temperature, and fixed order at temperature 0."""
    from itertools import combinations

    from sklearn.metrics import adjusted_rand_score
    fpv, _ = load_fpv(cfg)
    gold = dict(zip(fpv.sentence, fpv.event))
    var = res / "llm_variance"
    if not var.exists():
        return print("llm_variance: SKIPPED (run: python scripts/run_llm.py --mode zero --fixed-order, and again with --temperature 0)")
    rows = []
    for m in cfg["llm"]["models"]:
        if not m.get("enabled", True):
            continue
        for label, d in (("shuffled order, default temperature", res / "llm" / f"{m['name']}_zero"),
                         ("fixed order, default temperature", var / f"{m['name']}_zero_fixed"),
                         ("fixed order, temperature 0", var / f"{m['name']}_zero_fixed_t0")):
            runs = sorted(d.glob("run*.csv")) if d.exists() else []
            if not runs:
                continue
            preds = [pd.read_csv(f).set_index("sentence").pred for f in runs]
            s = sorted(preds[0].index)
            ga = [ga_correct([gold[x] for x in s], [p[x] for x in s]).mean() for p in preds]
            ari = [adjusted_rand_score([str(a[x]) for x in s], [str(b[x]) for x in s]) for a, b in combinations(preds, 2)]
            rows.append(dict(model=m["name"], condition=label, runs=len(runs), ga_mean=np.mean(ga),
                             ga_sd=np.std(ga, ddof=1) if len(ga) > 1 else np.nan, ga_min=min(ga), ga_max=max(ga),
                             ari_between_runs_min=min(ari) if ari else np.nan,
                             identical_partitions=bool(ari) and min(ari) == 1.0))
    df = pd.DataFrame(rows)
    df.to_csv(out / "llm_variance.csv", index=False)
    print(df.round(3).to_string(index=False))


def recluster(cfg, res, out):
    """Cluster each flight's unique sentences on its own (single-flight deployment) and compare with pooling."""
    from lasec.cluster import cluster_sentences, embed, load_encoder
    fpv, occ = load_fpv(cfg)
    gold = dict(zip(fpv.sentence, fpv.event))
    keys = load_key_events(path(cfg, "fpv_key_events"))
    base_sel = __import__("json").load(open(res / "baselines" / "selection.json", encoding="utf-8"))["pretrained"]
    models = []
    for b in reported_bases(cfg):
        meta_f = path(cfg, "models") / f"lasec-{b}" / "lasec_meta.json"
        if not meta_f.exists():
            sys.exit(f"{meta_f} not found: run train_final.py --base-model {b} first")
        meta = __import__("json").load(open(meta_f, encoding="utf-8"))
        models.append((f"LASeC ({b})", str(meta_f.parent), meta["threshold"]))
    pb = cfg["embedding"]["base_model"]
    models.append((f"Pre-trained {slug(pb)} + Agglomerative", pb, base_sel[pb]["threshold"]))
    sents = sorted(fpv.sentence)
    rows = []
    for name, model, t in models:
        X = dict(zip(sents, embed(load_encoder(model), sents, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])))
        pooled = pd.read_csv(res / "fpv" / "predictions" / f"{method_file(name)}.csv").set_index("sentence").pred.to_dict()
        for flight, sub in occ.groupby("flight"):
            s = sorted(set(sub.sentence))
            pred = cluster_sentences(s, np.stack([X[x] for x in s]), t, **cluster_kwargs(cfg))
            for mode, pr in (("per-flight", pred), ("pooled", {x: pooled[x] for x in s})):
                g, p = [gold[x] for x in s], [pr[x] for x in s]
                c = ga_correct(g, p)
                pp, rr, _ = pairwise_prf(g, p)
                tri = triage_report(sub.sentence.tolist(), gold, pr, keys[flight])
                rows.append(dict(method=name, mode=mode, flight=flight, n_unique=len(s), ga=c.mean(), n_correct=int(c.sum()),
                                 pair_precision=pp, pair_recall=rr, n_events=tri["n_triage_events"],
                                 key_recovered=tri["key_recovered"], key_events=tri["key_events"],
                                 hidden=";".join(tri["hidden_key_events"])))
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_flight_recluster.csv", index=False)
    print(df.groupby(["method", "mode"]).agg(ga=("ga", "mean"), events=("n_events", "sum"),
                                             key=("key_recovered", "sum")).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--recluster", action="store_true", help="cluster each flight on its own (needs trained models)")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    res = path(cfg, "results")
    out = res / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    group_analyses(cfg, res, out)
    per_flight_pooled(cfg, res, out)
    cv_analyses(cfg, res, out)
    llm_variance(cfg, res, out)
    if a.recluster:
        recluster(cfg, res, out)


if __name__ == "__main__":
    main()
