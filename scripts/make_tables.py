"""Step 10: paper tables (LaTeX + CSV) and a plain-text summary of the key numbers.

Outputs in results/paper/tables/ and results/paper/summary.md. Missing inputs (e.g. no LLM runs yet, no second
annotator yet) are skipped with a note instead of failing.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.data import load_key_events
from lasec.cluster import ALGORITHM_LABEL
from lasec.experiment import EVENT_TYPES, slug
from lasec.report import BASE_ORDER, BASE_PARAMS_M, FAMILY_LABEL, SUBSET_LABEL, llm_display, load_all, method_table
from lasec.utils import load_config, path


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci(lo, hi):
    return "" if pd.isna(lo) else f"[{lo:.2f}, {hi:.2f}]"


def pval(p):
    return "--" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def latex(df: pd.DataFrame, caption: str, label: str, align: str | None = None, note: str = "") -> str:
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    align = align or "l" + "r" * (df.shape[1] - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\small", r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\",
             r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in df.astype(str).values]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(r"\par\smallskip{\footnotesize " + note + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def markdown(df: pd.DataFrame) -> str:
    cells = [[str(c) for c in df.columns]] + [[str(v).replace("|", "/") for v in row] for row in df.values]
    return "\n".join(["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])] +
                     ["| " + " | ".join(r) + " |" for r in cells[1:]])


class Writer:
    def __init__(self, out):
        self.out, self.md = out, []
        out.mkdir(parents=True, exist_ok=True)

    def table(self, name, df, caption, note="", align=None):
        df.to_csv(self.out / f"{name}.csv", index=False)
        (self.out / f"{name}.tex").write_text(latex(df, caption, f"tab:{name}", align, note), encoding="utf-8")
        self.md += [f"### {caption}", "", markdown(df), "", note, ""]
        print(f"  table {name}: {len(df)} rows")

    def text(self, s):
        self.md.append(s)

    def skip(self, name, why):
        print(f"  table {name}: SKIPPED ({why})")
        self.md += [f"### {name}: not available ({why})", ""]


def dataset_tables(R, W, cfg):
    vol, cat, fpv, occ = R["volume"], R["cat_labels"], R["fpv_labels"], R["fpv_occ"]
    if vol is None or cat is None:
        return W.skip("dataset", "run prepare_data.py and import_annotations.py")
    ev = cat.sentence_type.isin(EVENT_TYPES)
    sizes = cat[ev].groupby("event").size()
    fev = fpv.sentence_type.isin(EVENT_TYPES)
    keys = load_key_events(path(cfg, "fpv_key_events"))
    rows = [
        ("Source", "DroPTC catalogue (training)", "DJI FPV flights (held-out test)"),
        ("Messages / log cells", vol["catalogue"]["messages"], vol["fpv"]["cells"]),
        ("Atomic app messages", "--", vol["fpv"]["atomic_messages"]),
        ("Sentence occurrences", vol["catalogue"]["sentences"], vol["fpv"]["sentences"]),
        ("Unique sentences", len(cat), len(fpv)),
        ("  of which event sentences", int(ev.sum()), int(fev.sum())),
        ("Event groups (gold)", cat[ev].event.nunique(), fpv[fev].event_annotator1.nunique()),
        ("  singleton event groups", int((sizes == 1).sum()), int((fpv[fev].groupby("event_annotator1").size() == 1).sum())),
        ("Non-event groups (instruction/mitigation/possible/info)", cat[~ev].event.nunique(),
         fpv[~fev].event_annotator1.nunique()),
        ("Expert key events", "--", sum(len({fpv.set_index('sentence').event_annotator1[k] for k in v}) for v in keys.values())),
        ("Flights", "--", len(keys)),
    ]
    W.table("dataset", pd.DataFrame(rows, columns=["", "Catalogue", "FPV"]),
            "Datasets. The catalogue uses the manual DroPTC sentence split; FPV logs are segmented automatically "
            "(';' then PySBD).", align="lrr")
    types = pd.DataFrame({"Catalogue": cat.sentence_type.value_counts(), "FPV": fpv.sentence_type.value_counts()}).fillna(0).astype(int)
    types = types.reindex([t for t in ["event", "compound", "instruction", "mitigation", "possible", "info"] if t in types.index])
    W.table("sentence_types", types.reset_index().rename(columns={"index": "Sentence type", "sentence_type": "Sentence type"}),
            "Sentence types (unique sentences) under the annotation guideline.", align="lrr")
    if occ is not None:
        pf = occ.groupby("flight").agg(cells=("cell_id", "nunique"), sentences=("sentence", "size"),
                                       unique=("sentence", "nunique")).reset_index()
        g = dict(zip(fpv.sentence, fpv.event_annotator1))
        pf["events"] = [occ[occ.flight == f].sentence.map(g).nunique() for f in pf.flight]
        pf["key events"] = [len({g[k] for k in keys[f]}) for f in pf.flight]
        W.table("fpv_flights", pf.rename(columns={"flight": "Flight", "cells": "Log cells", "sentences": "Sentences",
                                                  "unique": "Unique", "events": "Gold groups"}),
                "FPV flights used as the held-out test set.")
    seg = vol["catalogue"].get("pysbd_vs_manual_segmentation")
    if seg:
        W.text(f"- PySBD (after ';' split) reproduces the manual DroPTC sentence boundaries for "
               f"{seg['pysbd_matches_manual']}/{seg['messages']} catalogue messages ({seg['share']:.1%}); "
               f"{seg['skipped_incomplete']} messages skipped because the manual split omits text.\n")


def annotation_tables(R, W):
    s, iaa = R["ann_stats"], R["iaa"]
    if s:
        W.text(f"- Catalogue annotation: {s['sentences']} sentences, {s['labels']} groups "
               f"({s['event_labels']} event groups over {s['event_sentences']} event sentences); "
               f"expert correction rate of the proposed labels {s['correction_rate']:.1%}.\n")
    rows = []
    for name, v in (iaa or {}).items():
        if v:
            rows.append((name, v["n_items"], v["n_pairs"], f3(v["kappa"]), f3(v["ari"]), f3(v["pair_agreement"])))
    if rows:
        W.table("iaa", pd.DataFrame(rows, columns=["Set", "Sentences", "Pairs", "Cohen's kappa", "ARI", "Pair agreement"]),
                "Inter-annotator agreement (pairwise same-event decisions), before adjudication.")
    else:
        W.skip("iaa", "second-annotator labels not imported yet")


def selection_tables(R, W, cfg):
    a, e, sel = R["alg_summary"], R["emb_summary"], R["model_selection"]
    if a is None:
        return W.skip("selection", "run select_model.py")
    W.table("selection_algorithms", pd.DataFrame({
        "Algorithm": a.algorithm.map(ALGORITHM_LABEL), "Peak GA": a.peak_ga.map(f3), "Best threshold": a.best_threshold.map(f3),
        "Mean GA over grid": a.mean_ga_over_grid.map(f3), "Pair F1 at peak": a.peak_pair_f1.map(f3),
        "Groups at peak": a.clusters_at_peak}),
        f"Stage A of model selection: clustering algorithms on the catalogue with the pre-trained "
        f"{slug(cfg['selection']['embedding'])} encoder (no fine-tuning). Thresholds: cosine distance "
        "(BIRCH: Euclidean sub-cluster radius on unit vectors).")
    if e is not None:
        e = e.copy()
        e["_o"] = e.embedding.map({b: i for i, b in enumerate(BASE_ORDER)})
        e = e.sort_values("_o")
        W.table("selection_embeddings", pd.DataFrame({
            "Encoder (pre-trained)": e.embedding, "Params (M)": e.params_m, "Dim": e.dim, "Peak GA": e.peak_ga.map(f3),
            "Best threshold": e.best_threshold.map(f3), "Mean GA over grid": e.mean_ga_over_grid.map(f3),
            "Encoding (sentences/s)": e.sentences_per_s.map(lambda v: f"{v:,.0f}")}),
            f"Stage B of model selection: pre-trained encoders with {ALGORITHM_LABEL[cfg['clustering']['algorithm']]} "
            "clustering on the catalogue; throughput measured on the catalogue sentences.")
    if sel:
        W.text(f"- Model selection: Stage A best algorithm = {sel['stage_a']['best_algorithm']} (configured "
               f"{sel['stage_a']['configured_algorithm']}); Stage B most accurate encoder = {sel['stage_b']['most_accurate']}, "
               f"fastest within {sel['stage_b']['tolerance']} GA = {sel['stage_b']['efficient_within_tolerance']} "
               f"(configured {sel['stage_b']['configured_embedding']}).\n")


def cv_table(R, W, cfg):
    if not R["cv"]:
        return W.skip("cv", "run cross_validate.py")
    rows = []
    for b in [b for b in BASE_ORDER if b in R["cv"]] + [b for b in R["cv"] if b not in BASE_ORDER]:
        s = R["cv"][b]
        L, P = s["lasec"], s["pretrained"]
        abl = {k: v for k, v in s.get("lasec_by_algorithm", {}).items() if k != s.get("algorithm")}
        abl_txt = "; ".join(f"{ALGORITHM_LABEL[k]} {v['ga_mean']:.3f} (t={v['selected_threshold']:.2f})" for k, v in abl.items())
        rows.append((b, f"{BASE_PARAMS_M.get(b, float('nan')):.1f}", f3(P["selected_threshold"]), f"{P['ga_mean']:.3f} ± {P['ga_sd']:.3f}",
                     f3(L["selected_threshold"]), f"{L['ga_mean']:.3f} ± {L['ga_sd']:.3f}", f3(L["pair_f1_mean"]),
                     f3(L["ga_seen_event"]), f3(L["ga_novel_event"]), abl_txt or "--"))
    W.table("cv", pd.DataFrame(rows, columns=["Base model", "Params (M)", "t (pre-trained)", "GA pre-trained",
                                              "t (LASeC)", "GA LASeC", "Pair F1", "GA seen event", "GA novel event",
                                              "LASeC with ablation clustering"]),
            f"{R['cv'][next(iter(R['cv']))]['n_folds']}-fold cross-validation on the catalogue (mean ± sd over folds). "
            f"Main clustering: {ALGORITHM_LABEL[cfg['clustering']['algorithm']]}; t is the threshold selected by CV; 'seen/novel event' splits held-out sentences by whether their event "
            "occurs in the training folds.")


def main_tables(R, W, cfg):
    m = R["metrics"]
    if m is None:
        return W.skip("fpv_main", "run evaluate_fpv.py")
    for sub in [s for s in ("all", "events") if s in set(m.subset)]:
        t = method_table(m, sub, cfg)
        df = pd.DataFrame({
            "Method": t.display, "Family": t.family.map(FAMILY_LABEL),
            "Groups": t.n_clusters.round(1), "GA": [f"{g:.3f}" + (f" ± {s:.3f}" if not pd.isna(s) else "") for g, s in zip(t.ga, t.ga_sd)],
            "95% CI": [ci(a, b) for a, b in zip(t.ga_ci_low, t.ga_ci_high)],
            "GA (occ.-weighted)": t.ga_weighted.map(f3), "Pair P": t.pair_precision.map(f3), "Pair R": t.pair_recall.map(f3),
            "Over-merged groups": t.n_overmerged_clusters.round(1),
            "GA unseen": t.get("ga_unseen_sentence", pd.Series([np.nan] * len(t))).map(f3),
            "Time (s)": t.runtime_s.map(lambda x: f"{x:.2f}")})
        W.table(f"fpv_main_{sub}", df,
                f"Held-out FPV results ({SUBSET_LABEL[sub]}). GA = group accuracy over unique sentences; occurrence-weighted "
                "GA weights each sentence by its frequency in the logs. LLM rows: mean ± sd over repeated runs.",
                note="95% CI: bootstrap over unique sentences. Time: grouping time for the FPV sentences "
                     "(LLM: API latency). Pair P/R: pairwise precision (penalises over-merging) and recall.")
    llm = R["llm_summary"]
    if llm is not None and len(llm):
        a = llm[llm.subset == "all"]
        W.table("llm_cost", pd.DataFrame({
            "Model (mode)": a.llm_config.map(lambda c: llm_display(c, cfg)), "Runs": a.runs,
            "GA": [f"{x:.3f} ± {y:.3f}" if not pd.isna(y) else f"{x:.3f}" for x, y in zip(a.ga_mean, a.ga_sd)],
            "Groups": a.n_clusters_mean.round(1), "Latency (s)": a.latency_s_mean.round(1),
            "Cost (USD/run)": a.cost_usd_mean.map(lambda x: f"{x:.4f}")}),
            "LLM baselines via OpenRouter: run-to-run variation, latency and cost per run (all sentences).")


def significance_table(R, W):
    s = R["significance"]
    if s is None:
        return W.skip("significance", "needs a trained primary LASeC model")
    for sub in sorted(set(s.subset)):
        x = s[s.subset == sub]
        W.table(f"significance_{sub}", pd.DataFrame({
            "Baseline": x.baseline, "ΔGA (LASeC − baseline)": x.diff_ga.map(lambda v: f"{v:+.3f}"),
            "95% CI": [ci(a, b) for a, b in zip(x.ci_low, x.ci_high)],
            "LASeC-only / baseline-only correct": [f"{a} / {b}" for a, b in zip(x.lasec_only_correct, x.baseline_only_correct)],
            "p (McNemar, Holm)": x.p_mcnemar_holm.map(pval), "p (bootstrap, Holm)": x.p_bootstrap_holm.map(pval),
            "p (weighted bootstrap, Holm)": x.p_bootstrap_weighted_holm.map(pval)}),
            f"Paired comparison of {x.lasec.iloc[0]} with each baseline on FPV ({SUBSET_LABEL[sub]}). LLM baselines: "
            "pre-specified first run (run 0). Holm correction within the family of comparisons.")


def ablation_table(R, W, cfg):
    m = R["metrics"]
    if m is None:
        return
    rows = []
    for sub in ("all", "events"):
        main = cfg["clustering"]["algorithm"]
        x = m[(m.subset == sub) & m.family.isin(["pretrained", "lasec"]) & (m.algorithm.isna() | (m.algorithm == main))]
        for b in [b for b in BASE_ORDER if b in set(x.base_model)]:
            p, l = x[(x.base_model == b) & (x.family == "pretrained")], x[(x.base_model == b) & (x.family == "lasec")]
            if len(l) == 0:
                continue
            gp = p.ga.iloc[0] if len(p) else np.nan
            rows.append((SUBSET_LABEL[sub], b, f"{BASE_PARAMS_M.get(b, np.nan):.1f}", f3(gp), f3(l.ga.iloc[0]),
                         f"{l.ga.iloc[0] - gp:+.3f}" if len(p) else "--", f3(l.ga_weighted.iloc[0]),
                         int(l.n_overmerged_clusters.iloc[0]), f"{l.runtime_s.iloc[0]:.2f}"))
    if rows:
        W.table("base_models", pd.DataFrame(rows, columns=["Subset", "Base model", "Params (M)", "GA pre-trained",
                                                           "GA LASeC", "Δ fine-tuning", "GA weighted", "Over-merged",
                                                           "Time (s)"]),
                "Effect of the base sentence encoder on FPV (each pre-trained model with the threshold tuned on the "
                "catalogue; each LASeC model with the CV-selected threshold).")


def clustering_ablation_table(R, W, cfg):
    m = R["metrics"]
    if m is None:
        return
    x = m[(m.family == "lasec") & (m.base_model == slug(cfg["embedding"]["base_model"]))]
    if x.algorithm.nunique() < 2:
        return W.skip("clustering_ablation", "no ablation algorithm evaluated")
    W.table("clustering_ablation", pd.DataFrame({
        "Subset": x.subset.map(SUBSET_LABEL), "Clustering": x.algorithm.map(ALGORITHM_LABEL), "Threshold (CV)": x.threshold.map(f3),
        "Groups": x.n_clusters, "GA": x.ga.map(f3), "95% CI": [ci(a, b) for a, b in zip(x.ga_ci_low, x.ga_ci_high)],
        "GA weighted": x.ga_weighted.map(f3), "Over-merged": x.n_overmerged_clusters, "Time (s)": x.runtime_s.map(lambda v: f"{v:.2f}")}),
        f"Clustering ablation on FPV with the fine-tuned {slug(cfg['embedding']['base_model'])} encoder "
        "(each algorithm with its own CV-selected threshold).")


def case_table(R, W, cfg):
    c, s = R["case_flight"], R["case_summary"]
    if c is None:
        return W.skip("case_study", "run case_study.py")
    prim = R["primary"]
    x = c[(c.method == prim) & (c.subset == "all")]
    if len(x):
        W.table("case_study_lasec", pd.DataFrame({
            "Flight": x.flight, "Sentences": x.n_sentences, "Unique": x.n_unique_sentences, "Triage events": x.n_triage_events,
            "Workload reduction": x.workload_reduction.map(lambda v: f"{v:.1%}"),
            "Key events recovered": [f"{a}/{b}" for a, b in zip(x.key_recovered, x.key_events)],
            "Hidden key events": x.hidden_key_events.astype(str)}),
            f"Forensic case study with {prim}: per-flight triage outcome (all log sentences).")
    rows = []
    for sub in ("all", "events"):
        y = s[s.subset == sub].copy()
        y["display"] = y.method
        llm = y[y.family.astype(str).str.startswith("llm")].copy()
        if len(llm):
            llm["conf"] = llm.method.str.replace(r"^LLM (.*) run\d+$", r"\1", regex=True)
            agg = llm.groupby(["conf", "family"]).agg(triage_events=("triage_events", "mean"), sentences=("sentences", "first"),
                                                     key_recovered=("key_recovered", "mean"), key_events=("key_events", "first")).reset_index()
            agg["display"] = agg.conf.map(lambda c: llm_display(c, cfg))
            y = pd.concat([y[~y.family.astype(str).str.startswith("llm")], agg], ignore_index=True)
        y = y[~y.family.isin(["pretrained", "lasec"]) | (y.method.str.contains(slug(cfg["embedding"]["base_model"]), regex=False))]
        for r in y.itertuples():
            rows.append((SUBSET_LABEL[sub], r.display, FAMILY_LABEL.get(r.family, r.family), int(r.sentences),
                         f"{r.triage_events:.1f}", f"{1 - r.triage_events / r.sentences:.1%}",
                         f"{r.key_recovered:.1f}/{int(r.key_events)}", f"{r.key_recovered / r.key_events:.1%}"))
    W.table("case_study_methods", pd.DataFrame(rows, columns=["Lines", "Method", "Family", "Log lines", "Triage events (sum)",
                                                              "Workload reduction", "Key events recovered", "Key-event recall"]),
            "Triage over the 7 flights: events presented to the investigator (sum over flights), workload reduction "
            "(1 − events / log lines) and recall of the expert key events.")


def repeat_table(R, W):
    rr = R["repeat_runs"]
    if rr is None:
        return W.skip("repeatability", "run repeatability.py")
    g = rr.groupby("method").agg(orders=("ga", "size"), ga_mean=("ga", "mean"), ga_sd=("ga", "std"), ga_min=("ga", "min"),
                                 ga_max=("ga", "max"), groups_min=("n_clusters", "min"), groups_max=("n_clusters", "max")).reset_index()
    W.table("repeatability", pd.DataFrame({
        "Model": g.method, "Input orders": g.orders, "GA mean ± sd": [f"{a:.3f} ± {b:.3f}" for a, b in zip(g.ga_mean, g.ga_sd)],
        "GA range": [f"{a:.3f}–{b:.3f}" for a, b in zip(g.ga_min, g.ga_max)],
        "Groups range": [f"{a}–{b}" for a, b in zip(g.groups_min, g.groups_max)]}),
        "Repeatability of LASeC on FPV under random input orders, per clustering algorithm (reported results use "
        "alphabetical order).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    R = load_all(cfg)
    out = path(cfg, "results") / "paper"
    W = Writer(out / "tables")
    W.text("# LASeC results summary\n")
    W.text(f"Primary model: {R['primary']}. Generated from {path(cfg, 'results')}.\n")
    dataset_tables(R, W, cfg)
    annotation_tables(R, W)
    selection_tables(R, W, cfg)
    cv_table(R, W, cfg)
    main_tables(R, W, cfg)
    significance_table(R, W)
    ablation_table(R, W, cfg)
    clustering_ablation_table(R, W, cfg)
    case_table(R, W, cfg)
    repeat_table(R, W)
    (out / "summary.md").write_text("\n".join(map(str, W.md)), encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
