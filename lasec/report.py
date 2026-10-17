"""Shared loaders and naming for the table and figure scripts."""
import glob
import json
import os

import pandas as pd

from .experiment import primary_method, slug
from .utils import path, read_json

FAMILY_ORDER = ["string", "parser", "pretrained", "lasec", "llm-zero", "llm-few"]
FAMILY_LABEL = {"string": "String de-duplication", "parser": "Log parser", "pretrained": "Pre-trained embedding + clustering",
                "lasec": "LASeC (fine-tuned embedding + clustering)", "llm-zero": "LLM zero-shot", "llm-few": "LLM few-shot"}
BASE_ORDER = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1", "all-mpnet-base-v2"]
# parameter counts from the sentence-transformers model cards
BASE_PARAMS_M = {"all-MiniLM-L6-v2": 22.7, "all-MiniLM-L12-v2": 33.4, "all-distilroberta-v1": 82.1, "all-mpnet-base-v2": 109.5}
SUBSET_LABEL = {"all": "all sentences", "events": "event sentences"}


def _read_csv(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def _read_json(p):
    return read_json(p) if os.path.exists(p) else None


def llm_display(config_name: str, cfg) -> str:
    """'claude-opus-5.5_few' -> 'claude-opus-5.5 (few-shot)'."""
    name, mode = config_name.rsplit("_", 1)
    return f"{name} ({mode}-shot)"


def load_all(cfg) -> dict:
    res, proc, labels = path(cfg, "results"), path(cfg, "processed"), path(cfg, "catalogue_labels").parent
    r = dict(cfg=cfg, primary=primary_method(cfg))
    r["volume"] = _read_json(proc / "volume.json")
    r["cat_labels"] = _read_csv(path(cfg, "catalogue_labels"))
    r["cat_sentences"] = _read_csv(proc / "catalogue_sentences.csv")
    r["fpv_labels"] = _read_csv(path(cfg, "fpv_labels"))
    r["fpv_occ"] = _read_csv(proc / "fpv_occurrences.csv")
    r["ann_stats"] = _read_json(res / "annotation" / "catalogue_annotation_stats.json")
    r["selection"] = _read_json(res / "baselines" / "selection.json")
    sel = res / "selection"
    r["model_selection"] = _read_json(sel / "selection.json")
    r["alg_grid"], r["alg_summary"] = _read_csv(sel / "algorithm_grid.csv"), _read_csv(sel / "algorithm_summary.csv")
    r["emb_grid"], r["emb_summary"] = _read_csv(sel / "embedding_grid.csv"), _read_csv(sel / "embedding_summary.csv")
    r["cv"] = {}
    for f in glob.glob(str(res / "cv" / "*" / "summary.json")):
        s = read_json(f)
        s["mean_curves"] = _read_csv(os.path.join(os.path.dirname(f), "mean_curves.csv"))
        s["fold_curves"] = _read_csv(os.path.join(os.path.dirname(f), "fold_curves.csv"))
        r["cv"][slug(s["base_model"])] = s
    r["lasec_meta"] = {slug(read_json(f)["base_model"]): read_json(f)
                       for f in glob.glob(str(path(cfg, "models") / "lasec-*" / "lasec_meta.json"))}
    r["metrics"] = _read_csv(res / "fpv" / "metrics.csv")
    if r["metrics"] is not None:
        for c in ("threshold", "base_model", "algorithm", "cost_usd", "llm_config", "llm_run"):
            if c not in r["metrics"]:
                r["metrics"][c] = float("nan")
    r["llm_summary"] = _read_csv(res / "fpv" / "llm_summary.csv")
    r["significance"] = _read_csv(res / "fpv" / "significance.csv")
    r["sensitivity"] = _read_csv(res / "fpv" / "sensitivity.csv")
    om = res / "fpv" / "overmerges.json"
    r["overmerges"] = json.loads(om.read_text(encoding="utf-8")) if om.exists() else {}
    r["case_flight"] = _read_csv(res / "case_study" / "per_flight.csv")
    r["case_summary"] = _read_csv(res / "case_study" / "summary.csv")
    r["repeat_runs"] = _read_csv(res / "repeatability" / "order_runs.csv")
    r["repeat"] = _read_json(res / "repeatability" / "summary.json")
    return r


def reported_bases(cfg) -> list[str]:
    """Encoders shown in the main tables: the primary model, then the lightweight variant (if configured)."""
    base, light = slug(cfg["embedding"]["base_model"]), cfg["embedding"].get("lightweight_model")
    return [base] + ([slug(light)] if light and slug(light) != base else [])


def method_table(metrics: pd.DataFrame, subset: str, cfg, primary_base_only=True) -> pd.DataFrame:
    """One row per method for a subset; LLM runs collapsed to mean/sd per configuration, in display order.

    With primary_base_only, embedding rows are restricted to the main clustering algorithm: the pre-trained
    primary encoder and LASeC with the primary and lightweight encoders."""
    m = metrics[metrics.subset == subset].copy()
    if primary_base_only:
        bases, main = reported_bases(cfg), cfg["clustering"]["algorithm"]
        keep = (((m.family == "pretrained") & (m.base_model == bases[0]))
                | ((m.family == "lasec") & m.base_model.isin(bases) & (m.algorithm == main)))
        m = m[~m.family.isin(["pretrained", "lasec"]) | keep]
    non = m[~m.family.str.startswith("llm")].copy()
    non["display"] = non.method
    non["ga_sd"] = float("nan")
    non["runs"] = 1
    llm = m[m.family.str.startswith("llm")]
    rows = []
    for conf, g in llm.groupby("llm_config"):
        row = g.iloc[0].copy()
        for c in ("ga", "ga_weighted", "pair_precision", "pair_recall", "pair_f1", "n_clusters",
                  "n_overmerged_clusters", "runtime_s", "cost_usd", "ga_unseen_sentence", "ga_seen_sentence"):
            if c in g:
                row[c] = g[c].mean()
        row["ga_sd"] = g.ga.std() if len(g) > 1 else float("nan")
        row["runs"] = len(g)
        row["display"] = llm_display(conf, cfg)
        row["ga_ci_low"] = row["ga_ci_high"] = float("nan")
        rows.append(row)
    out = pd.concat([non, pd.DataFrame(rows)], ignore_index=True) if rows else non
    out["_f"] = out.family.map({f: i for i, f in enumerate(FAMILY_ORDER)})
    return out.sort_values(["_f", "display"]).drop(columns="_f").reset_index(drop=True)
