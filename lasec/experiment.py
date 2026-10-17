"""Shared helpers for the experiment scripts."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster import cluster_sentences, embed, load_encoder, threshold_grid
from .data import attach_labels, load_labels
from .metrics import ga_correct, grouping_report
from .utils import path


def norm(s: str) -> str:
    return re.sub(r"[\s.!]+$", "", re.sub(r"\s+", " ", s.strip().lower()))


def slug(model: str) -> str:
    return Path(model.rstrip("/")).name


def method_file(method: str) -> str:
    """File-system name for a method's prediction file."""
    return re.sub(r"[^\w.-]+", "_", method).strip("_")


def primary_method(cfg) -> str:
    return f"LASeC ({slug(cfg['embedding']['base_model'])})"


EVENT_TYPES = ("event", "compound")


def _with_types(units, labels_csv, cfg):
    """Attach sentence_type and, if labels.event_sentences_only is set, keep event sentences only."""
    lab = pd.read_csv(labels_csv)
    if "sentence_type" in lab:
        units["sentence_type"] = units.sentence.map(dict(zip(lab.sentence, lab.sentence_type)))
        if cfg["labels"].get("event_sentences_only"):
            units = units[units.sentence_type.isin(EVENT_TYPES)].reset_index(drop=True)
    return units


def load_catalogue(cfg) -> pd.DataFrame:
    units = pd.read_csv(path(cfg, "processed") / "catalogue_sentences.csv")
    units = attach_labels(units, load_labels(path(cfg, "catalogue_labels"), cfg["labels"]["catalogue_column"]), "catalogue")
    return _with_types(units, path(cfg, "catalogue_labels"), cfg)


def load_fpv(cfg):
    """Unique labelled FPV sentences (with seen/unseen flags) and the occurrence table."""
    units = pd.read_csv(path(cfg, "processed") / "fpv_sentences.csv")
    units = attach_labels(units, load_labels(path(cfg, "fpv_labels"), cfg["labels"]["fpv_column"]), "FPV")
    units = _with_types(units, path(cfg, "fpv_labels"), cfg)
    occ = pd.read_csv(path(cfg, "processed") / "fpv_occurrences.csv")
    occ = occ[occ.sentence.isin(set(units.sentence))].reset_index(drop=True)
    if path(cfg, "catalogue_labels").exists():
        cat = load_catalogue(cfg)
        seen_str = {norm(s) for s in cat.sentence}
        units["seen_sentence"] = units.sentence.map(lambda s: norm(s) in seen_str)
        units["seen_event"] = units.event.isin(set(cat.event))
    return units, occ


def cluster_kwargs(cfg, algorithm=None, order=None, seed=0) -> dict:
    c = cfg["clustering"]
    return dict(algorithm=algorithm or c["algorithm"], order=order or c["order"], seed=seed,
                linkage=c.get("linkage", "average"), branching_factor=c.get("branching_factor", 50))


def algorithm_grid(cfg, algorithm=None):
    """Threshold grid of an algorithm (per-algorithm override in clustering.threshold_grids)."""
    c = cfg["clustering"]
    alg = algorithm or c["algorithm"]
    return threshold_grid(c.get("threshold_grids", {}).get(alg, c["threshold_grid"]))


def predict_embedding(model, sentences, threshold, cfg, algorithm=None, order=None, seed=0, encoder=None, X=None):
    enc = encoder or load_encoder(model)
    X = X if X is not None else embed(enc, sentences, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
    return cluster_sentences(list(sentences), X, threshold, **cluster_kwargs(cfg, algorithm, order, seed))


def threshold_curve(X, sentences, events, cfg, weights=None, algorithm=None, grid=None) -> pd.DataFrame:
    rows = []
    for t in (grid if grid is not None else algorithm_grid(cfg, algorithm)):
        pred = cluster_sentences(list(sentences), X, t, **cluster_kwargs(cfg, algorithm))
        p = [pred[s] for s in sentences]
        r = grouping_report(events, p, sentences, weights)
        rows.append(dict(threshold=t, ga=r["ga"], ga_weighted=r["ga_weighted"], pair_f1=r["pair_f1"],
                         ari=r["ari"], n_clusters=r["n_clusters"]))
    return pd.DataFrame(rows)


def select_threshold(curve: pd.DataFrame, metric="ga") -> float:
    """Best mean score; ties resolved towards the smaller (more conservative) threshold."""
    best = curve[metric].max()
    return float(curve[np.isclose(curve[metric], best)].threshold.min())


def subset_ga(gold, pred, mask):
    c = ga_correct(gold, pred)
    m = np.asarray(mask, bool)
    return float(c[m].mean()) if m.any() else float("nan")
