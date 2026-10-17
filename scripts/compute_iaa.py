"""Inter-annotator agreement on FPV (all sentences) and the catalogue IAA subset."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.iaa import pairwise_agreement
from lasec.utils import load_config, path, write_json


def report(df, a, b, name):
    d = df[df[a].notna() & df[b].notna() & (df[b].astype(str).str.strip() != "")]
    if len(d) < 2:
        print(f"{name}: no second-annotator labels yet")
        return None
    r = pairwise_agreement(d[a].astype(str).tolist(), d[b].astype(str).tolist())
    r["n_events_annotator1"], r["n_events_annotator2"] = int(d[a].nunique()), int(d[b].nunique())
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    out = {}
    fpv = pd.read_csv(path(cfg, "fpv_labels"))
    out["fpv"] = report(fpv, cfg["labels"]["fpv_column"], cfg["labels"]["fpv_iaa_column"], "FPV")
    if path(cfg, "catalogue_labels").exists():
        cat = pd.read_csv(path(cfg, "catalogue_labels"))
        if "event_annotator2" in cat:
            out["catalogue_subset"] = report(cat[cat.iaa_subset], "event", "event_annotator2", "catalogue subset")
    write_json(out, path(cfg, "results") / "annotation" / "iaa.json")


if __name__ == "__main__":
    main()
