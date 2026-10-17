"""Step 2: tune baseline hyperparameters on the catalogue (training data), never on FPV.

  Drain3: grid over similarity threshold and tree depth, best GA on the catalogue.
  Pre-trained embedding + main clustering algorithm: threshold with the best GA on the catalogue.
"""
import argparse
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.baselines import drain3_groups
from lasec.cluster import embed, load_encoder
from lasec.experiment import load_catalogue, select_threshold, slug, threshold_curve
from lasec.metrics import grouping_report
from lasec.utils import load_config, path, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    cat = load_catalogue(cfg)
    s, e = cat.sentence.tolist(), cat.event.tolist()
    out = path(cfg, "results") / "baselines"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for sim, depth in itertools.product(cfg["drain3"]["sim_th_grid"], cfg["drain3"]["depth_grid"]):
        pred = drain3_groups(s, sim, depth)
        r = grouping_report(e, [pred[x] for x in s], s)
        rows.append(dict(sim_th=sim, depth=depth, ga=r["ga"], pair_f1=r["pair_f1"], n_clusters=r["n_clusters"]))
    grid = pd.DataFrame(rows).sort_values(["ga", "pair_f1", "sim_th"], ascending=[False, False, False])
    grid.to_csv(out / "drain3_grid.csv", index=False)
    best = grid.iloc[0]
    selection = dict(drain3=dict(sim_th=float(best.sim_th), depth=int(best.depth), catalogue_ga=float(best.ga)))

    selection["pretrained"] = {}
    for m in cfg["baselines"]["pretrained"]:
        X = embed(load_encoder(m), s, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
        c = threshold_curve(X, s, e, cfg)
        c.to_csv(out / f"pretrained_{slug(m)}_curve.csv", index=False)
        t = select_threshold(c)
        selection["pretrained"][m] = dict(threshold=t, algorithm=cfg["clustering"]["algorithm"],
                                          catalogue_ga=float(c[c.threshold == t].ga.iloc[0]))
    write_json(selection, out / "selection.json")
    print(selection)


if __name__ == "__main__":
    main()
