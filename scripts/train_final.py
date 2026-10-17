"""Step 4: fine-tune the final LASeC encoder on ALL catalogue sentences.

The clustering threshold(s) are taken from the cross-validation summary (never from the FPV test set).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.experiment import load_catalogue, slug
from lasec.train import finetune
from lasec.utils import environment_info, load_config, path, read_json, set_seed, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    base = a.base_model or cfg["embedding"]["base_model"]
    set_seed(cfg["seed"])
    tr = dict(cfg["training"])
    if a.smoke:
        tr.update(epochs=1, negatives="sampled", neg_per_pos=2)
    cv = read_json(path(cfg, "results") / "cv" / slug(base) / "summary.json")
    cat = load_catalogue(cfg)
    out = path(cfg, "models") / f"lasec-{slug(base)}"
    info = finetune(base, cat.sentence.tolist(), cat.event.tolist(), out, seed=cfg["seed"], device=a.device,
                    max_pairs=2000 if a.smoke else None, **tr)
    meta = dict(base_model=base, algorithm=cv["algorithm"], threshold=cv["lasec"]["selected_threshold"],
                thresholds_by_algorithm={k: v["selected_threshold"] for k, v in cv["lasec_by_algorithm"].items()},
                threshold_source="cross-validation",
                n_sentences=len(cat), n_events=int(cat.event.nunique()), training=tr, smoke=a.smoke, **info,
                environment=environment_info())
    write_json(meta, out / "lasec_meta.json")
    print(meta)


if __name__ == "__main__":
    main()
