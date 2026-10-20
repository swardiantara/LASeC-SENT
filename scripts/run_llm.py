"""Step 5: LLM grouping baselines on the FPV test sentences, via OpenRouter.

  python scripts/run_llm.py                              # every enabled model, zero- and few-shot
  python scripts/run_llm.py --model claude-opus-5.5 --mode few
  python scripts/run_llm.py --mode zero --fixed-order [--temperature 0]   # variance check -> results/llm_variance/
Few-shot demonstrations come from the labelled catalogue only, excluding any sentence that also occurs
in the FPV set. Raw responses are cached in results/llm/cache/, so predictions can be re-derived without
API access (a cached run makes no API call).
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.experiment import load_catalogue, load_fpv
from lasec.llm.grouping import llm_group, make_grouper
from lasec.llm.prompts import build_demonstrations
from lasec.utils import load_config, path, write_json


def run_one(cfg, mcfg, mode, sents, demo, runs, fixed_order=False, temperature=None):
    name = f"{mcfg['name']}_{mode}"
    variance = fixed_order or temperature is not None
    if variance:  # kept out of results/llm so that the main evaluation does not pick these runs up
        name += ("_fixed" if fixed_order else "_shuffled") + ("" if temperature is None else f"_t{temperature:g}")
    out = path(cfg, "results") / ("llm_variance" if variance else "llm") / name
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for run in range(runs):
        cache = path(cfg, "results") / "llm" / "cache"
        try:
            pred, rec = llm_group(sents, _Lazy(cfg, mcfg, temperature), cache, f"{name}_run{run}",
                                  demo if mode == "few" else None, cfg["llm"]["batch_size"],
                                  seed=0 if fixed_order else run)
        except Exception as e:  # keep going with the other models; the failure is recorded
            print(f"  {name} run {run} FAILED: {type(e).__name__}: {e}")
            records.append(dict(run=run, error=f"{type(e).__name__}: {e}"))
            continue
        pd.DataFrame({"sentence": sents, "pred": [pred[s] for s in sents]}).to_csv(out / f"run{run}.csv", index=False)
        records.append(rec)
        print(f"  {name} run {run}: {len(set(pred.values()))} groups")
    defaults = dict(cfg["llm"]["defaults"], **({} if temperature is None else {"temperature": temperature}))
    write_json(dict(model=mcfg, defaults=defaults, mode=mode, fixed_order=fixed_order,
                    demonstration=demo if mode == "few" else None, runs=records), out / "meta.json")


class _Lazy:
    """Create the API client only when a response is not cached (so cached re-runs need no key)."""
    def __init__(self, cfg, mcfg, temperature=None):
        self.cfg, self.mcfg, self.g = cfg, mcfg, None
        self.defaults = dict(cfg["llm"]["defaults"], **({} if temperature is None else {"temperature": temperature}))

    def __call__(self, system, user):
        if self.g is None:
            self.g = make_grouper(self.mcfg, self.defaults)
        t = time.perf_counter()
        r = self.g(system, user)
        r["latency_s"] = time.perf_counter() - t
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="name from llm.models (default: all enabled)")
    ap.add_argument("--mode", default="both", choices=["zero", "few", "both"])
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--fixed-order", action="store_true", help="same sentence order in every run (variance check)")
    ap.add_argument("--temperature", type=float, default=None, help="override the provider default (variance check)")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    fpv, _ = load_fpv(cfg)
    sents = sorted(fpv.sentence)
    cat = load_catalogue(cfg)
    fs = cfg["llm"]["few_shot"]
    demo = build_demonstrations(cat.sentence.tolist(), cat.event.tolist(), exclude=sents,
                                n_groups=fs["n_groups"], max_members=fs["max_members"], seed=cfg["seed"])
    models = [m for m in cfg["llm"]["models"] if (a.model is None and m.get("enabled", True)) or m["name"] == a.model]
    if not models:
        sys.exit(f"no model named {a.model!r} in llm.models")
    for m in models:
        for mode in (["zero", "few"] if a.mode == "both" else [a.mode]):
            print(f"{m['name']} ({m['model']}) {mode}-shot")
            run_one(cfg, m, mode, sents, demo, a.runs or cfg["llm"]["runs"], a.fixed_order, a.temperature)


if __name__ == "__main__":
    main()
