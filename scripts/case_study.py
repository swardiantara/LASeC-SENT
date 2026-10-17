"""Step 7: forensic case study on the 7 FPV flights.

For every method (from results/fpv/predictions) and every flight: number of sentences the analyst would
otherwise read, number of triage events, workload reduction, and recall of the expert key events.
For the primary LASeC model it also writes an annotated timeline per flight.
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.data import load_key_events
from lasec.experiment import EVENT_TYPES, load_fpv, method_file, primary_method
from lasec.metrics import representative, triage_report
from lasec.utils import load_config, path, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    fpv, occ = load_fpv(cfg)
    gold = dict(zip(fpv.sentence, fpv.event))
    keys = load_key_events(path(cfg, "fpv_key_events"))
    missing = sorted({k for v in keys.values() for k in v} - set(gold))
    if missing:
        sys.exit(f"key events not found among FPV sentences: {missing}")
    res = path(cfg, "results")
    out = res / "case_study"
    (out / "timelines").mkdir(parents=True, exist_ok=True)
    primary = primary_method(cfg)

    methods = pd.read_csv(res / "fpv" / "metrics.csv")
    methods = methods[methods.subset == "all"].reindex(columns=["method", "family", "llm_config"]).drop_duplicates("method")
    stype = dict(zip(fpv.sentence, fpv.get("sentence_type", pd.Series(["event"] * len(fpv)))))
    rows, hidden = [], {}
    for m in methods.itertuples(index=False):
        method = m.method
        p = pd.read_csv(res / "fpv" / "predictions" / f"{method_file(method)}.csv")
        pred = dict(zip(p.sentence, p.pred))
        for flight, sub in occ.groupby("flight"):
            for subset in ("all", "events"):
                lines = sub if subset == "all" else sub[sub.sentence.map(stype).isin(EVENT_TYPES)]
                r = triage_report(lines.sentence.tolist(), gold, pred, keys[flight])
                r.update(method=method, family=m.family, llm_config=m.llm_config, subset=subset, flight=flight,
                         n_cells=int(sub.cell_id.nunique()))
                hidden[f"{method}|{subset}|{flight}"] = r.pop("gold_events_hidden")
                rows.append(r)
            if method == primary:
                counts = sub.sentence.value_counts().to_dict()
                members = {}
                for s in counts:
                    members.setdefault(pred[s], []).append(s)
                rep = {c: representative(m, counts) for c, m in members.items()}
                key_gold = {gold[k] for k in keys[flight]}
                tl = sub[["time", "kind", "message", "sentence"]].copy()
                tl["triage_event"] = tl.sentence.map(lambda s: pred[s])
                tl["representative"] = tl.sentence.map(lambda s: rep[pred[s]])
                tl["gold_event"] = tl.sentence.map(gold)
                tl["expert_key_event"] = tl.gold_event.isin(key_gold)
                tl["sentence_type"] = tl.sentence.map(stype)
                tl.to_csv(out / "timelines" / f"flight{flight}_{method_file(method)}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_flight.csv", index=False)
    summary = df.groupby(["method", "family", "subset"], dropna=False).agg(
        sentences=("n_sentences", "sum"), unique_sentences=("n_unique_sentences", "sum"),
        triage_events=("n_triage_events", "sum"), key_events=("key_events", "sum"),
        key_recovered=("key_recovered", "sum"))
    summary["workload_reduction"] = 1 - summary.triage_events / summary.sentences
    summary["key_recall"] = summary.key_recovered / summary.key_events
    summary.to_csv(out / "summary.csv")
    write_json(hidden, out / "hidden_events.json")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
