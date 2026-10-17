"""Validate and freeze the expert-reviewed catalogue labels into a CSV label file.

  --catalogue  annotation/catalogue_review_v2.xlsx (expert review; columns final_label / final_type)
Writes data/labels/catalogue_sentence_labels.csv (with the proposed labels, so the correction rate is auditable)
and results/annotation/catalogue_annotation_stats.json.
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.utils import ROOT, load_config, path, write_json

ID = re.compile(r"^[A-Z0-9][A-Z0-9_]*$")


def clean(series):
    return series.astype(str).str.strip().replace({"nan": ""})


def check_ids(ids, where):
    bad = sorted({i for i in ids if not ID.match(i)})
    if bad:
        print(f"WARNING {where}: {len(bad)} IDs are not UPPER_SNAKE_CASE, e.g. {bad[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--catalogue", default=str(ROOT / "annotation" / "catalogue_review_v2.xlsx"))
    ap.add_argument("--proposals-as-final", action="store_true",
                    help="SMOKE TESTS ONLY: use the unreviewed proposals as labels")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()

    if not os.path.exists(a.catalogue):
        sys.exit(f"review workbook not found: {a.catalogue}")
    df = pd.read_excel(a.catalogue, sheet_name="Sentences", keep_default_na=False)
    df = df.rename(columns={"proposed type": "proposed_type", "proposed label": "proposed_label"})
    if a.proposals_as_final:
        df["final_label"], df["final_type"] = df["proposed_label"], df["proposed_type"]
    for c in ("final_label", "final_type", "reviewer_notes"):
        df[c] = clean(df[c])
    blank = df[(df.final_label == "") | (df.final_type == "")]
    if len(blank):
        sys.exit(f"{len(blank)} catalogue sentences lack final_label/final_type, e.g. {blank.sentence.head(3).tolist()}")
    check_ids(df.final_label, "catalogue")
    is_event = df.final_type.isin(["event", "compound"])
    mixed = df.assign(_ev=is_event).groupby("final_label")["_ev"].nunique() > 1
    if mixed.any():
        sys.exit(f"labels mixing event and non-event sentences: {mixed[mixed].index.tolist()}")
    changed = (df.final_label != df.proposed_label) | (df.final_type != df.proposed_type)
    out = pd.DataFrame({"sentence": df.sentence, "event": df.final_label, "sentence_type": df.final_type,
                        "proposed_label": df.proposed_label, "proposed_type": df.proposed_type,
                        "changed": changed, "notes": df.reviewer_notes})
    out.to_csv(path(cfg, "catalogue_labels"), index=False)
    stats = dict(sentences=len(out), labels=int(out.event.nunique()),
                 event_sentences=int(is_event.sum()), event_labels=int(out[is_event.values].event.nunique()),
                 sentence_types=out.sentence_type.value_counts().to_dict(),
                 correction_rate=float(changed.mean()),
                 correction_rate_by_proposed_type=out.groupby("proposed_type").changed.mean().round(3).to_dict(),
                 unsure=int(out.notes.str.contains("UNSURE", case=False).sum()))
    write_json(stats, path(cfg, "results") / "annotation" / "catalogue_annotation_stats.json")
    print("catalogue:", stats)

    fpv = pd.read_csv(path(cfg, "fpv_labels"))
    both = fpv.merge(out[["sentence", "event"]], on="sentence")
    diff = both[both.event != both[cfg["labels"]["fpv_column"]]]
    if len(diff):
        print(f"WARNING: {len(diff)} sentences occur in both sets with different IDs:\n{diff.to_string()}")


if __name__ == "__main__":
    main()
