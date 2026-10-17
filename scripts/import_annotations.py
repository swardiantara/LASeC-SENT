"""Validate and freeze annotation sheets into CSV label files.

  --catalogue  annotation/catalogue_correction_sheet.xlsx (expert; column final_event)
  --catalogue-a2  annotation/catalogue_iaa_annotator2.xlsx (blind subset; column event)
  --fpv-a2     annotation/fpv_iaa_annotator2_blind.xlsx (blind; column event)
Writes data/labels/catalogue_sentence_labels.csv and updates event_annotator2 in fpv_sentence_labels.csv.
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
    ap.add_argument("--catalogue-a2", default=str(ROOT / "annotation" / "catalogue_iaa_annotator2.xlsx"))
    ap.add_argument("--fpv-a2", default=str(ROOT / "annotation" / "fpv_iaa_annotator2_blind.xlsx"))
    ap.add_argument("--proposals-as-final", action="store_true",
                    help="SMOKE TESTS ONLY: use the unreviewed proposals as labels")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()

    # ---------------- catalogue (expert)
    if os.path.exists(a.catalogue):
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
        if os.path.exists(a.catalogue_a2):
            a2 = pd.read_excel(a.catalogue_a2, sheet_name="Label")
            out["iaa_subset"] = out.sentence.isin(set(a2.sentence))
            out = out.merge(pd.DataFrame({"sentence": a2.sentence, "event_annotator2": clean(a2.event)}),
                            on="sentence", how="left")
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

    # ---------------- FPV annotator 2 (blind)
    if os.path.exists(a.fpv_a2):
        a2 = pd.read_excel(a.fpv_a2, sheet_name="Label")
        a2["event"] = clean(a2.event)
        if (a2.event != "").any():
            fpv = pd.read_csv(path(cfg, "fpv_labels"))
            fpv[cfg["labels"]["fpv_iaa_column"]] = fpv.sentence.map(dict(zip(a2.sentence, a2.event)))
            fpv.to_csv(path(cfg, "fpv_labels"), index=False)
            print(f"FPV annotator 2: {int((a2.event != '').sum())}/{len(a2)} sentences labelled")


if __name__ == "__main__":
    main()
