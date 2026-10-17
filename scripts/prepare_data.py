"""Step 1: verify evidence integrity, load the sentence-level catalogue, segment the FPV logs.

The training catalogue uses the manual sentence split of the DroPTC dataset (vendor messages are noisy
and inconsistently punctuated, so automatic boundaries are unreliable there). The FPV evidence is
segmented automatically (';' then PySBD), exactly as LASeC does at inference time.

Outputs (data/processed/):
  catalogue_occurrences.csv  one row per (message, sentence) of the catalogue
  catalogue_sentences.csv    unique catalogue sentences (+ occurrence count, DroPTC problem type)
  fpv_occurrences.csv        one row per sentence occurrence in the 7 flights (timestamped)
  fpv_sentences.csv          unique FPV sentences (+ occurrence count)
  volume.json                cells -> atomic messages -> sentences counts
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.data import load_droptc_sentences, segment_fpv, segmentation_agreement, unique_sentences
from lasec.utils import load_config, path, read_json, sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--write-hashes", action="store_true", help="record SHA-256 of the decrypted CSVs (first time only)")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    seg = cfg["segmentation"]

    files = sorted(glob.glob(os.path.join(path(cfg, "fpv_decrypted"), "*.csv")))
    hashes = {os.path.basename(f): sha256(f) for f in files}
    hf = path(cfg, "fpv_decrypted_hashes")
    if a.write_hashes or not hf.exists():
        write_json(hashes, hf)
        print(f"wrote {hf}")
    else:
        expected = read_json(hf)
        bad = [k for k in expected if hashes.get(k) != expected[k]]
        if bad or set(hashes) != set(expected):
            sys.exit(f"integrity check FAILED for {bad or set(hashes) ^ set(expected)}")
        print(f"integrity check passed for {len(files)} decrypted flight logs")

    out = path(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    # training catalogue: manual sentence segmentation from DroPTC (not PySBD)
    cat = load_droptc_sentences(path(cfg, "catalogue_train"), path(cfg, "catalogue_test"), path(cfg, "catalogue_messages"))
    cat.to_csv(out / "catalogue_occurrences.csv", index=False)
    uniq = unique_sentences(cat)
    pt = cat.groupby("sentence").problem_type.agg(lambda x: "/".join(sorted(set(x))))
    uniq["problem_type"] = uniq.sentence.map(pt)
    uniq.to_csv(out / "catalogue_sentences.csv", index=False)
    seg_agree = segmentation_agreement(cat, seg["message_separator"], seg["language"])

    fpv = segment_fpv(path(cfg, "fpv_decrypted"), seg["message_separator"], seg["language"])
    fpv.to_csv(out / "fpv_occurrences.csv", index=False)
    unique_sentences(fpv).to_csv(out / "fpv_sentences.csv", index=False)

    vol = dict(
        catalogue=dict(source="DroPTC manual sentence segmentation", messages=int(cat.line_id.nunique()),
                       sentences=len(cat), unique_sentences=int(cat.sentence.nunique()),
                       pysbd_vs_manual_segmentation=seg_agree),
        fpv=dict(cells=int(fpv.cell_id.nunique()), unique_cells=int(fpv.drop_duplicates("cell_id").cell.nunique()),
                 atomic_messages=int(fpv.groupby(["cell_id", "msg_part"]).ngroups),
                 unique_atomic_messages=int(fpv.message.nunique()),
                 sentences=len(fpv), unique_sentences=int(fpv.sentence.nunique()),
                 per_flight=fpv.groupby("flight").agg(sentences=("sentence", "size"),
                                                      unique_sentences=("sentence", "nunique")).to_dict("index")),
    )
    write_json(vol, out / "volume.json")
    print(vol)


if __name__ == "__main__":
    main()
