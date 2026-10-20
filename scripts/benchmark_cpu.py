"""CPU-only runtime of the LASeC pipeline (deployment on a workstation without a GPU).

Speed depends only on the architecture, so the pre-trained checkpoints are timed (a fine-tuned LASeC model has the
same architecture and size; pass --model-dir to time a trained model directly). For each encoder:
  load_s            loading the encoder from the local cache
  case_*            the FPV case: segmentation of all tip/warning cells + encoding of the 78 unique sentences +
                    agglomerative clustering (median of --repeats runs after one warm-up)
  sentences_per_s   encoding throughput on the 783 catalogue sentences (best of 3)
Output: results/runtime/cpu_benchmark.csv and cpu_environment.json (hardware + library versions).
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.cluster import cluster_sentences, embed, load_encoder
from lasec.experiment import cluster_kwargs, load_catalogue, load_fpv, slug
from lasec.segment import split_messages, split_sentences
from lasec.utils import environment_info, load_config, path, read_json, write_json


def hardware():
    info = dict(platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count())
    if sys.platform == "win32":
        ps = ("$c=Get-CimInstance Win32_Processor|Select -First 1;$m=(Get-CimInstance Win32_PhysicalMemory|"
              "Measure Capacity -Sum).Sum;$s=Get-CimInstance Win32_ComputerSystem;"
              "\"$($c.Name.Trim())|$($c.NumberOfCores)|$($c.NumberOfLogicalProcessors)|$([math]::Round($m/1GB,1))|"
              "$($s.Manufacturer) $($s.Model)\"")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                 timeout=60).stdout.strip().split("|")
            info.update(cpu=out[0], cores=int(out[1]), threads=int(out[2]), ram_gb=float(out[3]), machine=out[4])
        except Exception as e:  # informational only
            info["hardware_error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None, help="torch threads (default: torch's choice)")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    fpv, _ = load_fpv(cfg)
    sents = sorted(fpv.sentence)
    cells = pd.read_csv(path(cfg, "processed") / "fpv_occurrences.csv").drop_duplicates("cell_id")
    cell_texts = cells["cell"].astype(str).tolist()
    cat = load_catalogue(cfg).sentence.tolist()
    thresholds = {slug(k): v["threshold"] for k, v in read_json(path(cfg, "results") / "baselines" / "selection.json")["pretrained"].items()}
    lasec_t = {}
    for b in cfg["embedding"]["base_model_variants"]:
        f = path(cfg, "results") / "cv" / slug(b) / "summary.json"
        if f.exists():
            lasec_t[slug(b)] = read_json(f)["lasec"]["selected_threshold"]
    rows = []
    for m in cfg["embedding"]["base_model_variants"]:
        t0 = time.perf_counter()
        enc = load_encoder(m, device="cpu")
        load_s = time.perf_counter() - t0
        t = lasec_t.get(slug(m), thresholds.get(slug(m), 0.2))
        bs, norm = cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"]

        def case():
            t1 = time.perf_counter()
            if cell_texts:
                for c in cell_texts:
                    for msg in split_messages(c, cfg["segmentation"]["message_separator"]):
                        split_sentences(msg)
            t2 = time.perf_counter()
            X = embed(enc, sents, bs, norm)
            t3 = time.perf_counter()
            cluster_sentences(sents, X, t, **cluster_kwargs(cfg))
            t4 = time.perf_counter()
            return t2 - t1, t3 - t2, t4 - t3

        case()  # warm-up
        runs = [case() for _ in range(a.repeats)]
        embed(enc, cat[:16], bs, norm)
        best = min(_timed(lambda: embed(enc, cat, bs, norm)) for _ in range(3))
        seg, emb, clu = (statistics.median(r[i] for r in runs) for i in range(3))
        rows.append(dict(encoder=slug(m), params_m=round(sum(p.numel() for p in enc.parameters()) / 1e6, 1),
                         load_s=load_s, case_segment_s=seg, case_encode_s=emb, case_cluster_s=clu,
                         case_total_s=seg + emb + clu, case_sentences=len(sents),
                         sentences_per_s=len(cat) / best, torch_threads=torch.get_num_threads()))
        print(rows[-1])
    out = path(cfg, "results") / "runtime"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "cpu_benchmark.csv", index=False)
    write_json(dict(hardware=hardware(), environment=environment_info(), repeats=a.repeats,
                    note="CPU only; fine-tuned models share the architecture of these checkpoints"),
               out / "cpu_environment.json")


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


if __name__ == "__main__":
    main()
