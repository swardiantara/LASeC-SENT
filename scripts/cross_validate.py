"""Step 3: k-fold cross-validation on the catalogue (training data only).

Purpose: (i) choose the clustering threshold (main algorithm and each ablation algorithm) without looking at the FPV test set, and (ii) estimate how
LASeC generalises to sentences it was not fine-tuned on. Folds split the UNIQUE catalogue sentences.
In each fold the encoder is fine-tuned on the training sentences only and the held-out fold is
clustered on its own over the threshold grid. Test-fold sentences whose event also occurs in the
training folds ('seen event') and those whose event does not ('novel event') are scored separately.
The selected threshold is the one with the best mean GA over folds.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.cluster import embed, load_encoder
from lasec.experiment import cluster_kwargs, load_catalogue, select_threshold, slug, subset_ga, threshold_curve
from lasec.cluster import cluster_sentences
from lasec.train import finetune
from lasec.utils import environment_info, load_config, path, set_seed, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-models", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run for testing the pipeline")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    base = a.base_model or cfg["embedding"]["base_model"]
    set_seed(cfg["seed"])
    tr = dict(cfg["training"])
    n_folds = cfg["cv"]["n_folds"]
    if a.smoke:
        tr.update(epochs=1, negatives="sampled", neg_per_pos=2)
        n_folds = 2
    cat = load_catalogue(cfg)
    sents, events = cat.sentence.to_numpy(), cat.event.to_numpy()
    out = path(cfg, "results") / "cv" / slug(base)
    out.mkdir(parents=True, exist_ok=True)

    main_alg = cfg["clustering"]["algorithm"]
    ablations = [x for x in cfg["clustering"].get("ablation_algorithms", []) if x != main_alg]
    curves, folds = [], []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg["seed"])
    base_enc = load_encoder(base, a.device)
    for k, (tr_i, te_i) in enumerate(kf.split(sents)):
        print(f"fold {k + 1}/{n_folds}: train {len(tr_i)} / test {len(te_i)} sentences")
        mdir = path(cfg, "models") / "cv" / slug(base) / f"fold{k}"
        info = finetune(base, list(sents[tr_i]), list(events[tr_i]), mdir, seed=cfg["seed"], device=a.device,
                        max_pairs=2000 if a.smoke else None, **tr)
        te_s, te_e = list(sents[te_i]), list(events[te_i])
        for name, enc in (("lasec", load_encoder(str(mdir), a.device)), ("pretrained", base_enc)):
            X = embed(enc, te_s, cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"])
            # main algorithm for both encoders; ablation algorithms for the fine-tuned encoder
            for alg in [main_alg] + (ablations if name == "lasec" else []):
                c = threshold_curve(X, te_s, te_e, cfg, algorithm=alg)
                c["fold"], c["encoder"], c["algorithm"] = k, name, alg
                curves.append(c)
            if name == "lasec":
                Xl = X
        novel = ~np.isin(te_e, events[tr_i])
        folds.append(dict(fold=k, n_train=len(tr_i), n_test=len(te_i), test_novel_event_share=float(novel.mean()),
                          **{f"train_{kk}": v for kk, v in info.items() if kk != "loss_per_epoch"}, X=Xl, te_s=te_s,
                          te_e=te_e, novel=novel))
        if not a.keep_models:
            shutil.rmtree(mdir, ignore_errors=True)

    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out / "fold_curves.csv", index=False)
    mean = curves.groupby(["encoder", "algorithm", "threshold"], as_index=False)[
        ["ga", "ga_weighted", "pair_f1", "ari", "n_clusters"]].mean()
    mean.to_csv(out / "mean_curves.csv", index=False)
    summary = dict(base_model=base, n_folds=n_folds, smoke=a.smoke, training=tr, algorithm=main_alg,
                   environment=environment_info(), lasec_by_algorithm={})

    def pick(enc_name, alg):
        m = mean[(mean.encoder == enc_name) & (mean.algorithm == alg)]
        t = select_threshold(m, "ga")
        per_fold = curves[(curves.encoder == enc_name) & (curves.algorithm == alg) & (curves.threshold == t)]
        return dict(algorithm=alg, selected_threshold=t, ga_mean=float(per_fold.ga.mean()), ga_sd=float(per_fold.ga.std()),
                    pair_f1_mean=float(per_fold.pair_f1.mean()), ari_mean=float(per_fold.ari.mean()))

    summary["lasec"], summary["pretrained"] = pick("lasec", main_alg), pick("pretrained", main_alg)
    for alg in [main_alg] + ablations:
        summary["lasec_by_algorithm"][alg] = pick("lasec", alg)
    # seen-event vs novel-event GA of the fine-tuned model at the selected threshold
    t = summary["lasec"]["selected_threshold"]
    seen, novel = [], []
    for f in folds:
        pred = cluster_sentences(f["te_s"], f["X"], t, **cluster_kwargs(cfg))
        p = [pred[s] for s in f["te_s"]]
        seen.append(subset_ga(f["te_e"], p, ~f["novel"]))
        novel.append(subset_ga(f["te_e"], p, f["novel"]))
    summary["lasec"].update(ga_seen_event=float(np.nanmean(seen)), ga_novel_event=float(np.nanmean(novel)),
                            novel_event_share=float(np.mean([f["test_novel_event_share"] for f in folds])))
    summary["folds"] = [{k: v for k, v in f.items() if k not in ("X", "te_s", "te_e", "novel")} for f in folds]
    write_json(summary, out / "summary.json")
    print({k: summary[k] for k in ("lasec", "pretrained")})


if __name__ == "__main__":
    main()
