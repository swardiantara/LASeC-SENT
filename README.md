# LASeC: sentence-level event abstraction of drone flight logs for forensic triage

LASeC groups the human-readable messages of drone flight logs into **events**, so that an investigator
reviews one representative sentence per event instead of every log line. Pipeline:

```
decrypted flight log -> APP.tip / APP.warning cells -> split at ";" (app messages) -> PySBD (sentences)
   -> sentence encoder fine-tuned with contrastive loss -> agglomerative clustering -> EventID + representative sentence
```

This repository (LASeC-SENT) contains the code, data, labels and annotation material needed to reproduce every number in the paper. It supersedes the message-level prototype (`drone-log-abstract`).

## Evaluation design at a glance

| Choice | Setting | Why |
|---|---|---|
| Unit of analysis | sentence (`;` split, then PySBD) | one sentence = one event; avoids the first-sentence bias of message-level abstraction ([DroPTC](https://doi.org/10.1016/j.fsidi.2026.302051)) |
| Training data | DroPTC sentence-level catalogue: 585 vendor messages manually split into 1,035 sentences (783 unique), labelled per `annotation/GUIDELINE.md` | manual boundaries, because catalogue messages are inconsistently punctuated; the only labelled data seen by any model |
| Test data | 7 DJI FPV flights → 2,158 sentence occurrences, 78 unique sentences, expert labels | held out; never used for training, tuning or threshold selection |
| Model selection | Stage A: clustering algorithms (Agglomerative, BIRCH, DBSCAN, HDBSCAN, OPTICS) with pre-trained SBERT; Stage B: pre-trained encoders with the chosen algorithm, plus throughput. Final: Agglomerative (average linkage, cosine) + all-distilroberta-v1; all-MiniLM-L6-v2 is reported as a lightweight variant for CPU-only workstations | catalogue only, pre-trained encoders; agglomerative is order-invariant and exact at flight-log scale |
| Threshold | cosine-distance threshold selected by 5-fold cross-validation on the catalogue (BIRCH ablation: its own CV threshold) | no test-set tuning |
| Baselines | exact dedup, regex-masked dedup, Drain3, pre-trained embedding + agglomerative, LLM zero/few-shot via OpenRouter (one model each from OpenAI, Anthropic and Google) | alternative ways an investigator could obtain event groups; all hyperparameters tuned on the catalogue |
| Metrics | reported on all sentences and on event sentences only; GA (unique and occurrence-weighted), pairwise P/R/F1, ARI, NMI, over-merges; per-flight workload reduction and expert key-event recall | grouping quality + forensic utility |
| Statistics | bootstrap 95% CIs (items and flights), paired bootstrap + exact McNemar, Holm correction | |
| Annotation | LLM-proposed labels under `annotation/GUIDELINE.md`, reviewed by the expert (catalogue: no changes; FPV: 9/78 sentences regrouped); proposals and corrections are kept in `data/labels/catalogue_sentence_labels.csv` | auditable correction rate |

## Repository layout

```
configs/default.yaml        every experimental setting (seeds, models, grids, providers)
configs/smoke.yaml          tiny CPU run to test the plumbing (numbers are meaningless)
data/raw/catalogue/         DroPTC sentence split (train+test = LASeC training catalogue) + original message list
data/raw/fpv/decrypted/     7 decrypted DJI FPV flight records (test) + SHA-256 hashes
data/labels/                fpv_sentence_labels.csv, fpv_key_events.yaml, catalogue_sentence_labels.csv
annotation/                 GUIDELINE.md (v2) and the LLM-proposed catalogue labels with rationales
lasec/                      library: segmentation, data, training, clustering, baselines, LLM, metrics, stats, report
scripts/                    numbered pipeline steps (see below)
tests/                      unit tests (pytest)
results/                    outputs (results/paper/ = tables, figures, summary.md)
```

## Setup

```bash
conda env create -f environment.yml      # or: install PyTorch, then pip install -r requirements.txt
conda activate lasec
pytest -q tests
cp .env.example .env                     # then paste your OpenRouter key into .env
```

`.env` is read automatically by every script (variables already set in the shell take precedence) and is
git-ignored. Only the LLM baselines need the key.

GPU is recommended for fine-tuning (5 CV folds + 1 final model per base model). Everything else runs on CPU.

## Reproducing the results

```bash
bash run_all.sh            # full run
bash run_all.sh --smoke    # quick CPU check of the whole pipeline incl. tables/figures (no LLM calls)
```

| Step | Script | Output |
|---|---|---|
| 1 | `prepare_data.py` | evidence hash check; `data/processed/*` (segmented sentences, `volume.json`) |
| 2a | `select_model.py` | `results/selection/` (algorithm grid, encoder grid + throughput, `selection.json`) |
| 2b | `tune_baselines.py` | `results/baselines/selection.json` (Drain3 parameters, pre-trained thresholds) |
| 3 | `cross_validate.py --base-model M` | `results/cv/M/summary.json` (selected threshold, CV GA, seen- vs novel-event GA) |
| 4 | `train_final.py --base-model M` | `models/lasec-M/` + `lasec_meta.json` |
| 5 | `run_llm.py [--model NAME] [--mode zero/few/both]` | `results/llm/…/run*.csv`, raw responses in `results/llm/cache/` |
| 6 | `evaluate_fpv.py` | `results/fpv/metrics.csv` (per method x subset), `significance.csv`, `llm_summary.csv`, `sensitivity.csv`, `overmerges.json`, predictions |
| 7 | `case_study.py` | `results/case_study/per_flight.csv`, `summary.csv`, annotated timelines |
| 8 | `repeatability.py` | `results/repeatability/` (input-order spread per clustering algorithm) |
| 9 | `make_tables.py` | `results/paper/tables/*.tex` + `.csv`, `results/paper/summary.md` |
| 10 | `make_figures.py` | `results/paper/figures/*.pdf` + `.png` |

To regenerate one figure only, e.g. the t-SNE plot (it needs the trained model in `models/lasec-<base model>/`):
`python scripts/make_figures.py --only tsne`.

**LLM baselines.** All models are called through OpenRouter: set `OPENROUTER_API_KEY` in `.env`. The models (one each from OpenAI,
Anthropic and Google: gpt-5.6-luna, claude-opus-5.5, gemini-3.8-flash) are listed in `configs/default.yaml` under `llm.models`; IDs were
checked against the OpenRouter model list on 2026-10-17. Each model runs zero- and few-shot, 3 times with
different presentation orders, with strict JSON-schema output and `reasoning.effort: high`. Raw responses
(including latency, cost and the model that actually served the request) are cached in `results/llm/cache/`, so
`evaluate_fpv.py` reproduces the LLM numbers without API access when the cache is shipped. A failed run is
recorded in the model's `meta.json` and the remaining models continue.

**Two evaluation views.** Every method groups all FPV sentences; scores are reported for *all sentences* and for
*event sentences* only (instructions, mitigations, possible events and information removed after grouping).
Setting `labels.event_sentences_only: true` instead removes non-event sentences before training and grouping.

**How the LLM groups sentences.** The prompt lists the test sentences with numeric IDs and asks for a JSON
partition `{"groups": [{"event": NAME, "members": [ids]}]}` (`lasec/llm/prompts.py`). *Zero-shot* gives only
the event-identity rules from the annotation guideline. *Few-shot* adds one solved demonstration of the same
task: 15 labelled catalogue events (multi-member and singleton) with their correct grouping. Demonstration
sentences that also occur in the test set are excluded. The output is validated: every ID exactly once, and
missing IDs become singletons and are logged.

## Annotation workflow

See `annotation/GUIDELINE.md`.

```bash
python scripts/build_review_sheet.py      # review workbook from annotation/catalogue_labels_v2_proposed.csv
# the expert edits annotation/catalogue_review_v2.xlsx (final_type / final_label)
python scripts/import_annotations.py      # validate + freeze labels into data/labels/
```

The frozen catalogue labels keep the proposed label and type next to the final ones (`changed` column), so the
expert correction rate can be checked. The FPV test labels were drafted the same way (LLM proposal under the
guideline) and verified by the expert, who changed the grouping of 9 of the 78 sentences.

## Reproducibility notes

- Seeds are fixed (`seed: 42`). The main clustering (agglomerative, average linkage on cosine distances) does not
  depend on input order; the BIRCH ablation does, so sentences are fed in alphabetical order and
  `repeatability.py` reports the spread over random orders for each algorithm.
- Embeddings are L2-normalised. Agglomerative/DBSCAN/HDBSCAN/OPTICS thresholds are cosine distances; the BIRCH
  threshold is a Euclidean sub-cluster radius on unit vectors (cosine distance = r²/2).
- Density-based algorithms (DBSCAN/HDBSCAN/OPTICS) label some points as noise; each noise point is treated as its own
  singleton event (a sentence that matches nothing else is still an event for triage).
- `select_model.py` prints a NOTE if the selection evidence favours a different algorithm or encoder than the one
  configured (`clustering.algorithm`, `embedding.base_model`).
- Only the primary model's significance tests depend on `embedding.base_model`. After changing it,
  `python scripts/evaluate_fpv.py --stats-only` recomputes `significance.csv` from the saved predictions, without
  the trained models.
- Each model directory stores `lasec_meta.json` with package versions, device, training pairs and loss.
- The decrypted evidence is checked against `data/raw/fpv/decrypted_sha256.json` on every run. The original
  encrypted records and their hashes are published with DroPTC (`raw_sha_hashes.json`).

## Data provenance

The catalogue is the sentence-level DroPTC dataset (Silalahi et al., 2026), built from DJI app messages in the DroNER dataset (Silalahi et al., 2023) and the AirData notification list (https://app.airdata.com/wiki/Notifications).

<!-- ## Citation

```bibtex
@article{silalahi2026lasec, title={LASeC: ...}, note={under revision, Forensic Science International: Digital Investigation}}
``` -->

## License

Apache License 2.0 (see `LICENSE`).
