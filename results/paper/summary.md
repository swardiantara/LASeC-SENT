# LASeC results summary

Primary model: LASeC (all-MiniLM-L6-v2). Generated from /media/ww-pc/Data-HDD/swardi/LASeC-SENT/results.

### Datasets. The catalogue uses the manual DroPTC sentence split; FPV logs are segmented automatically (';' then PySBD).

|  | Catalogue | FPV |
|---|---|---|
| Source | DroPTC catalogue (training) | DJI FPV flights (held-out test) |
| Messages / log cells | 585 | 658 |
| Atomic app messages | -- | 1466 |
| Sentence occurrences | 1035 | 2158 |
| Unique sentences | 783 | 78 |
|   of which event sentences | 553 | 63 |
| Event groups (gold) | 318 | 40 |
|   singleton event groups | 209 | 34 |
| Non-event groups (instruction/mitigation/possible/info) | 111 | 10 |
| Expert key events | -- | 35 |
| Flights | -- | 7 |



### Sentence types (unique sentences) under the annotation guideline.

| Sentence type | Catalogue | FPV |
|---|---|---|
| event | 538 | 63 |
| compound | 15 | 0 |
| instruction | 50 | 6 |
| mitigation | 127 | 4 |
| possible | 17 | 0 |
| info | 36 | 5 |



### FPV flights used as the held-out test set.

| Flight | Log cells | Sentences | Unique | Gold groups | key events |
|---|---|---|---|---|---|
| 1 | 22 | 45 | 17 | 17 | 3 |
| 2 | 133 | 392 | 35 | 30 | 5 |
| 3 | 186 | 593 | 37 | 30 | 5 |
| 4 | 34 | 126 | 32 | 29 | 6 |
| 5 | 75 | 186 | 30 | 27 | 5 |
| 6 | 58 | 276 | 36 | 25 | 5 |
| 7 | 150 | 540 | 39 | 32 | 6 |



- PySBD (after ';' split) reproduces the manual DroPTC sentence boundaries for 546/568 catalogue messages (96.1%); 17 messages skipped because the manual split omits text.

### iaa: not available (second-annotator labels not imported yet)

### Stage A of model selection: clustering algorithms on the catalogue with the pre-trained all-mpnet-base-v2 encoder (no fine-tuning). Thresholds: cosine distance (BIRCH: Euclidean sub-cluster radius on unit vectors).

| Algorithm | Peak GA | Best threshold | Mean GA over grid | Pair F1 at peak | Groups at peak |
|---|---|---|---|---|---|
| Agglomerative | 0.460 | 0.250 | 0.190 | 0.693 | 477 |
| DBSCAN | 0.453 | 0.200 | 0.146 | 0.647 | 490 |
| BIRCH | 0.439 | 0.300 | 0.233 | 0.618 | 590 |
| OPTICS | 0.415 | 0.150 | 0.349 | 0.353 | 597 |
| HDBSCAN | 0.331 | 0.050 | 0.111 | 0.583 | 390 |



### Stage B of model selection: pre-trained encoders with Agglomerative clustering on the catalogue; throughput measured on the catalogue sentences.

| Encoder (pre-trained) | Params (M) | Dim | Peak GA | Best threshold | Mean GA over grid | Encoding (sentences/s) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22.7 | 384 | 0.469 | 0.150 | 0.192 | 18,910 |
| all-MiniLM-L12-v2 | 33.4 | 384 | 0.465 | 0.150 | 0.185 | 13,982 |
| all-distilroberta-v1 | 82.1 | 768 | 0.492 | 0.250 | 0.207 | 12,685 |
| all-mpnet-base-v2 | 109.5 | 768 | 0.460 | 0.250 | 0.190 | 7,545 |



- Model selection: Stage A best algorithm = agglomerative (configured agglomerative); Stage B most accurate encoder = all-distilroberta-v1, fastest within 0.02 GA = all-distilroberta-v1 (configured all-MiniLM-L6-v2).

### 5-fold cross-validation on the catalogue (mean ± sd over folds). Main clustering: Agglomerative; t is the threshold selected by CV; 'seen/novel event' splits held-out sentences by whether their event occurs in the training folds.

| Base model | Params (M) | t (pre-trained) | GA pre-trained | t (LASeC) | GA LASeC | Pair F1 | GA seen event | GA novel event | LASeC with ablation clustering |
|---|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22.7 | 0.170 | 0.804 ± 0.043 | 0.230 | 0.872 ± 0.027 | 0.828 | 0.841 | 0.916 | BIRCH 0.853 (t=0.34) |
| all-MiniLM-L12-v2 | 33.4 | 0.160 | 0.808 ± 0.043 | 0.170 | 0.871 ± 0.033 | 0.822 | 0.830 | 0.933 | BIRCH 0.858 (t=0.29) |
| all-distilroberta-v1 | 82.1 | 0.230 | 0.793 ± 0.034 | 0.180 | 0.875 ± 0.027 | 0.817 | 0.832 | 0.937 | BIRCH 0.867 (t=0.33) |
| all-mpnet-base-v2 | 109.5 | 0.190 | 0.799 ± 0.040 | 0.200 | 0.877 ± 0.016 | 0.833 | 0.837 | 0.934 | BIRCH 0.874 (t=0.30) |



### Held-out FPV results (all sentences). GA = group accuracy over unique sentences; occurrence-weighted GA weights each sentence by its frequency in the logs. LLM rows: mean ± sd over repeated runs.

| Method | Family | Groups | GA | 95% CI | GA (occ.-weighted) | Pair P | Pair R | Over-merged groups | GA unseen | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact dedup | String de-duplication | 78.0 | 0.538 | [0.42, 0.65] | 0.719 | 1.000 | 0.000 | 0.0 | 0.523 | 0.00 |
| Regex-masked dedup | String de-duplication | 58.0 | 0.846 | [0.76, 0.92] | 0.736 | 1.000 | 0.833 | 0.0 | 0.862 | 0.00 |
| Drain3 | Log parser | 56.0 | 0.821 | [0.73, 0.90] | 0.732 | 0.987 | 0.844 | 1.0 | 0.831 | 0.01 |
| Pre-trained all-MiniLM-L6-v2 + Agglomerative | Pre-trained embedding + clustering | 53.0 | 0.782 | [0.69, 0.87] | 0.766 | 0.963 | 0.867 | 3.0 | 0.754 | 0.01 |
| LASeC (all-MiniLM-L6-v2) | LASeC (fine-tuned embedding + clustering) | 46.0 | 0.808 | [0.72, 0.88] | 0.798 | 0.912 | 0.922 | 4.0 | 0.800 | 0.01 |
| LASeC (all-MiniLM-L6-v2) + BIRCH | LASeC (fine-tuned embedding + clustering) | 47.0 | 0.821 | [0.73, 0.90] | 0.804 | 0.933 | 0.922 | 4.0 | 0.815 | 0.02 |
| claude-opus-5.5 (zero-shot) | LLM zero-shot | 51.3 | 0.902 ± 0.045 |  | 0.842 | 0.989 | 0.963 | 1.0 | 0.897 | 29.35 |
| gemini-3.8-flash (zero-shot) | LLM zero-shot | 48.7 | 0.949 ± 0.000 |  | 0.905 | 0.982 | 0.996 | 1.7 | 0.954 | 131.56 |
| gpt-5.6-luna (zero-shot) | LLM zero-shot | 49.7 | 0.940 ± 0.015 |  | 0.963 | 0.985 | 0.989 | 1.3 | 0.944 | 46.83 |
| claude-opus-5.5 (few-shot) | LLM few-shot | 51.7 | 0.940 ± 0.015 |  | 0.920 | 0.996 | 0.978 | 0.3 | 0.933 | 30.11 |
| gemini-3.8-flash (few-shot) | LLM few-shot | 48.3 | 0.957 ± 0.015 |  | 0.907 | 0.982 | 1.000 | 1.7 | 0.964 | 146.04 |
| gpt-5.6-luna (few-shot) | LLM few-shot | 48.7 | 0.915 ± 0.015 |  | 0.956 | 0.974 | 0.989 | 2.3 | 0.913 | 46.12 |

95% CI: bootstrap over unique sentences. Time: grouping time for the FPV sentences (LLM: API latency). Pair P/R: pairwise precision (penalises over-merging) and recall.

### Held-out FPV results (event sentences). GA = group accuracy over unique sentences; occurrence-weighted GA weights each sentence by its frequency in the logs. LLM rows: mean ± sd over repeated runs.

| Method | Family | Groups | GA | 95% CI | GA (occ.-weighted) | Pair P | Pair R | Over-merged groups | GA unseen | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact dedup | String de-duplication | 63.0 | 0.540 | [0.41, 0.67] | 0.742 | 1.000 | 0.000 | 0.0 | 0.529 | 0.00 |
| Regex-masked dedup | String de-duplication | 47.0 | 0.841 | [0.75, 0.92] | 0.759 | 1.000 | 0.823 | 0.0 | 0.863 | 0.00 |
| Drain3 | Log parser | 45.0 | 0.810 | [0.71, 0.89] | 0.754 | 0.985 | 0.835 | 1.0 | 0.824 | 0.01 |
| Pre-trained all-MiniLM-L6-v2 + Agglomerative | Pre-trained embedding + clustering | 42.0 | 0.762 | [0.65, 0.86] | 0.795 | 0.958 | 0.861 | 3.0 | 0.725 | 0.01 |
| LASeC (all-MiniLM-L6-v2) | LASeC (fine-tuned embedding + clustering) | 38.0 | 0.810 | [0.71, 0.90] | 0.828 | 0.935 | 0.911 | 3.0 | 0.804 | 0.01 |
| LASeC (all-MiniLM-L6-v2) + BIRCH | LASeC (fine-tuned embedding + clustering) | 38.0 | 0.810 | [0.71, 0.90] | 0.828 | 0.935 | 0.911 | 3.0 | 0.804 | 0.02 |
| claude-opus-5.5 (zero-shot) | LLM zero-shot | 40.3 | 0.910 ± 0.056 |  | 0.886 | 0.987 | 0.970 | 1.0 | 0.908 | 29.35 |
| gemini-3.8-flash (zero-shot) | LLM zero-shot | 38.7 | 0.937 ± 0.000 |  | 0.886 | 0.979 | 0.996 | 1.7 | 0.941 | 131.56 |
| gpt-5.6-luna (zero-shot) | LLM zero-shot | 40.0 | 0.937 ± 0.000 |  | 0.959 | 0.987 | 0.987 | 1.0 | 0.941 | 46.83 |
| claude-opus-5.5 (few-shot) | LLM few-shot | 40.7 | 0.958 ± 0.018 |  | 0.980 | 0.996 | 0.987 | 0.3 | 0.954 | 30.11 |
| gemini-3.8-flash (few-shot) | LLM few-shot | 38.3 | 0.947 ± 0.018 |  | 0.889 | 0.979 | 1.000 | 1.7 | 0.954 | 146.04 |
| gpt-5.6-luna (few-shot) | LLM few-shot | 39.7 | 0.926 ± 0.018 |  | 0.957 | 0.983 | 0.987 | 1.3 | 0.928 | 46.12 |

95% CI: bootstrap over unique sentences. Time: grouping time for the FPV sentences (LLM: API latency). Pair P/R: pairwise precision (penalises over-merging) and recall.

### LLM baselines via OpenRouter: run-to-run variation, latency and cost per run (all sentences).

| Model (mode) | Runs | GA | Groups | Latency (s) | Cost (USD/run) |
|---|---|---|---|---|---|
| claude-opus-5.5 (few-shot) | 3 | 0.940 ± 0.015 | 51.7 | 30.1 | 0.0653 |
| claude-opus-5.5 (zero-shot) | 3 | 0.902 ± 0.045 | 51.3 | 29.3 | 0.0577 |
| gemini-3.8-flash (few-shot) | 3 | 0.957 ± 0.015 | 48.3 | 146.0 | 0.1012 |
| gemini-3.8-flash (zero-shot) | 3 | 0.949 ± 0.000 | 48.7 | 131.6 | 0.0914 |
| gpt-5.6-luna (few-shot) | 3 | 0.915 ± 0.015 | 48.7 | 46.1 | 0.0078 |
| gpt-5.6-luna (zero-shot) | 3 | 0.940 ± 0.015 | 49.7 | 46.8 | 0.0076 |



### Paired comparison of LASeC (all-MiniLM-L6-v2) with each baseline on FPV (all sentences). LLM baselines: pre-specified first run (run 0). Holm correction within the family of comparisons.

| Baseline | ΔGA (LASeC − baseline) | 95% CI | LASeC-only / baseline-only correct | p (McNemar, Holm) | p (bootstrap, Holm) | p (weighted bootstrap, Holm) |
|---|---|---|---|---|---|---|
| Exact dedup | +0.269 | [0.13, 0.41] | 31 / 10 | 0.019 | 0.013 | 1.000 |
| Regex-masked dedup | -0.038 | [-0.14, 0.06] | 7 / 10 | 1.000 | 1.000 | 1.000 |
| Drain3 | -0.013 | [-0.12, 0.09] | 7 / 8 | 1.000 | 1.000 | 1.000 |
| Pre-trained all-MiniLM-L6-v2 + Agglomerative | +0.026 | [-0.06, 0.12] | 8 / 6 | 1.000 | 1.000 | 1.000 |
| Pre-trained all-MiniLM-L12-v2 + Agglomerative | +0.000 | [-0.09, 0.09] | 6 / 6 | 1.000 | 1.000 | 1.000 |
| Pre-trained all-mpnet-base-v2 + Agglomerative | +0.000 | [-0.08, 0.06] | 4 / 4 | 1.000 | 1.000 | 1.000 |
| Pre-trained all-distilroberta-v1 + Agglomerative | +0.090 | [0.00, 0.18] | 11 / 4 | 0.829 | 0.567 | 0.760 |
| LLM claude-opus-5.5_few run0 | -0.141 | [-0.24, -0.04] | 4 / 15 | 0.231 | 0.120 | 1.000 |
| LLM claude-opus-5.5_zero run0 | -0.090 | [-0.21, 0.03] | 8 / 15 | 1.000 | 1.000 | 1.000 |
| LLM gemini-3.8-flash_few run0 | -0.141 | [-0.24, -0.04] | 4 / 15 | 0.231 | 0.126 | 1.000 |
| LLM gemini-3.8-flash_zero run0 | -0.141 | [-0.24, -0.04] | 4 / 15 | 0.231 | 0.120 | 0.273 |
| LLM gpt-5.6-luna_few run0 | -0.115 | [-0.22, -0.01] | 4 / 13 | 0.392 | 0.280 | 0.341 |
| LLM gpt-5.6-luna_zero run0 | -0.141 | [-0.24, -0.04] | 4 / 15 | 0.231 | 0.120 | 0.273 |



### Paired comparison of LASeC (all-MiniLM-L6-v2) with each baseline on FPV (event sentences). LLM baselines: pre-specified first run (run 0). Holm correction within the family of comparisons.

| Baseline | ΔGA (LASeC − baseline) | 95% CI | LASeC-only / baseline-only correct | p (McNemar, Holm) | p (bootstrap, Holm) | p (weighted bootstrap, Holm) |
|---|---|---|---|---|---|---|
| Exact dedup | +0.270 | [0.11, 0.43] | 24 / 7 | 0.043 | <0.001 | 1.000 |
| Regex-masked dedup | -0.032 | [-0.14, 0.08] | 5 / 7 | 1.000 | 1.000 | 1.000 |
| Drain3 | +0.000 | [-0.10, 0.10] | 5 / 5 | 1.000 | 1.000 | 1.000 |
| Pre-trained all-MiniLM-L6-v2 + Agglomerative | +0.048 | [-0.05, 0.14] | 6 / 3 | 1.000 | 1.000 | 1.000 |
| Pre-trained all-MiniLM-L12-v2 + Agglomerative | +0.016 | [-0.06, 0.10] | 4 / 3 | 1.000 | 1.000 | 1.000 |
| Pre-trained all-mpnet-base-v2 + Agglomerative | +0.016 | [-0.05, 0.08] | 3 / 2 | 1.000 | 1.000 | 1.000 |
| Pre-trained all-distilroberta-v1 + Agglomerative | +0.159 | [0.06, 0.27] | 11 / 1 | 0.076 | 0.048 | <0.001 |
| LLM claude-opus-5.5_few run0 | -0.159 | [-0.27, -0.05] | 2 / 12 | 0.142 | 0.055 | 0.036 |
| LLM claude-opus-5.5_zero run0 | -0.095 | [-0.22, 0.05] | 6 / 12 | 1.000 | 1.000 | 1.000 |
| LLM gemini-3.8-flash_few run0 | -0.127 | [-0.25, 0.00] | 4 / 12 | 0.768 | 0.420 | 1.000 |
| LLM gemini-3.8-flash_zero run0 | -0.127 | [-0.24, -0.02] | 4 / 12 | 0.768 | 0.420 | 0.836 |
| LLM gpt-5.6-luna_few run0 | -0.127 | [-0.24, -0.02] | 4 / 12 | 0.768 | 0.420 | 0.836 |
| LLM gpt-5.6-luna_zero run0 | -0.127 | [-0.24, -0.02] | 4 / 12 | 0.768 | 0.420 | 0.836 |



### Effect of the base sentence encoder on FPV (each pre-trained model with the threshold tuned on the catalogue; each LASeC model with the CV-selected threshold).

| Subset | Base model | Params (M) | GA pre-trained | GA LASeC | Δ fine-tuning | GA weighted | Over-merged | Time (s) |
|---|---|---|---|---|---|---|---|---|
| all sentences | all-MiniLM-L6-v2 | 22.7 | 0.782 | 0.808 | +0.026 | 0.798 | 4 | 0.01 |
| all sentences | all-MiniLM-L12-v2 | 33.4 | 0.808 | 0.821 | +0.013 | 0.728 | 3 | 0.01 |
| all sentences | all-distilroberta-v1 | 82.1 | 0.718 | 0.859 | +0.141 | 0.811 | 3 | 0.01 |
| all sentences | all-mpnet-base-v2 | 109.5 | 0.808 | 0.833 | +0.026 | 0.807 | 4 | 0.02 |
| event sentences | all-MiniLM-L6-v2 | 22.7 | 0.762 | 0.810 | +0.048 | 0.828 | 3 | 0.01 |
| event sentences | all-MiniLM-L12-v2 | 33.4 | 0.794 | 0.810 | +0.016 | 0.737 | 2 | 0.01 |
| event sentences | all-distilroberta-v1 | 82.1 | 0.651 | 0.857 | +0.206 | 0.837 | 2 | 0.01 |
| event sentences | all-mpnet-base-v2 | 109.5 | 0.794 | 0.825 | +0.032 | 0.832 | 3 | 0.02 |



### Clustering ablation on FPV with the fine-tuned all-MiniLM-L6-v2 encoder (each algorithm with its own CV-selected threshold).

| Subset | Clustering | Threshold (CV) | Groups | GA | 95% CI | GA weighted | Over-merged | Time (s) |
|---|---|---|---|---|---|---|---|---|
| all sentences | Agglomerative | 0.230 | 46 | 0.808 | [0.72, 0.88] | 0.798 | 4 | 0.01 |
| event sentences | Agglomerative | 0.230 | 38 | 0.810 | [0.71, 0.90] | 0.828 | 3 | 0.01 |
| all sentences | BIRCH | 0.340 | 47 | 0.821 | [0.73, 0.90] | 0.804 | 4 | 0.02 |
| event sentences | BIRCH | 0.340 | 38 | 0.810 | [0.71, 0.90] | 0.828 | 3 | 0.02 |



### Forensic case study with LASeC (all-MiniLM-L6-v2): per-flight triage outcome (all log sentences).

| Flight | Sentences | Unique | Triage events | Workload reduction | Key events recovered | Hidden key events |
|---|---|---|---|---|---|---|
| 1 | 45 | 17 | 16 | 64.4% | 3/3 | [] |
| 2 | 392 | 35 | 30 | 92.3% | 5/5 | [] |
| 3 | 593 | 37 | 30 | 94.9% | 5/5 | [] |
| 4 | 126 | 32 | 28 | 77.8% | 6/6 | [] |
| 5 | 186 | 30 | 27 | 85.5% | 5/5 | [] |
| 6 | 276 | 36 | 22 | 92.0% | 4/5 | ['LOW_BATTERY_RTH'] |
| 7 | 540 | 39 | 32 | 94.1% | 6/6 | [] |



### Triage over the 7 flights: events presented to the investigator (sum over flights), workload reduction (1 − events / log lines) and recall of the expert key events.

| Lines | Method | Family | Log lines | Triage events (sum) | Workload reduction | Key events recovered | Key-event recall |
|---|---|---|---|---|---|---|---|
| all sentences | Drain3 | Log parser | 2158 | 210.0 | 90.3% | 35.0/35 | 100.0% |
| all sentences | Exact dedup | String de-duplication | 2158 | 226.0 | 89.5% | 35.0/35 | 100.0% |
| all sentences | LASeC (all-MiniLM-L6-v2) | LASeC (fine-tuned embedding + clustering) | 2158 | 185.0 | 91.4% | 34.0/35 | 97.1% |
| all sentences | LASeC (all-MiniLM-L6-v2) + BIRCH | LASeC (fine-tuned embedding + clustering) | 2158 | 186.0 | 91.4% | 34.0/35 | 97.1% |
| all sentences | Pre-trained all-MiniLM-L6-v2 + Agglomerative | Pre-trained embedding + clustering | 2158 | 206.0 | 90.5% | 35.0/35 | 100.0% |
| all sentences | Regex-masked dedup | String de-duplication | 2158 | 213.0 | 90.1% | 35.0/35 | 100.0% |
| all sentences | claude-opus-5.5 (few-shot) | LLM few-shot | 2158 | 197.7 | 90.8% | 34.7/35 | 99.0% |
| all sentences | claude-opus-5.5 (zero-shot) | LLM zero-shot | 2158 | 197.7 | 90.8% | 34.3/35 | 98.1% |
| all sentences | gemini-3.8-flash (few-shot) | LLM few-shot | 2158 | 188.3 | 91.3% | 34.0/35 | 97.1% |
| all sentences | gemini-3.8-flash (zero-shot) | LLM zero-shot | 2158 | 189.7 | 91.2% | 34.0/35 | 97.1% |
| all sentences | gpt-5.6-luna (few-shot) | LLM few-shot | 2158 | 192.7 | 91.1% | 33.7/35 | 96.2% |
| all sentences | gpt-5.6-luna (zero-shot) | LLM zero-shot | 2158 | 193.0 | 91.1% | 34.0/35 | 97.1% |
| event sentences | Drain3 | Log parser | 1801 | 174.0 | 90.3% | 35.0/35 | 100.0% |
| event sentences | Exact dedup | String de-duplication | 1801 | 190.0 | 89.5% | 35.0/35 | 100.0% |
| event sentences | LASeC (all-MiniLM-L6-v2) | LASeC (fine-tuned embedding + clustering) | 1801 | 157.0 | 91.3% | 34.0/35 | 97.1% |
| event sentences | LASeC (all-MiniLM-L6-v2) + BIRCH | LASeC (fine-tuned embedding + clustering) | 1801 | 157.0 | 91.3% | 34.0/35 | 97.1% |
| event sentences | Pre-trained all-MiniLM-L6-v2 + Agglomerative | Pre-trained embedding + clustering | 1801 | 170.0 | 90.6% | 35.0/35 | 100.0% |
| event sentences | Regex-masked dedup | String de-duplication | 1801 | 177.0 | 90.2% | 35.0/35 | 100.0% |
| event sentences | claude-opus-5.5 (few-shot) | LLM few-shot | 1801 | 161.7 | 91.0% | 34.7/35 | 99.0% |
| event sentences | claude-opus-5.5 (zero-shot) | LLM zero-shot | 1801 | 161.7 | 91.0% | 34.3/35 | 98.1% |
| event sentences | gemini-3.8-flash (few-shot) | LLM few-shot | 1801 | 156.3 | 91.3% | 34.0/35 | 97.1% |
| event sentences | gemini-3.8-flash (zero-shot) | LLM zero-shot | 1801 | 157.7 | 91.2% | 34.0/35 | 97.1% |
| event sentences | gpt-5.6-luna (few-shot) | LLM few-shot | 1801 | 160.7 | 91.1% | 33.7/35 | 96.2% |
| event sentences | gpt-5.6-luna (zero-shot) | LLM zero-shot | 1801 | 161.0 | 91.1% | 34.0/35 | 97.1% |



### Repeatability of LASeC on FPV under random input orders, per clustering algorithm (reported results use alphabetical order).

| Model | Input orders | GA mean ± sd | GA range | Groups range |
|---|---|---|---|---|
| LASeC (all-MiniLM-L12-v2) | 20 | 0.821 ± 0.000 | 0.821–0.821 | 50–50 |
| LASeC (all-MiniLM-L12-v2) + BIRCH | 20 | 0.802 ± 0.046 | 0.679–0.821 | 50–52 |
| LASeC (all-MiniLM-L6-v2) | 20 | 0.808 ± 0.000 | 0.808–0.808 | 46–46 |
| LASeC (all-MiniLM-L6-v2) + BIRCH | 20 | 0.731 ± 0.050 | 0.692–0.833 | 45–48 |
| LASeC (all-distilroberta-v1) | 20 | 0.859 ± 0.000 | 0.859–0.859 | 48–48 |
| LASeC (all-distilroberta-v1) + BIRCH | 20 | 0.847 ± 0.036 | 0.744–0.859 | 47–48 |
| LASeC (all-mpnet-base-v2) | 20 | 0.833 ± 0.000 | 0.833–0.833 | 48–48 |
| LASeC (all-mpnet-base-v2) + BIRCH | 20 | 0.859 ± 0.000 | 0.859–0.859 | 49–49 |


