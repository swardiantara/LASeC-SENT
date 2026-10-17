#!/usr/bin/env bash
# Reproduce all results, tables and figures.  Usage: bash run_all.sh [--smoke]
# Requires frozen labels in data/labels/ (annotation/GUIDELINE.md, Section 7) and, for the LLM baselines,
# OPENROUTER_API_KEY (cached responses in results/llm/cache are reused without a key).
set -euo pipefail
CFG=configs/default.yaml
SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then CFG=configs/smoke.yaml; SMOKE="--smoke"; fi
PY=${PYTHON:-python}

$PY scripts/prepare_data.py --config $CFG                 # 1. evidence integrity + sentence data
$PY scripts/select_model.py --config $CFG                # 2a. model selection: clustering algorithm, then encoder
$PY scripts/tune_baselines.py --config $CFG               # 2b. Drain3 / pre-trained thresholds on the catalogue

# 3-4. cross-validation (threshold selection) and final training, for every base model
for M in $($PY -c "import yaml;print(' '.join(yaml.safe_load(open('$CFG'))['embedding']['base_model_variants']))"); do
  if [[ -n "$SMOKE" && "$M" != *MiniLM-L6* ]]; then continue; fi
  $PY scripts/cross_validate.py --config $CFG --base-model $M $SMOKE
  $PY scripts/train_final.py --config $CFG --base-model $M $SMOKE
done

# 5. LLM baselines via OpenRouter (every enabled model, zero- and few-shot)
if [[ -z "$SMOKE" ]]; then
  $PY scripts/run_llm.py --config $CFG || echo "LLM step incomplete - see messages above"
fi

$PY scripts/evaluate_fpv.py --config $CFG                 # 6. held-out evaluation, significance, sensitivity
$PY scripts/case_study.py --config $CFG                   # 7. forensic case study per flight
$PY scripts/repeatability.py --config $CFG                # 8. input-order repeatability per clustering algorithm
$PY scripts/compute_iaa.py --config $CFG                  # 9. inter-annotator agreement (when available)
$PY scripts/make_tables.py --config $CFG                  # 10. LaTeX/CSV tables + summary.md
$PY scripts/make_figures.py --config $CFG                 # 11. figures (PDF + PNG)
