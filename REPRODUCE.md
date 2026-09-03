# Reproduction Guide

This guide reproduces the released TEST-set metrics from the GitHub repository and the
companion data archive. Commands assume the following layout:

```text
work/
├── NeuroBioR-Reproducibility/
└── NeuroBioR_Aligned_TEST_Data_v1.1/
```

## 1. Create a clean environment

```bash
cd work/NeuroBioR-Reproducibility
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
```

Python 3.12 is the tested runtime. Exact direct dependencies are pinned in
`requirements.txt`.

## 2. Validate data and archived outputs

```bash
python scripts/validate_release.py \
  --data-root ../NeuroBioR_Aligned_TEST_Data_v1.1/data \
  --output reproduced/validation_report.json
```

Expected result: 75 retained runs are present; all available APAVA and ADFTD tensors
load as finite `float32` arrays; all 50 APAVA/ADFTD probability matrices reproduce the
stored six metrics with zero numerical difference at the recorded precision.

## 3. Re-score all redistributable TEST outputs

```bash
python scripts/score_archived_runs.py \
  --data-root ../NeuroBioR_Aligned_TEST_Data_v1.1/data \
  --output-root reproduced/scored_runs
```

Outputs:

- `reproduced/scored_runs/per_run_metrics.csv`
- `reproduced/scored_runs/mean_sample_sd.csv`
- `reproduced/scored_runs/summary.json`

TDBRAIN is reported as unavailable unless an authorized aligned pack is added to the
data root. Its archived metrics remain visible in `results/tables/`.

## 4. Replay one synthesized predictor

The following smoke test re-executes one retained Qwen predictor on every APAVA TEST
window and compares its hard labels with the archived probability matrix:

```bash
python scripts/replay_frozen_predictors.py \
  --data-root ../NeuroBioR_Aligned_TEST_Data_v1.1/data \
  --dataset APAVA \
  --model-tag qwen3_235b_a22b_thinking_2507_v1 \
  --request-index 41 \
  --output reproduced/qwen_apava_41.json
```

The same command accepts any retained model directory, dataset, and request index
41--45 when the corresponding aligned data are available.

## 5. Locate the paper tables

```text
results/tables/apava_submission_table.csv
results/tables/tdbrain_submission_table.csv
results/tables/adftd_submission_table.csv
results/tables/agent_llm_per_seed_metrics.csv
results/tables/agent_llm_mean_sample_sd.csv
```

The three submission tables combine the six published supervised reference rows with
the five complete Agent+LLM rows. Values are percentages reported as mean ± sample SD.

## 6. What this procedure does and does not reproduce

- It exactly re-scores retained Agent+LLM outputs on the same fixed TEST samples.
- It replays retained deterministic predictors without another paid API request.
- It does not re-contact a commercial LLM endpoint by default.
- It does not retrain the supervised baselines; the pinned Medformer source snapshot
  and original run scripts are retained for that separate, GPU-intensive workflow.
- It cannot redistribute TDBRAIN signals; authorized access is required for signal-level
  replay of that task.

