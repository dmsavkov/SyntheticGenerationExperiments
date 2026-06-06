# Synthetic HardNLP — Experiment Spec (v0.1)

Domain-only collision tests on RouterArena. Forked patterns from `../newest/EXPERIMENT_SPEC.md`.

Topic group: **`baseline`** → `results/baseline/<experiment>/<timestamp>/`

## Splits

| File | Contents |
|------|----------|
| `data/splits/seed42.json` | `eval_1000`, `train_pool`, … |
| `data/splits/domain_synth_seed42.json` | `train_2k`, `holdout_500`, `train_500`, **`train_100`**, `donor_pool` |

Rebuild: `python routers/scripts/build_splits.py`

## Experiments

| ID | Description |
|----|-------------|
| exp01 | ModernBERT baseline, train 2k |
| exp02 | Failure-driven synthetic, 1 iter |
| exp03 | 3 synthetic iterations |
| exp04 | Within **train_2k**: one Domain at 10%, synth-restore removed 90% from kept refs |
| exp05 | Donor-pool real copies (ablation) |
| exp06 | Train 500 + 500 proportional synthetic |
| exp07 | Confusion-pair boundary synthesis |
| exp08 | Generate + validate YES-only + probe scores |
| exp09 | ModernBERT baseline, **train_500** (exp06 real-data floor) |
| exp10 | train_2k + frozen exp02 synth; **context → question** |
| exp11 | train_2k + frozen exp02 synth; **context dropped** |

Frozen synthetics: `data/synthetic/baseline_exp02.json` (exp02 generation log or `synthetic_samples.json` list).
Prompt layout stays embedder: `Context: … | Question: … | Options: …`.

## SetFit experiments (topic `setfit`)

Install: `pip install -e ".[setfit]"`

Results: `results/setfit/<experiment>/<timestamp>/`

| ID | Description |
|----|-------------|
| setfit01 | **train_100 only**, no synth — eval_1000 floor |
| setfit02 | train_100 + 30 synth (uncertainty refs, Ollama) |
| setfit03 | top 10% confidence correct refs |
| setfit04 | bottom 10% confidence correct refs |
| setfit05 | uncertainty synth + verification v2 (fastembed + SetFit prob filter) |
| setfit06 | diversity triplet prompts (3 items/call) |
| setfit07 | hard + diverse rewrite (15+15 refs) |
| setfit08 | uncertainty synth via **Google GenAI gemini-3.1-flash-lite** |
| setfit09 | **OpenTDB-only** pool, bottom 10% confidence correct refs |
| setfit10 | Like setfit05 (verification v2), but reference + verification prob band **[0.4, 0.6]** |

Protocol: 100 real + up to 30 **appended** synthetics; max **3 items per LLM request**; holdout_500 pre/post; primary eval eval_1000.

OpenTDB splits: `data/splits/opentdb_setfit_seed42.json` — rebuild via `python routers/scripts/build_splits.py`.

Google smoke: `python routers/scripts/smoke_google_genai.py` (requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`).

Colab runner: `notebooks/colab_setfit_experiments.ipynb`

## Ensemble / combination (OpenTDB, Gemini 3.1 flash-lite)

Results: `results/ensemble/` and `results/combination/`

Splits: `data/splits/opentdb_ensemble_seed42.json` (100/100/rest) and `data/splits/opentdb_combination_binary_seed42.json` (20/80/rest, CS vs Technology)

| ID | Description |
|----|-------------|
| ensemble01 / combination01 | Multiclass SetFit baseline (no synth) |
| ensemble02 | General + ≤2 OVR experts (min support **8**), 2 mutation iters; expert override if max expert P≥**0.85** else general |
| ensemble03 | Full OVR panel; **confused** if ≥2 classes scream P≥**0.85**; pred = argmax OVR prob |
| combination02 | Hard-negative pair synth (2 per holdout failure) |
| combination03 | 3-iter cascading chat; eval iter1+2+3 and iter2+3 |
| combination04 | Failure-driven label_failure synth |
| combination05 | Binary CS vs Technology cascade (20 train, 80 holdout); Gemini thinking **high** |

Confused routing: mark `confused` when **≥2 distinct labels** have prob ≥ **0.85** (general and/or experts). Ensemble02 final pred: expert class if max expert P≥0.85, else general pred. Also report `metrics_excluding_confused`.

```bash
python routers/scripts/build_splits.py
python routers/main.py ensemble01
```

## Smart experiments (OpenTDB, Gemini; topic `smart`)

Results: `results/smart/<experiment>/<timestamp>/`

Shared protocol: `HARD_NEGATIVE_FEWSHOT` (same as combination02), thinking **high** except smart06/07 (**no reasoning**, temp=1, top_p=0.95). Reference buckets use **bottom/top 10%** `p_gold` among correct (setfit04-style), not absolute thresholds. Async LLM: max **15** in flight, **25** parse retries.

| ID | Description |
|----|-------------|
| smart01 | Contrastive 3-step: draft → 5-bullet critique → refine |
| smart02 | Binary CS vs Technology hard-negative pairs |
| smart03 | Mislabeled + bottom-percentile correct → hard negatives |
| smart04 | 5-fold CV on train_100 for failures; **no holdout**; train 100+synth |
| smart05 | SKIP ambiguous gold else hard-negative pair |
| smart06 | Per mislabeled ref: 3× parallel pair gen → 1 pair judge → **2 items** (like comb02 volume) |
| smart07 | Per ref in pool: diversity paths A/B/C → judge picks **1** |
| smart08 | Top/bottom/mislabeled buckets → **hard-negative pair (2 items) per ref** |
| smart09 | **3-iter** smart03 cycle: score holdout → synth → retrain; **holdout+test metrics each iter** |

```bash
python routers/main.py smart02
```

## Playground (manual lab)

Install: `pip install -e ".[playground]"`

- Module: `routers/playground/manual_lab.py`
- Notebook: `notebooks/playground_manual_lab.ipynb`
- Splits: `data/splits/playground_arc_med_mmlu_seed42.json` — rebuild: `python routers/scripts/build_playground_splits.py`
- Default sources: ArcMMLU, MedMCQA, MMLU_management (`CFG["source_preset"]`)
- Default **2-class** subset; 50 train / 100 eval per class (stratified by source dataset)

Legacy script `notebooks/playground_setfit_script.py` is deprecated.

## Artifacts (per timestamp folder)

| File | Contents |
|------|----------|
| `metrics.json` | Primary **eval_1000** metrics + `holdout_metrics` (pre/post) + `generation_stats` |
| `config.json` | Full run configuration |
| `predictions_*.json` | Primary eval predictions (prompts truncated to 2000 chars) |
| `confusion.png` | Primary eval confusion matrix |
| `synthetic_samples.json` | **Full** `context`, `question`, `options`, `domain`, `prompt` |
| `synthetic_validation.json` | exp08 only |
| `augmentation_samples.json` | exp05 only |

Ollama batches that fail JSON parse after 3 retries are **skipped** (logged in `generation_stats.skipped_batches`).

## Ollama / Colab

- `pip install -e ".[llm]"` in notebooks (not uv)
- Start server in background via `ensure_ollama_server()` before generation
