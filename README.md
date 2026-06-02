# Synthetic Generation Experiments

Research codebase for **domain-routing collision experiments** on [RouterArena](https://huggingface.co/datasets/RouteWorks/RouterArena). We train lightweight classifiers (ModernBERT linear probe, SetFit) on real and LLM-generated multiple-choice items, then measure whether synthetic augmentation helps or hurts on held-out splits.

Full protocol definitions: [`EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md).

## Research question

> When synthetic MCQ items are generated to mimic RouterArena *Domain* labels, do they improve routing accuracy—or create distributional collisions that degrade generalization?

We isolate this under fixed seeds, frozen synthetics where noted, and paired **eval_1000** (primary) / **holdout_500** (pre/post augmentation) reporting.

## Executed experiments (May–Jun 2026)

All runs below completed locally; aggregated metrics live in [`analysis/leaderboard.csv`](analysis/leaderboard.csv). Raw artifacts are under `results/` (gitignored; regenerate via CLI).

### Baseline track — ModernBERT + 2k real (`train_2k`)

| ID | Status | Primary eval_1000 (acc / macro-F1) | Notes |
|----|--------|-------------------------------------|-------|
| **exp01** | ✅ | **0.794 / 0.794** | Floor: 2k real only |
| **exp02** | ✅ | 0.789 / 0.789 | Failure-driven synth (+104); holdout post **0.788** vs pre 0.792 |
| **exp03** | ✅ | 0.787 / 0.786 | 3 synth iterations (+315); holdout post **0.826** but primary eval drops |
| **exp04** | ✅ | 0.754 / 0.752 | Rare-class downsample within train_2k (+201 synth) |
| **exp05** | ✅ (legacy) | 0.793 / 0.793 | Donor-pool real-copy ablation |
| **exp06** | ✅ | 0.688 / 0.690 | train_500 + 490 proportional synth |
| **exp07** | ✅ | 0.791 / 0.792 | Confusion-pair boundary synth (+58) |
| **exp08** | ✅ (legacy) | 0.790 / 0.790 | Generate + YES-only validation (+104) |
| **exp09** | ✅ | 0.717 / 0.717 | train_500 real floor (pairs with exp06) |
| **exp10** | ✅ | 0.785 / 0.785 | Frozen exp02 synth; context moved into question |
| **exp11** | ✅ | 0.785 / 0.785 | Frozen exp02 synth; context dropped |

**Takeaways (baseline):** Synthetic augmentation rarely beats the exp01 floor on **eval_1000**. exp03 improves holdout sharply (+3.4 pp) while hurting primary eval (−0.7 pp)—a train/eval collision signal. Low-data regimes (exp06/exp09) sit ~68–72%. Format ablations (exp10/11) do not recover exp02 gains.

### SetFit track — 100 real + up to 30 synth

| ID | Status | eval_1000 (acc / macro-F1) | Holdout Δ (post − pre) |
|----|--------|----------------------------|-------------------------|
| **setfit01** | ✅ | 0.577 / 0.583 | 0.000 |
| **setfit02** | ✅ | 0.574 / 0.578 | +0.004 |
| **setfit03** | ✅ | 0.541 / 0.541 | −0.016 |
| **setfit04** | ✅ | 0.587 / 0.593 | +0.008 |
| **setfit05** | ✅ | 0.539 / 0.542 | −0.028 (verification v2; 16 accepted) |
| **setfit06** | ✅ | 0.557 / 0.565 | +0.010 |
| **setfit07** | ✅ | 0.569 / 0.573 | +0.010 |
| **setfit08** | ✅ | 0.556 / 0.563 | +0.004 (Gemini flash-lite) |
| **setfit09** | ✅ | **0.634 / 0.578**† | **+0.042** (OpenTDB-only pool) |
| **setfit10** | ✅ | 0.546 / 0.549 | −0.020 (acceptance band 0.4–0.6) |

† setfit09 primary eval on OpenTDB-filtered subset (n=281).

**Takeaways (SetFit):** With only 100 labeled items, all variants cluster ~54–59% on full eval_1000. **setfit09** (OpenTDB-only, low-confidence refs) is the clear outlier—holdout +4.2 pp and best subset accuracy. Cross-run prediction agreement (Fleiss κ ≈ 0.56 across 10 SetFit runs) is reported under `analysis/plots/agreement/setfit/`.

## Setup

```bash
git clone https://github.com/dmsavkov/SyntheticGenerationExperiments.git
cd SyntheticGenerationExperiments
pip install -e ".[llm]"              # ModernBERT + Ollama experiments
pip install -e ".[setfit]"           # SetFit track
python routers/scripts/build_splits.py
```

Ollama (local synth): `ollama pull llama3:8b-instruct-q4_K_M`  
Google GenAI (setfit08): set `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

## Reproduce a run

```bash
python routers/main.py exp01
python routers/main.py exp02          # requires Ollama
python routers/main.py setfit01       # pip install -e ".[setfit]"
```

Each saved run writes one timestamp folder:

```text
results/<topic>/<experiment>/<timestamp>/
  metrics.json              # eval_1000 + holdout_metrics + generation_stats
  config.json
  predictions_*.json
  confusion.png
  synthetic_samples.json    # full generated text (when applicable)
```

Regenerate leaderboard: `python analysis/temporal_baseline_leaderboard.py`  
Agreement plots: `python analysis/experiment_agreement.py --topic setfit`

## Repository layout

| Path | Role |
|------|------|
| `routers/experiments/` | Experiment implementations (exp01–11, setfit01–10) |
| `routers/synthetic/` | LLM generation, parsing, verification |
| `data/splits/` | Fixed seed splits (rebuild via scripts) |
| `data/synthetic/` | Frozen synthetic corpora (e.g. exp02 log) |
| `analysis/` | Leaderboard aggregation + agreement analysis |
| `notebooks/` | Colab / manual lab notebooks |
| `tests/` | Unit tests for parsing, batching, splits |

## Notebooks

- `notebooks/domain_synthetic.ipynb` — baseline experiment walkthrough
- `notebooks/colab_domain_synthetic.ipynb` — Colab variant (with outputs)
- `notebooks/colab_setfit_experiments.ipynb` — SetFit group runner
- `notebooks/playground_manual_lab.ipynb` — manual SetFit lab (ArcMMLU / MedMCQA / MMLU)

Use `pip install -e ".[llm]"` or `".[playground]"` in notebooks—not `uv sync` (Colab kernel compatibility).
