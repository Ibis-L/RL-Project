# AutoClassify

**Reinforcement learning for budget-aware automated algorithm selection in classification**

Given a new classification dataset, AutoClassify picks a strong classifier while spending far less
compute than evaluating every classifier. An RL agent, trained across many datasets, makes the
decisions step by step:

* **State:** simple dataset features (size, types, class balance, ...), the scores and costs it has
  observed so far, and the budget left.
* **Action:** evaluate classifier *k* at budget level *l* (Continue = same classifier, more data;
  Switch = another classifier), or **STOP**.
* **Agents:** tabular Q-learning, DQN, Double DQN.

There is no separate meta-learner model: the policy itself learns which classifiers suit which data.

Everything is evaluated offline against a precomputed **performance database**, so RL episodes are
lookups, not re-training.

---

## 1. Repository layout

```
autoclassify/
├── configs/                 smoke.yaml (offline test) · stage1.yaml · stage2.yaml · final.yaml
├── autoclassify/
│   ├── config.py            defaults + YAML loading, output paths
│   ├── data/                sources.py (OpenML / synthetic, cleaning, filters) · splits.py · preprocess.py
│   ├── perfdb/              classifiers.py (the 12) · runner.py (grid of fits, resumable) · store.py
│   ├── meta/                metafeatures.py (dataset features) · context.py (scaling, baseline statistics)
│   ├── env/                 autoclassify_env.py (Gymnasium MDP) · compat.py
│   ├── agents/              mlp.py (NumPy MLP + Adam) · dqn.py (DQN + Double DQN) · qlearning.py
│   ├── baselines/           policies.py (random, fixed RF, exhaustive, average rank,
│   │                        best-on-small-budget, successive halving, RL wrappers)
│   ├── eval/                stats.py (regret, ranks, AURC, Friedman/Nemenyi, Wilcoxon+Holm) · plots.py
│   ├── pipeline.py          workspace loading, training, evaluation, ablations
│   └── cli.py               one command per stage
├── notebooks/run_on_kaggle_or_colab.ipynb
├── tests/                   unit tests + a full synthetic mini-pipeline
├── requirements.txt · pyproject.toml · Makefile
```

## 2. Setup

### Local machine (Python 3.10–3.12 recommended)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                  # ~10 s; should say "passed"
python -m autoclassify.cli all --config configs/smoke.yaml   # offline end-to-end test, ~3-5 min
```

The smoke run uses synthetic datasets and scikit-learn classifiers only, so it works without internet
and without GPU. **Never report smoke results.**

### Kaggle (recommended for the performance database)

1. Put the code on GitHub (private repo is fine) **or** upload the project zip as a Kaggle Dataset.
2. New Notebook → *File → Import Notebook* → `notebooks/run_on_kaggle_or_colab.ipynb`.
3. Settings: Accelerator **GPU T4 x2** (or P100), Internet **On** (requires a phone-verified account).
4. Edit `REPO_URL` (or `UPLOADED`) in cell 1 and `CONFIG` in cell 5, then *Save Version → Save & Run All*.
   The run continues in the background; `/kaggle/working` becomes the notebook output.
5. To continue in a new session, add the previous version's output as an input and set `PREVIOUS`.
   Finished fits are skipped automatically.

### Colab

Open the same notebook, choose **T4 GPU**, run the Google Drive cell first so `results/` lives on
your Drive and survives disconnects.

### Dependencies

| Package | Used for |
| --- | --- |
| numpy, pandas, scipy, scikit-learn, pyyaml, matplotlib, joblib | everything (core) |
| gymnasium | the RL environment interface (a minimal stand-in is used if missing) |
| lightgbm, catboost | gradient-boosting classifiers |
| pytabkit | RealMLP (deep learning, CPU) |
| tabpfn | TabPFN (deep learning, GPU recommended; research-only licence) |
| pymfe | optional extra dataset features (off by default; the 17 hand-written ones are always computed) |
| openml | fallback for listing the OpenML-CC18 suite |
| pytest | tests |

The DQN agents are written in NumPy (no PyTorch needed for RL); torch is installed anyway by
pytabkit/tabpfn. If a classifier package will not install, remove it from `classifiers:` in the
config (sklearn fallbacks `hgb` and `mlp_sklearn` exist). The run refuses to start if a configured
classifier is missing, so the database never mixes different classifier sets.

**TabPFN weights:** downloaded on first use. If the download asks for authentication, follow its
message (typically: accept the licence on the Hugging Face model page and provide a token, e.g. via
Kaggle *Add-ons → Secrets* as `HF_TOKEN`). Check the tabpfn README for the current procedure.

## 3. Datasets: where they come from

All datasets come from **[OpenML](https://www.openml.org)**, downloaded automatically by
`scikit-learn.fetch_openml` and cached in `data/openml_cache/`. Nothing needs to be downloaded by hand.

| Stage | Datasets | Config |
| --- | --- | --- |
| 1 | 5 OpenML-CC18 datasets: 31 credit-g, 37 diabetes, 54 vehicle, 1464 blood-transfusion, 1494 qsar-biodeg | `stage1.yaml` |
| 2 | those 5 + a seeded random sample of OpenML-CC18, 25 total | `stage2.yaml` |
| 3 | all of OpenML-CC18 that passes the filters + 60 extra datasets from the OpenML catalogue | `final.yaml` |

OpenML-CC18 is a curated benchmark of 72 classification tasks (Bischl et al., *OpenML Benchmarking
Suites*, NeurIPS Datasets & Benchmarks 2021). Extra datasets are drawn from active OpenML datasets
that fit the size filters; one version per dataset name is kept. **Review `data/final/candidates.csv`
and `data/final/metadata.csv`**; put duplicates or unsuitable datasets in `exclude_ids` and re-run
`fetch` before splitting.

### Filters and cleaning (`autoclassify/data/sources.py`)

Applied identically to every dataset and logged in `metadata.csv` (status + reason):

* 500 ≤ rows; datasets above 50,000 rows are stratified-subsampled to 50,000
* ≤ 500 features after one-hot encoding (≤ 20 levels per categorical column), ≤ 50 classes
* classes with fewer than 10 rows are dropped (count logged); at least 2 classes must remain
* constant columns and ID-like text columns (> 90% unique) are dropped

### Splits (`autoclassify/data/splits.py`)

* **Dataset level:** meta-train / meta-val / meta-test (70/10/20 in Stage 3), stratified by size and
  binary-vs-multiclass, saved to `data/<stage>/dataset_split.json`. The file is never overwritten
  (unless `split --overwrite`); commit it to git as the frozen split.
* **Row level:** for each dataset and seed, stratified 60/20/20 train/validation/test rows.
* **Budgets:** 10/25/50/100% of the train rows, stratified and nested (10% ⊂ 25% ⊂ 50% ⊂ 100%).

### Preprocessing (`autoclassify/data/preprocess.py`)

Fitted on the rows a model is trained on, never on validation or test rows:

* numeric: median imputation + standardisation
* categorical: constant "missing" imputation + one-hot (rare levels grouped)
* CatBoost receives raw categorical columns and encodes them itself

## 4. Running the project, stage by stage

```bash
CFG=configs/stage1.yaml
python -m autoclassify.cli fetch    --config $CFG   # download + clean + filter      -> data/stage1/
python -m autoclassify.cli split    --config $CFG   # frozen dataset split + row splits
python -m autoclassify.cli perfdb   --config $CFG   # all fits (long, resumable)    -> results/stage1/perfdb/
python -m autoclassify.cli status   --config $CFG   # completeness, time-outs, median seconds per classifier
python -m autoclassify.cli features --config $CFG   # dataset features for the RL state -> results/stage1/meta/
python -m autoclassify.cli train    --config $CFG   # Q-learning, DQN, Double DQN    -> results/stage1/agents/
python -m autoclassify.cli evaluate --config $CFG   # all methods on meta-val        -> results/stage1/eval/
python -m autoclassify.cli figures  --config $CFG   #                                -> results/stage1/figures/
python -m autoclassify.cli ablate   --config $CFG   # ablations on meta-val
```

`python -m autoclassify.cli all --config $CFG` runs fetch → figures (meta-val only).

Useful options: `perfdb --shard 0/3` (split datasets over sessions), `--dids 31 37` (subset),
`train --algos ddqn --seeds 0 1`, `ablate --names no_dataset_features one_shot`.

### Final evaluation on meta-test (once)

```bash
python -m autoclassify.cli evaluate --config configs/final.yaml --split test
python -m autoclassify.cli figures  --config configs/final.yaml --split test
```

The first call writes `results/final/eval/TEST_OPENED.json`; a second call is refused unless you
pass `--force`. Tune only on meta-val (Stage 2), freeze `final.yaml`, then open meta-test.

## 5. What each output is

| File | Content |
| --- | --- |
| `data/<s>/metadata.csv` | every dataset tried: kept/rejected, reason, sizes |
| `results/<s>/perfdb/<did>.csv` | one row per (seed, classifier, budget): val/test balanced accuracy and macro-F1, times, status, hardware, versions |
| `results/<s>/meta/metafeatures.csv` | raw dataset features per (dataset, seed), from training rows |
| `results/<s>/meta/context.csv` | the 17 standardised features the agent sees (scaling fitted on meta-train) |
| `results/<s>/meta/baseline_stats.json` | average-rank order and average cost per classifier on meta-train (used by baselines) |
| `results/<s>/agents/full/*_history.csv` | training curves (meta-val regret during training) |
| `results/<s>/eval/val_results.csv` | one row per (method, dataset, seed, budget): regret, test score, budget used, seconds, full search trace |
| `results/<s>/eval/val_summary.csv`, `val_report.md` | summary table, average ranks, Friedman/Nemenyi, Wilcoxon (Holm) |
| `results/<s>/figures/` | regret-vs-budget, CD diagrams, cost-vs-regret, action heatmaps, search traces, training curves |

## 6. Key definitions (state these in the report)

* **State (3K + 4 + 17 numbers; 57 with 12 classifiers).** Per classifier: level reached, best observed
  validation score minus best so far, observed cost scaled to a 100% estimate. Budget used /
  remaining / total, best score so far minus chance. 17 standardised dataset features: log rows,
  log features, log encoded features, log rows/features, classes, categorical ratio, missing ratio,
  class entropy, majority ratio, imbalance ratio, mean |skewness|, mean kurtosis, mean |correlation|,
  PCA components for 95% variance, first PCA component share, mean mutual information, mutual
  information / class entropy. Ablations zero a block (`flags`) to measure what it contributes.

* **Budget semantics.** Budget level = fraction of train rows. Every fit is from scratch; no classifier
  is trained incrementally. "Continue" = refit on more data, and its full cost is charged. For TabPFN
  (no training), the fraction is the number of in-context rows.
* **Cost.** Wall-clock seconds of preprocessing + fit + validation prediction (test prediction is not
  charged), normalised per dataset by the cost of exhaustive search (all classifiers at 100%).
  All classifiers of a dataset run in the same session, so the normalisation compares like with like.
* **Time-outs.** A fit longer than `timeout_s` (default 300 s) counts as a time-out (chance score,
  cost = timeout). Larger budgets are skipped when the extrapolated time exceeds the limit
  (`growth_exponent`). This rule is explicit because Python cannot safely kill a running C-level fit.
* **Selection rule (all methods).** The classifier with the best observed validation score is selected.
  Final refit at 100% is not charged, for every method alike. No evaluation → regret 1.
* **Normalised regret** = (best − selected) / (best − worst) of the 100%-budget test balanced accuracy.
* **Reward (training).** −λ·cost per evaluation, and 1 − regret at the end, with regret computed from
  **validation** scores: no test data shapes the policy, even of training datasets.
* **Leakage guards.** Dataset features computed on training rows; their scaling and the baseline
  statistics use meta-train datasets only; preprocessing fitted on training rows; meta-test opened
  once (marker file).

## 7. Recommended workflow (matches the project plan)

1. `make smoke` / `pytest` locally → everything runs.
2. **Stage 1** on Kaggle GPU → confirm all 12 classifiers work and read `status` to see seconds per
   dataset. Use it to estimate Stage 3 time (≈ seconds per dataset × 3 seeds × ~120 datasets).
3. **Stage 2** → debug, tune `agents.dqn`, `env.lam`, training length on **meta-val** only.
4. Copy the chosen hyperparameters into `final.yaml`, review candidate datasets, `git tag final-v1`.
5. **Stage 3** → `fetch`, `split`, `perfdb` (shards), `meta`, `train`, `evaluate`, `ablate`, `figures`.
6. Open meta-test once; write up.

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Missing packages for: tabpfn ...` | `pip install tabpfn` or remove it from `classifiers:` |
| fetch fails with network errors | Kaggle: Settings → Internet On. Datasets are cached after the first download |
| `dataset X seed Y incomplete` | `perfdb` did not finish: run it again (it resumes) |
| many `timeout_predicted` for svm | expected on large data; it is part of the cost trade-off. Raise `timeout_s` only before Stage 3 |
| TabPFN errors on many classes | recorded as `error` (chance score) for that dataset; check `status` |
| Kaggle session ended mid-run | add previous output as input, set `PREVIOUS`, run again |
| meta-test refused | intended; see section 4 |

## 9. References

Rice (1976) *The Algorithm Selection Problem* · Nguyen, Sun-Hosoya, Guyon (2024) *Meta-Learning from
Learning Curves for Budget-Limited Algorithm Selection*, arXiv:2410.07696 · Abdulrahman, Brazdil, van
Rijn, Vanschoren (2018) *Speeding up algorithm selection using average ranking and active testing by
introducing runtime*, Machine Learning 107 · Feurer et al. (2022) *Auto-Sklearn 2.0*, JMLR 23 ·
Jamieson & Talwalkar (2016) successive halving · Bischl et al. (2021) *OpenML Benchmarking Suites* ·
Alcobaça et al. (2020) *MFE*, JMLR 21 · Holzmüller et al. (2024) *RealMLP*, NeurIPS · TabPFN-2.5
(Prior Labs, 2025) · Mnih et al. (2015) DQN · van Hasselt et al. (2016) Double DQN · Demšar (2006)
*Statistical comparisons of classifiers over multiple data sets*, JMLR 7.
