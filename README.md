# ExoDetect — classifying Kepler exoplanet candidates

Individual first-year project, BSc in Artificial Intelligence and Data Science,
University of Beira Interior (Portugal).

The Kepler telescope watched thousands of stars for small dips in brightness
caused by planets passing in front of them. Most of those signals turn out not to
be planets. The goal here is to train a model that, given a signal, decides
between three classes — **confirmed planet**, **candidate**, or **false
positive** — and to work out what that model is actually using to decide.

Data: the cumulative *Kepler Objects of Interest* table from the NASA Exoplanet
Archive (9,564 observations, 50 columns).

## Results

Random Forest, 500 trees, 26 features, stratified 80/20 split, `random_state=42`.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CANDIDATE | 0.66 | 0.46 | 0.54 | 449 |
| CONFIRMED | 0.81 | 0.85 | 0.83 | 459 |
| FALSE POSITIVE | 0.83 | 0.92 | 0.87 | 1,005 |
| **Accuracy** | | | **79.30%** | 1,913 |

Confusion matrix (rows = true class, columns = predicted class):

```
                 CAND  CONF   FP
CANDIDATE         205    72  172
CONFIRMED          45   391   23
FALSE POSITIVE     62    22  921
```

As a baseline, always answering "FALSE POSITIVE" would give 52.5%, since that is
the majority class. The 79.30% sits 27 points above it.

Candidates are the hardest class (recall 0.46), and the model's most frequent
error is classifying them as false positives. That does not strike me as a defect
of the model: a candidate is by definition an observation whose evidence was
inconclusive, and its properties overlap those of the other two classes.

## The part that taught me most: data leakage

The first version of this project reached 90.96% accuracy. It was using, among
the features, the `koi_score` column and the four `koi_fpflag_*` flags.

Those columns are not telescope measurements. They are the **output of NASA's own
vetting process**: `koi_score` is the confidence NASA itself assigned to the
classification, and the `koi_fpflag_*` flags record the reasons an observation was
marked as a false positive. Handing them to the model means handing it part of the
answer — it does not learn to recognise a planet, it learns to copy a decision
that had already been made.

I removed them and retrained, same code and same seed:

| | Accuracy | Recall CANDIDATE | Recall CONFIRMED | Recall FALSE POSITIVE |
|---|---|---|---|---|
| With `koi_score` and `koi_fpflag_*` (31 features) | 90.96% | 0.80 | 0.85 | 0.99 |
| Without them (26 features) | 79.30% | 0.46 | 0.85 | 0.92 |

Eleven accuracy points, and half the recall on candidates, were coming from those
five columns. The higher number was the wrong number, and 79.30% is what the model
can do from the properties of the signal. What made me suspicious was the SHAP
plot: `koi_score` showed up as the most important feature by a wide margin.

## What SHAP shows

After the fix, the three most influential features for the CONFIRMED class are
`koi_steff_err1`, `koi_duration_err1` and `koi_model_snr`.

I expected signal-to-noise (`koi_model_snr`) to come first, since it was the
feature that separated the classes best in the exploratory histograms. It appears,
but third. The top two are **measurement uncertainties** — the error on the star's
effective temperature and the error on the transit duration. So part of what the
model learned is how well each observation was measured, rather than the physics
of the planet. That is a limitation of this work and it is recorded as one.

The next step, if I return to this project, is to train using only the physical
measurement columns, dropping every `_err1` column, to measure how much of the
result depends on measurement quality.

## Layout

```
projeto_ExoDetect/
├── ExoDetect.ipynb          analysis, cleaning, training and SHAP, by phase
├── cumulative.csv           data from the NASA Exoplanet Archive
└── relatorio_exodetect.tex  written report (LaTeX, in Portuguese)
```

The notebook is divided into five phases, each stating what was done and why:
loading and inspecting the data, exploratory analysis, preprocessing, training and
evaluation, and explainability with SHAP.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
jupyter notebook projeto_ExoDetect/ExoDetect.ipynb
```

Run the cells in order. The notebook reads `cumulative.csv` from the same
directory. Computing the SHAP values is the slow part and takes a few minutes.

## Decisions worth justifying

- **Random Forest** — handles columns on very different scales, is robust to the
  outliers this dataset has in abundance, and exposes per-feature importance,
  which the last phase needed.
- **Median rather than mean** for filling missing values, because the
  distributions are strongly skewed and the mean would be dragged by the extreme
  values.
- **`stratify=y`** on the train/test split, so all three classes keep their
  proportion in both sets; without it the CANDIDATE metrics would stop being
  reliable.
- **MinMax scaling** — not required for decision trees. I did it so SHAP values
  stay comparable across columns measured in different units.
- **Dropping the `_err2` columns** — they are the lower error bar of the same
  measurement as the `_err1` columns, so the information is largely duplicated.
  For several columns the correlation between the two is exactly −1.

---

**Author:** Bernardo Nabais · University of Beira Interior · 2026
