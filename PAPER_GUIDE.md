# Experiment Reduction for Turbine Hill Charts — Co-Author Guide

*Written for the mechanical-engineering side of the team (no data-science background assumed) and for whoever drafts the manuscript. Every term is defined in plain language the first time it appears. Numbers below come from the real `bf-gv` dataset and are reproducible with `uv run python bayesian_reduction.py`.*

---

## 1. What problem are we solving?

Building a full hill chart means running the turbine at **many operating points** and measuring efficiency at each. Our ground-truth map has **72 measured experiments** (8 heads, 15–22 m, × ~9 guide-vane angles). Each real test costs time, water, and instrumentation.

**Question of the paper:** how few of those 72 tests do we actually need? If a computer model can "fill in" the points we *didn't* measure, we can run a small subset and still recover the whole chart. The paper compares **strategies for choosing which points to measure**.

Two things we might want out of the reduced test campaign — and they are **not the same goal**:

- **(A) Global reconstruction** — get the *entire* efficiency map η(Q, H) right, so we can draw the hill chart, check off-design behaviour, write guarantees.
- **(B) BEP localization** — just find the single **Best Efficiency Point** (the peak).

The central finding of the paper is that **these two goals need different sampling strategies.** A method that is excellent at one can be poor at the other.

---

## 2. The "fill-in" model: Gaussian Process (GP), in plain terms

A **Gaussian Process** is a flexible interpolating surface fitted through the measured points. Think of a rubber sheet pushed through your data points. Two properties matter:

1. It predicts efficiency at any (Q, H), not just where you measured.
2. **It also reports its own uncertainty** — "I'm confident near measured points, unsure far from them." This uncertainty is what the smart sampling methods exploit.

We use the same GP for every method, so the comparison is purely about *which points each method chooses to measure* — nothing else changes.

---

## 3. The six sampling strategies

The first three ("space-filling") choose all their points up front. The last three ("sequential") choose one point at a time, re-fitting the GP after each, so each new point depends on what's been learned so far.

| Method | What it does | Mechanical-engineer analogy | Really good at |
|---|---|---|---|
| **Random** | Picks points at random | Blindfolded dart throws | Nothing in particular (baseline) |
| **Grid** | Even Q×H grid | Classic full-factorial test matrix | Coverage, simplicity |
| **LHS** (Latin Hypercube) | Spreads points so every Q-band and H-band is hit once | Stratified test plan | Robust coverage with few points |
| **BO-UCB** (Bayesian Optimization) | Chases the peak: samples where efficiency looks *high and uncertain* | Hill-climbing toward max efficiency | **Finding the BEP** |
| **ALM** (active learning, MacKay) | Samples where the GP is **least certain** | "Go measure where we're most blind" | **Reconstructing the whole map** |
| **ALC** (active learning, Cohn) | Samples the point that reduces the model's **total uncertainty over the whole map** the most | "Which one test tightens the entire chart the most?" | **Reconstructing the whole map** |

**Key distinction to put in the paper:** BO-UCB is an **optimization** method (built to find a maximum). ALM and ALC are **active-learning** methods (built to make an accurate model). Using an optimizer to build a full map is the wrong tool — and our data shows exactly why.

> **ALM vs ALC in one line:** ALM looks only at the single most-uncertain point (cheap, greedy). ALC looks ahead at how a candidate would shrink uncertainty *everywhere* (smarter, slower). In our data they perform almost the same; ALM is the cheaper default.

---

## 4. How we score them (the comparison metrics)

Two metric families, matching the two goals.

### (A) Global reconstruction — surrogate vs. all 72 measured points
| Metric | Meaning | Better when | Use it for |
|---|---|---|---|
| **MAE** | Average absolute efficiency error (e.g. 0.002 = 0.2 efficiency-points off) | lower | Most interpretable headline |
| **RMSE** | Like MAE but punishes big misses harder | lower | Sensitivity to outliers |
| **MAPE** | Average error as a **percentage** | lower | Abstract-friendly ("<0.5 % error") |
| **R²** | Fraction of the efficiency variation the model captures; 1.0 = perfect | higher | "How good overall" (≥0.99 = excellent) |

### (B) BEP localization — predicted best point vs. true best point
| Metric | Meaning | Better when |
|---|---|---|
| **ΔQ** | Discharge distance to true BEP (m³/s) | lower |
| **ΔH** | Head distance to true BEP (m) | lower |
| **Δη** | Efficiency error at the located point | lower |

**Recommended headline metrics for the manuscript:** report **R² and MAPE** for reconstruction and **Δη (plus ΔQ, ΔH)** for BEP. The full set for every method/budget is in `result/study_summary.csv`.

---

## 5. The reduction sweep — and why we vary the number of experiments

Ground truth = 72 experiments. If we keep **n**, the **reduction = (72 − n) / 72**:

| n kept | Reduction | |
|---|---|---|
| 6 | **91.7 %** | extreme |
| 8 | **88.9 %** | aggressive |
| 10 | **86.1 %** | |
| **12** | **83.3 %** | recommended demo budget (see §6) |
| 16 | 77.8 % | |
| 20 | 72.2 % | |
| 25 | 65.3 % | |
| 30 | 58.3 % | conservative |

We don't test just one budget — we **sweep n from 6 to 30** and average over **20 random seeds** (see §7). This produces the **learning curves** (`fig_learning_curves`): error vs. number of experiments. That sweep *is* the argument of the paper — at large n everyone is accurate (boring); the value is showing **how few tests you can get away with**, which is only visible at low n.

---

## 6. Why the illustration figures use n = 12 (it is a criterion, not a guess)

The learning curves already use *all* budgets. But the 3D/2D reconstruction pictures need **one** budget to display. We pick it by an engineering criterion:

> **Smallest n at which the recommended method reaches engineering-grade accuracy: R² ≥ 0.99 and MAPE < 0.5 %.**

ALC reaches this at **n = 10** (R² 0.995, MAPE 0.46 %); ALM at **n = 12** (R² 0.996, MAPE 0.37 %). We use **n = 12 (83.3 % reduction)** as the conservative point where *both* recommended methods clear the bar. It is defensible to also show an aggressive **n = 8 (88.9 %)** panel to illustrate the practical floor. **Do not** headline a single budget without stating this criterion — a reviewer will ask "why 12?".

---

## 7. What "seed" and "regime" mean (so the text is precise)

- **Seed** = the random-number setting that fixes the random initial points (and the Random/LHS draws). We run **20 seeds (0–19)** and report **mean ± standard error** — this turns one-off luck into a statistically honest result. The picture figures use **seed 0**.
- **Regime** = which part of the sweep we're discussing. **Low-n regime** = few experiments (n ≈ 6–12), where methods separate — the interesting region. **Saturated regime** = n ≈ 25–30, where everyone is already ~R² 0.99 and differences vanish. *Your original single run at n = 30 sat in the saturated regime — that's why Bayesian and Random looked identical.*

---

## 8. Headline results (real numbers, use these in the abstract)

**At n = 12 (83 % fewer experiments), mean over 20 seeds** — from `result/table_n12.csv`:

| Method | Reconstruction MAE | R² | MAPE | BEP Δη |
|---|---|---|---|---|
| **ALC** | **0.0022** | **0.9965** | **0.35 %** | 0.0071 |
| **ALM** | 0.0024 | 0.9961 | 0.38 % | 0.0102 |
| LHS | 0.0027 | 0.9945 | 0.44 % | 0.0075 |
| Grid | 0.0031 | 0.9938 | 0.50 % | 0.0113 |
| Random | 0.0061 | 0.9380 | 1.07 % | 0.0086 |
| BO-UCB | 0.0099 | 0.9207 | 1.77 % | **0.0003** |

Two clean, opposite stories (this is the paper):

- **For the full hill chart → active learning wins.** ALM/ALC reconstruct with R² ≈ 0.996 at 83 % reduction — **3–4× lower error than BO-UCB**, which clusters its tests near the peak and starves the rest of the map (see `fig_sample_placement`).
- **For the BEP alone → Bayesian optimization wins.** BO-UCB pins the BEP (Δη ≈ 2×10⁻⁶, exact Q and H) by **n ≈ 14**, while the others plateau ~10⁻². See `fig_learning_curves` panel (b).

**One honest nuance to include:** below ~8 experiments the GP is fitted on too little data, so the model-based active learners can wobble (at n = 6, ALC drops to R² 0.84). In that extreme-reduction corner, **space-filling LHS is the safer choice** (R² 0.98 at n = 6). Recommendation for practitioners: **LHS for n < 8, active learning (ALM/ALC) for n ≥ 10.**

---

## 8.1 Is the difference statistically significant? (Wilcoxon signed-rank)

Averages can be luck. We test each claim with a **Wilcoxon signed-rank test** — a standard non-parametric paired test — using the 20 seeds as matched replications. It answers: "across seeds, does method A reliably beat method B, or could the gap be noise?" `p < 0.05` = significant. Files: `result/wilcoxon_n12.csv` (headline) and `result/wilcoxon_mae_vs_n.csv` (every budget).

**Reconstruction (MAE), reference = ALC, at n = 12 (20 seeds):**

| ALC vs | p-value | significant? |
|---|---|---|
| BO-UCB | <10⁻⁵ | **yes** |
| Random | <10⁻⁴ | **yes** |
| Grid | 0.0001 | **yes** |
| LHS | 0.009 | **yes** |
| ALM | 0.09 | no (they are equivalent) |

**BEP (Δη), reference = BO-UCB, at n = 12:** BO-UCB beats **every** other method at `p < 10⁻⁵`. BEP localization is BO-UCB's decisive, unambiguous win.

**Reading it across budgets** (`wilcoxon_mae_vs_n.csv`) gives the fair, defensible wording:
- **ALC/ALM significantly beat BO-UCB and Random** at essentially every budget n ≥ 8.
- **ALC/ALM beat LHS and Grid at n = 12 and above** (with 20 seeds the LHS gap is now significant, p = 0.009 — it was only borderline at 10 seeds). LHS remains the strongest *non-adaptive* competitor and the safe choice at the very lowest budgets (n < 8).
- **ALM vs ALC is never significant** → they are statistically equivalent, so **recommend ALM** (same accuracy, cheaper to compute).

> **Note on the p-value floor:** with 20 seeds the smallest one-sided p the exact test can return is 1/2²⁰ ≈ **10⁻⁶**, so the strong wins above are genuine, not clipped. More seeds → smaller achievable p and tighter error bars.

---

## 9. Figure & table list for the manuscript

Everything below is already generated in `result/` (PDF for LaTeX + PNG). Grouped by the two perspectives.

**The two-goals figure (put it up front)**
- `fig_goal_contrast` — ALC vs BO-UCB reconstruction at the *same* n = 12, each marking sampled points, true BEP (★) and predicted BEP (✕). Shows in one frame: BO-UCB pins the BEP but distorts the map; ALC recovers the map. **This is the paper's thesis in one figure.**

**Perspective A — global reconstruction**
- `fig_ground_truth_3d`, `fig_ground_truth_2d` — the measured hill chart (72 pts).
- `fig_recon_alc_3d`, `fig_recon_alc_2d` — ALC reconstruction at n = 12 with sampled points.
- `fig_recon_boucb_3d`, `fig_recon_boucb_2d` — the **BO-UCB** reconstruction for contrast (accurate near the peak, biased elsewhere).
- `fig_error_2d` — where the ALC reconstruction error lives.
- `fig_learning_curves` (panel a) — MAE vs n, all methods, ±SE bands. **Core evidence.**
- `fig_metrics_bars` (panel a) — method ranking at n = 12.
- `fig_sample_placement` — *why* methods differ: where each one puts its 12 tests.

**Perspective B — BEP localization**
- `fig_learning_curves` (panel b) — BEP error vs n. **Core evidence.**
- `fig_metrics_bars` (panel b) — BEP ranking at n = 12 (note the inversion vs panel a).

**Tables**
- `table_n12.csv` — methods × all metrics at the recommended budget (drop straight into LaTeX).
- `wilcoxon_n12.csv` / `wilcoxon_mae_vs_n.csv` — significance tests (§8.1).
- Reduction-level table (§5).

All 3D figures share one viewing angle (`elev = 22°, azim = −52°`) so surfaces are directly comparable. Style is IEEE (Times, ~8 pt, viridis colormap, vector PDF).

---

## 10. Method theory for the manuscript

The mathematical formulation of the GP surrogate and the acquisition functions (ALM, ALC, and UCB for contrast) — equations, the sequential-design algorithm, complexity, and references — is written up separately in **`METHODS_ACTIVE_LEARNING.md`**, ready to drop into the paper's *Methods* section.

## 11. Still to do before submission

- ~~Significance test~~ — **done** (§8.1); Wilcoxon signed-rank in `result/wilcoxon_*.csv`.
- ~~Seed count~~ — now **20 seeds**; p-value floor ≈ 10⁻⁶.
- **Generalization** — repeat on a second dataset (e.g. `bf-bulb`) to show the conclusion isn't specific to one turbine.
- **Kernel sensitivity** — one paragraph confirming results are stable to the GP kernel choice.

## 12. Reproduce everything

```bash
uv run python bayesian_reduction.py     # writes all figures + CSVs to result/
```
Knobs at the top of `bayesian_reduction.py`: `SAMPLE_SIZES` (the sweep), `SEEDS`, `N_DEMO`/`DEMO_METHOD` (which budget/method the illustration figures use). Method internals live in `src/` — see `CLAUDE.md` for the code map.
