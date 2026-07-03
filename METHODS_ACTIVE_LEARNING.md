# Methods: Gaussian-Process Active Learning (ALM & ALC) and Bayesian Optimization

*Draft methodology text for the manuscript. Notation is LaTeX-ready — equations use `$...$` / `$$...$$` and can be pasted into the paper. References are collected at the end. The core message threaded through this section: **ALM/ALC are active-learning rules that minimise model uncertainty (goal: reconstruct the whole hill chart); UCB is an optimisation rule that seeks the maximum (goal: locate the BEP).** Same surrogate, different acquisition, different objective.*

---

## 1. Problem setup

Let $x = (Q, H, \alpha)^\top$ denote an operating condition (discharge, head, guide-vane angle) and $y = \eta(x)$ the measured efficiency. We have a finite candidate pool $\mathcal{X} = \{x_1,\dots,x_N\}$ of $N = 72$ physically realisable test points. A sampling strategy selects a subset $S \subset \mathcal{X}$, $|S| = n \ll N$, at which experiments are actually run; a surrogate trained on $S$ then predicts $\eta$ over all of $\mathcal{X}$.

We seek the subset $S$ that best serves one of two objectives:

- **Global reconstruction:** minimise prediction error over the whole domain, $\frac{1}{N}\sum_i |\eta(x_i) - \hat\eta(x_i)|$.
- **Optimum (BEP) localisation:** identify $x^\star = \arg\max_x \eta(x)$.

These objectives call for different acquisition functions, developed below.

---

## 2. Gaussian-process surrogate

We model $\eta(\cdot)$ as a Gaussian process (GP),
$$\eta(x) \sim \mathcal{GP}\big(m(x),\, k(x,x')\big),$$
with constant mean $m$ and a Matérn covariance ($\nu = 5/2$),
$$k(x,x') = \sigma_f^2 \left(1 + \sqrt{5}\,r + \tfrac{5}{3}r^2\right)\exp\!\big(-\sqrt{5}\,r\big), \qquad r = \lVert x - x' \rVert_{\ell},$$
where $r$ is the length-scale-weighted distance and $\theta = (\sigma_f, \ell)$ are hyperparameters fitted by maximum marginal likelihood. Features are standardised before fitting.

Given training data $(X_S, y_S)$ at the sampled set $S$, the GP posterior at any $x$ is Gaussian with closed-form mean and variance:
$$\mu_S(x) = m + k(x, X_S)\,\big[K_S + \sigma_n^2 I\big]^{-1}(y_S - m), \tag{1}$$
$$\sigma_S^2(x) = k(x,x) - k(x, X_S)\,\big[K_S + \sigma_n^2 I\big]^{-1} k(X_S, x), \tag{2}$$
where $K_S = k(X_S, X_S)$ and $\sigma_n^2$ is a small noise/jitter term. **Equation (2) is central to active learning:** the posterior variance $\sigma_S^2(x)$ is the model's own admission of how uncertain it is at $x$, and it depends only on the *locations* $X_S$ — not on the observed values $y_S$. This lets us score a candidate point's informativeness *before* measuring it.

---

## 3. Sequential design as acquisition maximisation

All three model-based methods share one greedy loop; they differ only in the acquisition function $a(x)$:

> **Algorithm (sequential design).**
> 1. Initialise $S$ with $n_0$ points (here a small random design).
> 2. Fit the GP on $(X_S, y_S)$; obtain $\mu_S,\sigma_S$ from (1)–(2).
> 3. Select $x_{\text{next}} = \arg\max_{x \in \mathcal{X}\setminus S} a(x)$.
> 4. Run the experiment, append to $S$. Repeat 2–4 until $|S| = n$.

---

## 4. ALM — Active Learning MacKay (maximum variance)

**Principle:** measure where the model is least certain. The acquisition is simply the posterior standard deviation,
$$a_{\text{ALM}}(x) = \sigma_S(x). \tag{3}$$
Each step removes the single largest pocket of predictive uncertainty. ALM is the discrete, GP analogue of maximising information gain about the function value at the queried point; under a GP it is equivalent to greedily maximising the entropy of the selected observations [MacKay 1992]. It is cheap — one GP fit per step — and tends to place points in a near space-filling pattern, with a mild bias toward domain boundaries where variance is naturally highest.

---

## 5. ALC — Active Learning Cohn (integrated variance reduction)

**Principle:** don't just measure where *this* point is uncertain — measure the point whose observation most reduces uncertainty *everywhere else*. Define the integrated mean-square predictive variance over the domain,
$$\Phi(S) = \int_{\mathcal{X}} \sigma_S^2(x)\,dx \;\approx\; \frac{1}{N}\sum_{i=1}^{N}\sigma_S^2(x_i). \tag{4}$$
ALC selects the candidate that minimises the *look-ahead* integrated variance,
$$a_{\text{ALC}}(x) = -\,\Phi\big(S \cup \{x\}\big) = -\frac{1}{N}\sum_{i=1}^{N}\sigma_{S\cup\{x\}}^2(x_i). \tag{5}$$
Using (2), adding a candidate $x$ reduces the variance at every reference point $x_i$ by a closed-form amount, so the sum in (5) can be evaluated for all candidates without re-optimising hyperparameters — we hold the fitted kernel fixed within a step (standard practice; the variance in (2) is value-independent). ALC directly targets the reconstruction objective (it is the greedy minimiser of integrated MSE, the IMSE-optimal design criterion [Cohn 1996; Sacks et al. 1989]). It is more expensive than ALM — $O(N)$ candidate evaluations per step — but yields the most uniform global accuracy.

**ALM vs ALC in one sentence.** ALM is myopic (reduces uncertainty at the query point); ALC is one-step non-myopic (reduces *total* uncertainty across the domain). When the GP is well specified they behave similarly; ALC is preferable when uncertainty at unmeasured regions matters most, ALM when compute is constrained.

---

## 6. Contrast: Bayesian Optimisation (UCB)

For completeness and because it is the natural tool for the *other* objective, we include Bayesian optimisation with the Upper Confidence Bound acquisition [Srinivas et al. 2010; cf. Jones et al. 1998]:
$$a_{\text{UCB}}(x) = \mu_S(x) + \kappa\,\sigma_S(x), \qquad \kappa = 2. \tag{6}$$
UCB trades off exploitation ($\mu_S$, predicted-high efficiency) against exploration ($\sigma_S$, uncertainty). It deliberately concentrates samples near the predicted optimum. This is exactly right for **locating the BEP** and exactly wrong for **reconstructing the full map**: by design it under-samples low-efficiency regions, so the surrogate is accurate near the peak but biased elsewhere. Our results quantify both sides of this trade-off.

---

## 7. Baselines (non-adaptive)

- **Random** — $S$ drawn uniformly without replacement (control).
- **Grid** — a regular $Q\times H$ lattice snapped to the nearest realisable points (classical factorial design).
- **Latin Hypercube Sampling (LHS)** — a stratified space-filling design [McKay et al. 1979] snapped to realisable points; strong low-budget coverage without a model.

These choose all $n$ points up front (no GP in the loop) and isolate the value added by *adaptivity*.

---

## 8. Complexity and practical notes

| Method | GP fits per campaign | Per-step candidate cost | Adaptive? |
|---|---|---|---|
| Random / Grid / LHS | 1 (final) | — | no |
| ALM | $n - n_0$ | $O(N)$ variance eval | yes |
| BO-UCB | $n - n_0$ | $O(N)$ mean+var eval | yes |
| ALC | $n - n_0$ | $O(N^2)$ (variance at all $x_i$ per candidate) | yes |

A GP fit is $O(n^3)$ in the (small) training-set size; with $n \le 30$ all methods run in seconds to a couple of minutes over the full sweep. For larger candidate pools, ALC admits rank-one covariance updates and reference-set subsampling to reduce the per-step $O(N^2)$ cost.

---

## 9. Why this matters for turbine testing

The acquisition functions encode *what question the test campaign is asking.* If the deliverable is the **hill chart** — efficiency guaranteed across the whole operating envelope — the campaign should minimise global uncertainty (ALM/ALC). If the deliverable is only the **rated/best operating point**, it should climb toward the optimum (UCB). Choosing the sampler to match the deliverable is the actionable recommendation of this work; using an optimiser to build a map (or vice versa) wastes experiments.

---

## References

- MacKay, D. J. C. (1992). *Information-based objective functions for active data selection.* Neural Computation 4(4), 590–604. **[ALM]**
- Cohn, D. A. (1996). *Neural network exploration using optimal experiment design.* Neural Networks 9(6), 1071–1083; and Cohn, Ghahramani & Jordan (1996), *Active learning with statistical models,* JAIR 4, 129–145. **[ALC]**
- Sacks, J., Welch, W. J., Mitchell, T. J., Wynn, H. P. (1989). *Design and analysis of computer experiments.* Statistical Science 4(4), 409–423. **[IMSE-optimal design]**
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press. **[GP eqs. (1)–(2)]**
- Gramacy, R. B. (2020). *Surrogates: Gaussian Process Modeling, Design and Optimization for the Applied Sciences.* CRC Press. **[practical active learning / ALC]**
- McKay, M. D., Beckman, R. J., Conover, W. J. (1979). *A comparison of three methods for selecting values of input variables...* Technometrics 21(2), 239–245. **[LHS]**
- Srinivas, N., Krause, A., Kakade, S., Seeger, M. (2010). *Gaussian process optimization in the bandit setting: no regret and experimental design.* ICML. **[GP-UCB]**
- Jones, D. R., Schonlau, M., Welch, W. J. (1998). *Efficient global optimization of expensive black-box functions.* J. Global Optimization 13, 455–492. **[EGO / Bayesian optimisation]**
