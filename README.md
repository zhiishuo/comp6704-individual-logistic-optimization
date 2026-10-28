# Certified optimization of L2-regularized logistic regression

COMP6704 individual study: compare GD, accelerated GD, monotone function-restarted
AGD, L-BFGS, Newton-CG and Jacobi-preconditioned Newton-CG on real binary
classification data. Educational optimizer updates are implemented in
`experiment.py`; an independent SciPy solve supplies a reference solution.
This is a comparative study, not a claim to invent any of these algorithms.

## Reproduce

Python 3.11 was used. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py
python verify.py
python analyze.py
```

The checked-in `results/` contain 102 real runs, final parameter vectors, split
indices, full per-iteration traces and reference solutions. Existing run files
are reused, so the default command resumes an interrupted run. For a fresh
benchmark, rename `results` to `results_original` before executing. For a small
18-run check use `python experiment.py --quick`; it writes `results_quick/` and
does not replace the full results. `analyze.py` intentionally reads the full
`results/`. It writes `analysis/` figures, numerical tables and a summary.
The downloadable report source bundles an unmodified copy of the generated
analysis tables and figures; no numerical table entries are manually typed.
To rebuild it after new experiments, copy the contents of `analysis/` into
the report source's `analysis/` folder, then compile `main.tex` with Tectonic
or an appropriate LaTeX installation. The public code repository does not
include the personal report or student ID.

## Data and attribution

* Wisconsin Diagnostic Breast Cancer, 569 observations, 30 input features.
  Positive label is benign (the scikit-learn target convention).
  W. Wolberg, O. Mangasarian, N. Street and W. Street (1993), UCI.
  https://doi.org/10.24432/C5DW2B
* Optical Recognition of Handwritten Digits: the 1,797-image, 64-feature subset
  bundled with scikit-learn, **not** the complete original UCI dataset. The
  binary target is odd versus even (odd = 1). E. Alpaydin and C. Kaynak (1998),
  UCI. https://doi.org/10.24432/C50P49
* Bank Marketing, `bank-full.csv`, 45,211 observations. S. Moro, P. Rita and
  P. Cortez (2014), UCI. https://doi.org/10.24432/C5K306
  `bank_marketing.zip` is the original UCI download, including both nested ZIP
  files. Only `bank.zip/bank-full.csv` is used. The post-contact `duration`
  feature is excluded. Six numeric columns are standardized; nine categorical
  columns are one-hot encoded. `pdays=-1` remains a documented sentinel value;
  unknown categorical values are retained as categories.

All three source datasets are distributed by UCI under CC BY 4.0; credit the
creators and preserve source license notices. No private student or group data
are included. Source ZIP SHA-256:
`e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`.

## Objective and protocol

`f(w) = mean(log(1+exp(Xw)) - y*(Xw)) + lambda*||w||^2/2`.
The appended intercept is penalized too. Stable logaddexp and sigmoid formulas
are used. The stopping certificate is `||gradient||^2/(2*lambda) <= 1e-6`,
which upper-bounds absolute objective suboptimality by strong convexity.
The same certificate is used for every optimizer; SciPy's status alone is not
accepted as proof of convergence. Reference certificates are below 1e-8.

* 80/20 stratified holdout, seeds 0, 1, 2 for main comparisons; transformations
  fit only on training rows. No test-driven hyperparameter selection.
* Main lambda = 1e-3; seed-0 sensitivity lambda = 1e-4 and 1e-2.
* Seed-0 raw-unit stress tests on Cancer and Bank. Isotropic L2 penalties in
  raw and standardized coordinates define different models: this is NOT a
  pure reparameterization ablation.
* Zero initialization, float64, single thread, 2,500 outer iterations or 30 s.
* First-order methods use the safe global spectral upper bound
  `L = lambda_max(X.T@X)/(4*n) + lambda`, computed once outside solve timing.
  This dense Gram setup limits the method to moderate feature dimension.
* L-BFGS keeps 10 curvature pairs and uses Armijo backtracking (not strong
  Wolfe); pairs failing a positive-curvature check are skipped.
* Newton inner CG uses residual forcing `min(0.5,sqrt(||g||))`, capped at
  `min(200,2*d)` iterations. PCG uses the exact Hessian diagonal. Both use
  Armijo backtracking with coefficient 1e-4 and halving (maximum 60 trials).
* Restart-AGD resets momentum and takes a GD fallback whenever the candidate
  increases objective by more than 1e-14. This makes the accepted trace
  monotone, but an optimal accelerated rate is not asserted for the variant.

Solve times include logging, line searches and inner iterations, but exclude
data loading, preprocessing, spectral setup, reference solving and test metrics.
One solve per split/configuration was timed. Ranges across three main splits
are descriptive; they are not repeated-run timing confidence intervals.
Small sub-millisecond differences should not be overinterpreted.

`passes = 2*fg + f + 2*hvp + diag`: forward/transpose products count separately;
curvature construction counts one forward pass and Jacobi construction counts
one additional scan. This is an approximate data-access cost, not a FLOP count.
Histogram/pair operations, Python overhead and spectral setup are excluded.

## Verification and limits

`verify.py` independently recomputes objectives, reference bounds, certificates,
test metrics, split disjointness, work counts and monotonicity from saved
parameter vectors. Central differences check gradient and Hessian-vector
products on a fixed random fixture. All failures remain in the archive:
84/102 runs certify; 18 reach the iteration limit. Of 54 main runs, 48 certify.

Bank observations are randomly split for optimizer benchmarking; this is not
a time-forward deployment evaluation, and repeated contacts may share clients.
The study does not establish clinical utility, fairness or deployment readiness.
It does not benchmark stochastic solvers or high-dimensional neural networks.
Prediction accuracy is distinct from optimization convergence.

Implementation, experiments and report preparation used AI assistance. The
saved receipts document what was executed and checked; they do not imply any
unobserved manual review or independent implementation by the student.
