# Cross-Fitting Under Dependent Data: Replication & Extension

Empirical evaluation of [Balkus, Laith & Hejazi (2026)](https://arxiv.org/abs/2601.10899) and [Chiang, Kato, Ma & Sasaki (2021)](https://arxiv.org/abs/1909.03489) on cross-fitting strategies for causal machine learning with two-way clustered data.

## Key Findings

1. **As-IID cross-fitting matches or beats Chiang K²-fold** across all DGPs, learners, and sample sizes — confirming Balkus et al.'s Theorem 1.
2. **MARS doesn't need cross-fitting at all** — its GCV pruning makes it approximately Donsker, achieving 89–96% coverage without any sample splitting.
3. **Deep RF requires cross-fitting** — without it, coverage drops to 0–45% due to empirical process bias from fully-grown trees memorizing training data.
4. **The real issue is variance estimation, not splitting strategy** — when Corollary 2's bounded-cluster assumption breaks, both strategies produce invalid inference because the CGM cluster-robust SE formula underestimates uncertainty.
5. **Software design flaw**: DoubleML bundles cross-fitting strategy with variance estimation into a single toggle. The optimal combination (as-IID splitting + cluster-robust SE) is not available through their API.

## Project Structure

```
├── src/                           # Shared simulation framework
│   ├── protocols.py               # DGPProtocol, EstimatorProtocol (PEP 544)
│   ├── scenarios.py               # ScenarioConfig dataclass
│   ├── evaluation.py              # Morris et al. (2019) performance measures
│   ├── utils.py                   # sigmoid
│   ├── dgps/                      # Data Generating Processes
│   │   ├── balkus_original.py     # Balkus hard_cluster.R (Section 3.1)
│   │   ├── balkus_nocompress.py   # Balkus new_cluster.R (no propensity /5)
│   │   ├── shared_effects.py      # Shared cluster confounders
│   │   ├── balkus_adversarial.py  # Bounded d-regular bipartite graph
│   │   ├── balkus_extreme.py      # Amplified bounded-cluster DGP
│   │   ├── balkus_fullgrid.py     # Dense grid (violates Corollary 2)
│   │   └── chiang_exact.py        # Chiang et al. (2021, Section 4.1) PLIV
│   ├── estimators/
│   │   ├── dml.py                 # AIPW/DR cross-fit estimator (Balkus R faithful)
│   │   ├── mars_r.py              # R earth() bridge via rpy2
│   │   └── plm_iv_dml.py         # Partially linear IV DML2 (Chiang Algorithm 1)
│   └── orchestration/
│       ├── runner.py              # Single-replication executor
│       └── orchestrator.py        # Parallel execution with CSV resume
│
├── replications/
│   ├── balkus/                    # Balkus et al. (2026) experiments
│   │   ├── main.py                # Entry: grid DGPs + adversarial
│   │   ├── main_limitations.py    # Entry: dense fullgrid experiment
│   │   ├── scenarios.py           # Scenario builders
│   │   ├── analysis.py            # Plot generation
│   │   └── results/               # CSVs + plots
│   └── chiang/                    # Chiang et al. (2021) experiments
│       ├── main.py                # Entry: PLIV + convergence tests
│       ├── scenarios.py           # Scenario builders
│       ├── analysis.py            # Plot generation
│       ├── doubleml_validation.py # DoubleML head-to-head (standalone)
│       └── results/               # CSVs + plots
│
└── report/
    └── balkus_replication_report.tex
```

## How to Reproduce

### Prerequisites
- Python ≥ 3.11
- R ≥ 4.0 with `earth` package (`Rscript -e "install.packages('earth')"`)

### Run simulations
```bash
pip install -r requirements.txt

# Balkus replication (grid DGPs + adversarial)
python -m replications.balkus.main

# Balkus limitations (dense fullgrid, Corollary 2 violation)
python -m replications.balkus.main_limitations

# Chiang replication (PLIV + convergence)
python -m replications.chiang.main
```

### Generate plots
```bash
python -m replications.balkus.analysis
python -m replications.chiang.analysis
```

### DoubleML validation (optional, requires `pip install doubleml`)
```bash
python -m replications.chiang.doubleml_validation
```

## Evaluation Framework

All performance measures follow [Morris, White & Crowther (2019)](https://doi.org/10.1002/sim.8086):

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| Bias | E[θ̂ − θ₀] | Systematic error |
| EmpSE | SD(θ̂₁, ..., θ̂ₙ) | True sampling variability |
| ModSE | √E[SE²] | What the variance formula reports |
| RelSE | ModSE / EmpSE | SE calibration (target: 1.0) |
| Coverage | P(θ₀ ∈ CI) | Nominal target: 95% |

## References

- Balkus, S. V., Laith, H., & Hejazi, N. S. (2026). *On the use of cross-fitting in causal machine learning with correlated units.* arXiv:2601.10899.
- Chiang, H. D., Kato, K., Ma, Y., & Sasaki, Y. (2021). *Multiway cluster robust double/debiased machine learning.* JBES, 40(3), 1046–1056.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). *Robust inference with multiway clustering.* JBES, 29(2), 238–249.
- Morris, T. P., White, I. R., & Crowther, M. J. (2019). *Using simulation studies to evaluate statistical methods.* Statistics in Medicine, 38(11), 2074–2102.
- Chernozhukov, V. et al. (2018). *Double/debiased machine learning for treatment and structural parameters.* The Econometrics Journal, 21(1), C1–C68.
