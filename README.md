# SYNTHIA

**A physics-informed neural network for the 1D heat equation, trained to obey the PDE instead of memorizing its solution.**

[![Live App](https://img.shields.io/badge/live-synthia--pinn.streamlit.app-f77f00?style=flat-square)](https://synthia-pinn.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c?style=flat-square)](https://pytorch.org/)

**[Try it live →](https://synthia-pinn.streamlit.app/)**

<!-- 
DEMO GIF, record this before publishing:
1. Open the live app
2. Forward Solver tab: move the α slider through 2-3 values, let the heatmap redraw
3. Parameter Recovery tab: drag the noise slider across all 4 levels, let the chart update
4. ~8-10 seconds total, export at <5MB (LICEcap or Kap), save as figures/demo.gif
5. Uncomment the line below
-->
<!-- ![SYNTHIA demo](figures/demo.gif) -->

---

## What this is

A neural network that learns to solve the 1D heat equation by minimizing how badly it violates the PDE at sampled points. No labeled solution data, no mesh. It's validated against an independent finite-difference solver, extended to recover an unknown physical parameter (thermal diffusivity, α) from sparse noisy measurements, and wrapped with Monte Carlo Dropout so it reports how confident it is at every point, not just a number.

Three things happen in the deployed app:

- **Forward Solver**: set α, watch the temperature field evolve, see the PINN's prediction validated live against a classical solver it never saw during training
- **Parameter Recovery**: an inverse problem. Given noisy scattered temperature readings, recover α. Compared head-to-head against `scipy.optimize.curve_fit`
- **Uncertainty Quantification**: Bayesian-approximate confidence bands via MC Dropout, visualized as a full domain heatmap

## Results

**Forward solver** (α = 0.1, validated against both an independent FD solver and the exact analytical solution)

| Metric | Value |
|---|---|
| Relative L2 error vs. FD solver (t=1.0) | 2.43% |
| Relative L2 error vs. FD solver (t=0.5) | 1.54% |
| Relative L2 error vs. exact analytical solution (t=0.5) | 1.54% |
| Inference speedup vs. FD solver | **32.5×** |
| Training time | 124 seconds |

**Uncertainty quantification** (200 MC Dropout passes, full (x,t) domain)

| Metric | Value |
|---|---|
| Mean uncertainty (std) across domain | 0.0530 |
| Max uncertainty | 0.0857, at (x=0.47, t=0.14) |
| Pattern | Uncertainty is lowest at the directly-constrained boundaries and t=0, highest in the unsupervised interior. Confidence tracks proximity to constraints, not local gradient steepness. |

**Inverse problem, parameter recovery from noisy data** (α_true = 0.1, initialized at 0.5, 500 obs. points, 5 seeds)

| σ (noise) | `curve_fit` error | PINN mean error |
|---|---|---|
| 0.01 | 1.05% | 1.63% |
| 0.05 | 5.37% | 8.48% |
| 0.10 | 10.77% | 17.14% |
| 0.20 | 21.54% | **49.06%** |

> **Honest finding:** across every noise level and every setup I tested, including a version of `curve_fit` deliberately handed a misspecified model function, and versions with far fewer observation points, `curve_fit` outperformed the PINN, and the gap widens sharply with noise. This contradicts my original hypothesis, that the PDE constraint would act as regularization and let the PINN win at high noise. See [why `curve_fit` wins](#why-curve_fit-wins-and-why-its-worth-reporting) below, the why is the actual result here.

## Why `curve_fit` wins, and why it's worth reporting

My original hypothesis was a crossover: `curve_fit` should win at low noise (it's handed the exact analytical solution form), the PINN should win at high noise (its PDE constraint should act as regularization that `curve_fit` doesn't have). That crossover never happened, across three separate experiments:

1. **Baseline** (500 obs, `curve_fit` given the exact single-mode solution form). `curve_fit` won at every σ.
2. **Model misspecification.** I added a spurious second Fourier mode to `curve_fit`'s model function to remove its "exact form" advantage. It barely mattered: higher Fourier modes decay fast (rate ∝ n²απ²), so the nuisance mode was cheap for `curve_fit` to absorb.
3. **Fewer observations** (500 → 50 points). The gap widened, the opposite of what the "PDE constraint helps most when data is scarce" hypothesis predicted. At σ ≥ 0.1, PINN failures were bimodal, not uniform: some seeds recovered α correctly, others collapsed to non-physical values near zero, which inflates the mean error far more than gradual degradation would. The same pattern shows up in the full 500-obs sweep above: at σ=0.2, PINN error balloons to 49% with a standard deviation (32pp) nearly as large as the mean. One seed recovered α ≈ 5.6×10⁻⁵, essentially zero, while others landed within a few percent of the true value.

**Takeaway:** `curve_fit`'s advantage isn't really about model correctness or data density. It's a structurally simpler, better-behaved optimization problem: few parameters, no coupled PDE constraint that can go degenerate. The PINN's failures look like an optimization pathology, not a hard ceiling on the method.

**Leading suspect, not yet fixed:** the PDE/BC/IC collocation points in `inverse_pinn.py` sample `t` from `[0,1)` without scaling by the solver's actual `T_max`, while the data loss correctly scales observed `t` values. This likely dilutes how tightly the data term can pin down α, which would explain the bimodal seed collapse. Documented here as a known limitation, fixing it is the top item on my follow-up list.

## Method

The network takes `(x, t)` and predicts temperature `u`. It's trained on three physics-based loss terms, no solution data at all:

```
∂u/∂t = α · ∂²u/∂x²      on [0,1] × [0,1]
u(0,t) = u(1,t) = 0        (boundary)
u(x,0) = sin(πx)           (initial condition)
```

- **L_pde**: penalizes violation of the PDE residual at 2,000 random collocation points, computed via `torch.autograd.grad` (exact derivatives, no finite differences)
- **L_bc**: penalizes nonzero temperature at the rod's fixed ends
- **L_ic**: penalizes deviation from the initial condition at t=0

For the **inverse problem**, α becomes a learnable `nn.Parameter` (initialized at 0.5, far from the true 0.1), and a fourth term, **L_data**, fits the network to sparse noisy observations. L_data is what breaks the underlying degeneracy: L_pde alone can't pin down α, since the network can reshape its output to satisfy the PDE for any α.

For **uncertainty**, dropout (p=0.1) is left active at inference (`model.train()`, not `.eval()`). 200 stochastic forward passes approximate sampling from a Bayesian posterior over the network's weights ([Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)).

Architecture: `[2] → [64, tanh] × 3 → [1]`. Reference implementation: [Raissi et al., 2019](https://arxiv.org/abs/1711.10561).

## Quick start

```bash
git clone https://github.com/Abhineeer/synthia.git
cd synthia
pip install -r requirements.txt
streamlit run app.py
```

Runs on CPU, no GPU required for inference (training the PINN from scratch uses an RTX 4060, about 2 minutes).

## Repo structure

```
synthia/
├── app.py                       # Streamlit app (3 tabs)
├── pinn.py                      # Forward PINN model + training
├── inverse_pinn.py              # Inverse PINN (learnable α)
├── baseline_curvefit.py         # scipy.optimize.curve_fit baseline
├── solvers/
│   └── heat_fd.py                # Finite-difference ground truth solver
├── benchmarks/
│   ├── benchmark_phase1.json
│   ├── benchmark_inverse_final.json
│   └── benchmarks_uncertainty.json
├── models/
│   └── heat_pinn.pth
└── figures/
    └── fig_uncertainty.png
```

## Tech stack

PyTorch (CPU wheel for deployment), NumPy, SciPy, Matplotlib, Plotly, Streamlit. Deployed on Streamlit Community Cloud.

## License

MIT, see [LICENSE](LICENSE).

## Author

**Adii Singh**, ASU, Applied Physics + Computer Science
[GitHub](https://github.com/Abhineeer/synthia) · [Live app](https://synthia-pinn.streamlit.app/)
