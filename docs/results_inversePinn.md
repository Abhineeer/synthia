Experimental Log
Inverse PINN vs curve_fit

Goal:
Crossover hypothesis: curve_fit should wins at low noise (has exact analytical model form), PINN should win at high noise (PDE constraint acts as regularization curve_fit lacks). 
Every experiment so far has curve_fit winning at every noise level tested. This log tracks what I have tried, what it showed, and what's left to explain the gap.

Experiment 1: Baseline (500 obs points, well-specified curve_fit)
Setup: 500 obs points, noise levels: {0.01, 0.05, 0.1, 0.2}, curve_fit given exact analytical solution form (single Fourier mode, n=1), 6 seeds.

Result: curve_fit wins at every noise level. At σ=0.05, PINN mean error ~7.6%.

Interpretation: Not a fair fight since curve_fit is handed the exact functionalform.
Encouraged me to design the misspecification experiment, detailed below.

Experiment 2: Model misspecification (500 obs, n=3 nuisance mode)
Setup: Added a spurious second Fourier eigenmode (n=3, nuisance coefficient c) to curve_fit's model function, trying to remove its "exact form" advantage.

Result: curve_fit still wins at every noise level. Misspecification cost curve_fit almost nothing, the nuisance mode was cheaply absorbed. 

Why n=3 didn't hurt curve_fit much: higher Fourier modes decay at rate n²απ², n=3 decays 9× faster than n=1, so by the time you're sampling observations, that mode's contribution is nearly gone. n =2 for that reason decays 4 times faster, which is stronger than n =3, its next on the docket.

Experiment 3: Fewer observation points (50 obs, well-specified, no misspecification)
Setup: Same as Experiment 1 but 500 → 50 obs points, 5 seeds, curve_fit
well-specified (n=1 only).

Result: Gap widened, not narrowed, opposite of what the "fewer obs points" hypothesis predicted.
I did not expect this at all, I still want to go deep and understand what went wrong, because I know soemthing did

| σ    | curve_fit err | PINN mean err | PINN std |
|      |               |               |          |
| 0.01 | 1.20%         | 2.06%         | 0.25pp   |
| 0.05 | 6.17%         | 10.06%        | 1.21pp   |
| 0.10 | 12.36%        | 33.89%        | 4.19pp   |
| 0.20 | 24.72%        | 48.67%        | 4.46pp   |

Key observation, bimodal failure, not uniform degradation: At σ=0.1 and σ=0.2, individual PINN seed values include values like 0.043, 0.0001, and -0.00004 (nonphysical negative α) alongside seeds landing near-truth (0.098–0.11).
It's not that every seed gets moderately worse, some seeds collapse entirely to ~0 while others still recover correctly. This bimodality is what inflates the mean error, more than a uniform noise-driven degradation would. Again something I want to understand, the reads someway while I hypothesized something completely different.

Mechanistic read: L_data is what breaks the degeneracy where L_pde alone can't pin α (any α admits some reshaped u satisfying the PDE). 
Cutting obs points from 500→50 weakens that anchor. With fewer, noisier points, there's more room for the optimizer to fall into a degenerate solution where α drifts to a nonphysical value while L_pde is still nominally satisfied. curve_fit has no equivalent failure mode, it's plain least-squares against a fixed functional form, no coupled constraint to go degenerate.

Cross-experiment synthesis

Three different levers pulled (misspecify curve_fit, reduce obs count, noise
level was already varied within each) and curve_fit has not lost once. Two
takeaways:

1. curve_fit's advantage isn't primarily about model correctness or obs
   density, it's structurally robust because it's a simpler, better-behaved
   optimization problem (few parameters, convex-ish, no PDE coupling to go
   degenerate).
2. The PINN's failures are catastrophic-seed-driven, not uniform. Both the
   n=3 misspecification run (one seed collapsing at σ=0.2) and the 50-obs run
   (bimodal collapse at σ≥0.1) show this pattern. This points toward an
   optimization/training pathology in the inverse PINN itself, not just "PINNs
   are inherently worse at this task."

Leading suspect: T_max domain mismatch bug (not yet tested)
I flagged it earlier, still remains unfixed: in `inverse_pinn.py`, PDE/BC/IC collocation
points sample t from [0,1) without scaling by T_max, while the data loss correctly scales t_obs by T_max. If the FD solver's T_max is well below 1, this means L_pde/L_bc/L_ic are being evaluated over a much wider effective time domain than L_data ever sees , diluting L_data's relative grip on the network exactly where it needs to constrain α. 
This would directly explain the seed-collapse pattern: when L_data's anchor is diluted, there's more room for a degenerate (α, u) solution to form, and which seeds fall into it looks stochastic (bimodal) rather than uniform.

This is now higher-priority to test than n=2 or further obs-count sweeps because it could be a root cause behind *both* prior failure patterns rather than a third independent lever.

Open threads / next experiments (unranked until we decide)
- [ ] Fix T_max scaling bug, rerun Experiment 3 (50 obs) , cheapest test of the
      leading hypothesis, reuses existing seeds/setup.
- [ ] n=2 nuisance mode misspecification (harder to absorb than n=3) , still
      untested.
- [ ] Seed-level trace: check whether the *same* seeds collapse across
      experiments (init-dependent) or whether it's noise-instance dependent.
- [ ] Combine reduced obs count + misspecification (compounding, not yet tried).
- [ ] Correlated/heteroscedastic noise , not yet tried, lower priority.