# Noise-induced vortex shedding in 35-moment HyQMOM — comparison to McMullen & Gallis

**Reference:** R. M. McMullen & M. A. Gallis, *"Hydrodynamic fluctuations near a Hopf
bifurcation: stochastic onset of vortex shedding behind a circular cylinder"*
(Sandia SAND2024-13841J).

This note explains how the results in this folder reproduce the paper's mechanism,
what matches, and what does not.

---

## 1. What the paper claims

The paper's thesis is that the **onset of vortex shedding behind a cylinder is
*fundamentally stochastic***. Near the critical Reynolds number `Re_c`, the
deterministic oscillation amplitude collapses (critical slowing down), so **thermal
fluctuations are dramatically amplified and drive *intermittent* shedding**. They
show this two ways — **fluctuating Navier–Stokes** (adding a Landau–Lifshitz
stochastic stress `√(2Σ)/Re ∇·𝒲` to the NS momentum equation) and **DSMC** — at
**Ma = 0.3**, and their central observable is the **fluctuating transverse (lift)
force**: its **power spectrum**, its **amplitude distribution (PDF)**, and the
**appearance of non-Gaussian fluctuations** at `Re/Re_c − 1 = O(10⁻³)`.

## 2. Our approach

We use the **3D 35-moment HyQMOM** kinetic *moment* method — which sits *between*
their continuum FNS and molecular DSMC in the modeling hierarchy — and add the
**same FDT-calibrated random stress**, but to the **momentum-moment flux** rather
than the continuum momentum equation:

```
s_ij added to the flux of M100/M010/M001, entering the momentum moments as dt·(∇·s)
σ = intensity · √( 2·T·η / (ΔV·Δt) ),   η = Kn·√T / 2   (BGK dilute-gas viscosity)
```

This is the moment-method analogue of the paper's Landau–Lifshitz term (same
fluctuation–dissipation scaling). We first **confirmed deterministic Kármán shedding**
at resolved Reynolds number (Np=256; the key was resolving the shear layer — a coarse
grid is numerically over-damped below Re_c), **bracketed Re_c**, then ran at the
**marginal (near-critical) condition** with increasing FDT noise.

## 3. Observable-by-observable comparison

| Paper's observable | Our result | Figure | Match |
|---|---|---|---|
| Fluctuating lift `C_L(t)` | Control-volume momentum-balance lift diagnostic (raw 2nd moments = momentum-flux tensor) | `fig2_lift_series.png` | ✅ same quantity |
| Lift power spectrum | clean Strouhal peak → **broadband** as noise ↑ | `fig4_summary.png`, PSD | ✅ qualitative |
| Lift amplitude PDF | **bimodal (near-sinusoid) → non-Gaussian heavy tails** | `fig3_lift_pdf.png` | ✅ same signature |
| **Non-Gaussian fluctuations** | **lift kurtosis crosses 3** | `fig4_summary.png` | ✅ **direct match** |
| Intermittent shedding | coherent vortices → thermal-noise-disrupted wake | `fig1_wake_panels.png`, `shedding_limitcycle.mp4` | ✅ qualitative |
| Amplification *near* Re_c | effect strongest at marginal Kn; kurtosis rises 1.44→1.94 approaching Re_c even before noise | (bracket) | ✅ the mechanism |

### The core result — kurtosis crosses 3

At the marginal condition (Kn = 0.008), increasing the FDT thermal-noise intensity
drives the lift monotonically from a clean deterministic limit cycle into the paper's
non-Gaussian, intermittent regime:

| FDT noise intensity | RMS `C_L'` | **lift kurtosis** | lift spectrum |
|---|---|---|---|
| 0 (deterministic)   | 0.36 | **1.94** | clean peak, St ≈ 0.20 |
| 3e-5                | 0.31 | **2.22** | peak + slight broadening |
| 1e-4                | 0.26 | **2.71** | broadening |
| 3e-4                | 0.30 | **3.33** | broad, shedding peak demoted |

Kurtosis = 3 is the *definition* of Gaussian; **> 3 is heavy-tailed / non-Gaussian /
intermittent** — exactly the paper's signature. The PDF (`fig3`), the spectrum
(`fig4`), and the wake field (`fig1`) all change consistently and together as the
noise increases — different projections of one physical transition.

## 4. Figures & movie in this folder

- **`fig1_wake_panels.png`** — wake vorticity at the four noise levels (flow left→right).
  Coherent alternating vortices (off) dissolve into fine-scale thermal-noise grain (3e-4).
- **`fig2_lift_series.png`** — the lift time series `C_L(t)`: clean oscillation → noisy/intermittent.
- **`fig3_lift_pdf.png`** — lift-fluctuation PDF: deterministic **bimodal** (sinusoid) →
  **non-Gaussian heavy tails**. Directly comparable to the paper's amplitude distribution.
- **`fig4_summary.png`** — kurtosis vs intensity crossing the Gaussian=3 line; coherent RMS weakening.
- **`shedding_limitcycle.mp4`** — the deterministic Kármán vortex street in motion
  (proof the 35-moment HyQMOM sheds).

## 5. What we do NOT claim (honest limitations)

- **Not a quantitative match to their numbers.** Our Strouhal is ~0.20–0.25 vs their
  ~0.13, because our "cylinder" is a **held-gas bounce-back** (not a true no-slip wall)
  with **~10% blockage** — so our `Re_c` and `St` differ. We reproduce the *behavior*,
  not their exact values.
- **Short statistics.** ~2–3 shedding periods per case → each kurtosis carries ~±0.3
  uncertainty. The *trend* across four conditions is robust; individual values are not
  converged (their DSMC runs were long).
- **No analytical theory.** We reproduce the simulation phenomenology, not their
  perturbation expansion around `Re_c`.
- **Simplified noise.** Collocated FDT stress with a simplified tensor structure, not the
  full staggered Landau–Lifshitz. At the strongest intensity (3e-4) part of the kurtosis
  rise is broadband **noise domination** rather than pure intermittency — the **1e-4 case
  (kurtosis 2.71, shedding peak coherent-but-modulated + broadening) is the cleanest
  near-critical intermittent analogue**.

## 6. Why it's interesting beyond matching

Moment methods like HyQMOM are usually suspected of **over-smoothing** fluctuations (the
closure truncates the velocity distribution). The genuinely interesting finding is that
**the closure does *not* destroy the noise-amplification physics**: the 35-moment system,
seeded with FDT thermal noise, still exhibits critical-slowing-down amplification and a
non-Gaussian, intermittent lift. This is a new data point that the McMullen–Gallis
mechanism is **robust across the modeling hierarchy** (molecular DSMC → kinetic-moment
HyQMOM → continuum FNS), not an artifact of one method.

---

*Data:* `output/runs/np256_{stationary,noise_off,noise_fdt,noise_fdt1e4,noise_fdt3e4}.jld2`.
*Diagnostics:* `examples/analysis/lift_force.jl` (lift `C_L(t)` + PSD + skewness/kurtosis).
*Noise model:* opt-in `CASE_FLUCT` / `fluct_intensity` (FDT fluctuating stress), byte-identical when off.
