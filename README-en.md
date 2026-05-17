# MyMBD — Méthode des Blocs Disjoints "MBD" / Kappa4 & Generalized Rayleigh / SRE-SRX / SDF vibration analysis tool

> **Author**: Guillaume LE ROUSSEAU
> **Documented main program**: `mbd_simple-multi-process_v3_4.py`
> **Current version**: V3.4
> **Language**: Python 3.10+
> **Domain**: vibration testing — mechanical environment tailoring

---

## Table of contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Normative framework and reference documents](#2-normative-framework-and-reference-documents)
3. [Detailed program workflow](#3-detailed-program-workflow)
4. [Technical choices and rationale](#4-technical-choices-and-rationale)
5. [User guide](#5-user-guide)
6. [User parameters — full description](#6-user-parameters--full-description)
7. [Output files](#7-output-files)
8. [Code architecture (internal sections)](#8-code-architecture-internal-sections)
9. [Known pitfalls and precautions](#9-known-pitfalls-and-precautions)
10. [Improvement suggestions](#10-improvement-suggestions)

---

## 1. Purpose and scope

From a measured acceleration signal (typically from a vehicle test, a test rig, or a road-load recording), the program computes **three spectral quantities** that depend on a natural frequency f₀:

- **SRC(f₀)** — Shock Response Spectrum (max of the SDOF response)
- **SRE(f₀)** — Extreme Response Spectrum (upper quantile, MBD Kappa4 method)
- **SDF(f₀)** — Fatigue Damage Spectrum (Basquin law + Miner, rainflow counting)

And their **projection to the target service life** (T_proj typically 36×10⁶ s ≈ 1 year).

The pipeline implements the **uncorrelated MBD** method (Method of the Duration Block) from standard **NF X50-144-3 (2021 edition), Annex C**, with the improvement proposed by **B. Colin (COFREND 2023)** consisting of using a **Kappa4 distribution (Hosking, 4 parameters)** as a single parametric law — Kappa4 degenerates analytically into GUM/FRE/WBN/EV1/EV2/EV3 depending on (k, h), which avoids the standard's LAP decision tree (Figure C.5).

The approach is complemented by:
- a **long-duration projection** using EVT (extreme value theory) for the SRE and CLT (central limit theorem, log-normal law) for the SDF;
- a **K-Means classification** of the excitation blocks to handle non-stationary signals as locally stationary classes ;
- a **strict ASTM E1049-85 rainflow counting** (Downing-Socie 4-point stack algorithm), Numba-accelerated;
- an **analytical SRE/SRX/SDF computation from the PSD** (Welch + analytical integration `∫ Pxx·|Fd|² df` → **Gaussian narrow-band SRE** following **PR NORMDEF 0101 §5.4.2** and **SRX at exceedance risk α** following **§5.4.3 equation [5.2]**; Bendat-Lalanne narrow-band for the SDF) for comparison purposes.

---

## 2. Normative framework and reference documents

### Main references (local PDFs)

| # | Document | Local file | Role |
|---|----------|------------|------|
| **[1]** | **NF X50-144-3 (2021)** — *Demonstration of resistance to mechanical environments, Part 3: Tailoring* | `[NF X50 144-3] 2022.pdf` | Annex C normative framework (MBD method) |
| **[2]** | **B. Colin (KNDS / COFREND 2023)** — *Predictive maintenance of critical equipment embedded on land weapon systems* | `MBD&KAPPA4_ME3E2_B_Colin.pdf` | Kappa4 formulas via analytical L-moments |
| **[3]** | **Clou & Lelan (DGA TT / CFM 2025)** — *Development of statistical methods for vibration analysis — Application to land vehicles* | `2025-12-08_Article_CFM_2025_CLOU_LELAN.pdf` | SSI criteria, Kappa4 limits, generalized Rayleigh |
| **[4]** | **PR NORMDEF 0101 (DGA 2009)** — *Defense Standard — Tailoring of mechanical environment tests* | `_docs-SRX_extract_31-39_prnormdef0101pcemv12versionaste.pdf` | §5.4.2 (SRE), §5.4.3 (SRX) eq. [5.2]/[5.3], fig. 5.3 |
| **[5]** | **B. Colin (DGA, MI0460 / 2008)** — *Definition of a Response Spectrum at exceedance risk (SRX) — Non-asymptotic model* | `_docs-SRX_Colin_mi0460-2008.pdf` | Origin of the non-asymptotic SRX model vs Gumbel/Poisson |

### Key pages to consult

**[1] NF X50-144-3 — Annex C (PDF pages 70–89):**
- p. 71 (Figure C.1): **σ(t) = K · z(t)** — stress/displacement relation, K=1 by convention
- p. 74 (Figure C.3): full MBD synoptic (8 steps)
- p. 76 (§C.3): FOH z-transform — Smallwood recursive formulas
- p. 78–80 (§C.5–C.7): LAP, MMP, KS-M / MSDI / Cunnane (ν=0.4) choices
- p. 84 (§C.9): criterion M > 100 (EVT) / M > 50 (CLT)
- p. 88 (§C.11): **SRE_α(f₀) = 4π²·f₀²·Z_sup,α(f₀)** ; **SDF_α(f₀) = D_c,α**
- p. 89 (Table C.3): analytical L-moment expressions of the LAPs

**[2] Colin 2023:**
- p. 3–4: Basquin (eq 1) and criterion SRE_PGQE^α < SRE_PE^α (eq 3)
- p. 4 (Table 1): risk α according to criticality (10% / 1% / 0.1%)
- p. 10–11: **equations 16–19 and 20.1–20.3** = Kappa4 formulas (g_r, λ₁, λ₂, τ₃, τ₄)
- p. 12–13: **equations 28–34** = inference procedure (k*, h*, α*, ξ*)

**[3] Clou & Lelan 2025:**
- p. 2: definitions of MRS and FDS
- p. 7 (§4.1): SSI criterion (Spectral Similarity Index), recommended threshold 85%
- p. 11 (Figure 10): Kappa4 fluctuations ±135% on a Gaussian signal (identified limitation)

**[4] PR NORMDEF 0101 (DGA 2009):**
- p. 33 (§5.4.2): SRE definition — mean peak over T of the SDOF response in pseudo-acceleration `(2π·f₀)²·z_sup`
- p. 34–35 (§5.4.3): **SRX**, equations [5.2] (non-asymptotic) and [5.3] (SRX/SRE ratio); figure 5.2 (SRX/SRE ratio vs n₀⁺·T)
- p. 36 (fig. 5.3): comparative illustration SRC shock / SRE / SRX(1%) / SRX(99%) of a random vibration

**[5] Colin (MI0460, 2008):**
- discussion of the non-asymptotic SRX model vs asymptotic Gumbel ([5.6]) and Poisson ([5.7]) models — equivalent for n₀⁺·T > 1000

### Secondary references (algorithms)

- **ASTM E1049-85 (2017)** — *Standard Practices for Cycle Counting in Fatigue Analysis* (strict rainflow algorithm, 4-point Downing-Socie). Implemented here in Numba.
- **AFNOR A03-406** — *Cycle counting methods for fatigue analysis* (French rainflow equivalent).
- **Hosking, J.R.M. (1994)** — *The four-parameter kappa distribution*, IBM Journal of Research and Development, 38(3):251–258. Original definition of Kappa4.
- **Bendat, J.S. (1964)** ; **Lalanne, C. (2009)** — *Mechanical Vibration and Shock Analysis*, Vol. 4 *Fatigue Damage*, Wiley/ISTE. Gaussian narrow-band approximation of the spectral SDF.
- **Smallwood, D.O. (1981)** — *An improved recursive formula for calculating shock response spectra*, Shock & Vibration Bulletin 51. Recursive FOH formulas used on p. 76 of standard [1].
- **Cunnane, C. (1978)** — *Unbiased plotting positions — A review*, Journal of Hydrology 37, 205–222. Plotting positions p_i:n = (i−a)/(n+1−2a), a=0.4.
- **Mielke, P.W. (1973)** — Order-6 polynomials for τ₄(τ₃) at fixed h (referenced in [2], Tables 2.1–2.2; replaced here by analytical computation).
- **Lalanne, C. (2002)** — Mechanical Vibration and Shock, Volume 3 (Rayleigh SRX formula)
LALANNE C. (2002) — Mechanical Vibration and Shock, Volume 3: Random Vibration, Hermes Penton, 2002

### Scientific libraries used

- **NumPy / SciPy** — numerical computation (`scipy.signal.lfilter`, `scipy.special.gamma/beta`, `scipy.optimize.fsolve`, `scipy.stats.kappa4`, `scipy.stats.lognorm`)
- **scikit-learn** — unsupervised classification (`KMeans`, `StandardScaler`, `silhouette_score`)
- **Numba** — JIT acceleration of rainflow (disk cache shared between workers)
- **pandas** — CSV I/O and exports
- **plotly** — interactive HTML reports
- **tqdm** — progress bars

---

## 3. Detailed program workflow

The pipeline reproduces **Figure C.3 of NF X50-144-3** ([1] p. 74) with a few adaptations (Kappa4 instead of the LAP tree, parallelization over f₀).

```
┌───────────────────────────────────────────────────────────────────┐
│  INPUT: CSV signal  (t, ẍ)  ─  fs ≈ 12.8 kHz typical              │
└───────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 1. CSV import (auto-encoding)      │  importer_signal_csv()
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 2. Per-block Tb feature extraction │  extraire_caracteristiques()
│    (mean, var, skew, kurt, mav,    │  Split into n_blocs of
│     crest_factor, …)               │  size = round(Tb·fs),
└────────────────────────────────────┘  adjusted to a divisor of N
              │
              ▼
┌────────────────────────────────────┐
│ 3. K-Means classification          │  Optional auto-K (silhouette)
│    Auto fallback if MIN_SAMPLES    │  → clusters[n_blocs] vector
│    not satisfied                   │
└────────────────────────────────────┘
              │
              ▼  for each f0 ∈ [F0_MIN .. F0_MAX] (parallel)
┌────────────────────────────────────┐
│ 4A. SDOF response z(t) — FOH       │  reponse_sdof()
│     Smallwood recursive (std §C.3) │  2nd-order IIR filter
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4B. Pseudo-acceleration            │  contrainte = (2πf₀)²·z(t)
│     for SRC / Z_ext computation    │  (m/s² — used for the SRE)
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4C. Per-block maxima → {Z_ext(i)}  │  SRE branch
│     Per-block rainflow → {D_p(i)}  │  MBD damage branch
│     (on z(t))                      │  ⚠ NF X50-144-3 §C.2
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4bis. IID Quality Gate             │  quality_gate_iid()
│   Spearman lag-1 ρ (on ranks)      │  agreger_quality_gate()
│   + Wald-Wolfowitz runs test       │  → 🟢 GO / 🟡 WARNING /
│   per f₀ (global + per class)      │     🔴 NO-GO (spec §5)
└────────────────────────────────────┘
              │
              ▼  for each class i
┌────────────────────────────────────┐
│ 4D. Kappa4 on Z_ext (SRE)          │  ajuster_kappa4()
│     Kappa4 on D_p   (SDF)          │  → {ξ, α, k, h}
│     Analytical L-moments + fsolve  │  Ref. [2] §28-34
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4E. Cunnane / target quantile      │  Kappa4 PPF at p_eff
│     SRE(f₀,class) = PPF(p_eff)     │  Ref. [1] §C.7, [2] eq. 3
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4F. Class synthesis                │  SRE(f₀) = max_i SRE(f₀,i)
│                                    │  SDF(f₀) = Σᵢ Σⱼ D_p,ij
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 5. Analytical SRE/SDF from PSD     │  calculer_sre_analytique()
│    Welch → ∫|H(f)|²·G_xx df        │  Rayleigh + Bendat
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 6. Long-duration projection        │  M(j) = (T_v/T_b)·Occ(j)
│    SRE  : F_M(x) = F(x)^M          │  EVT  → Ref. [1] §C.9-C.10
│    SDF  : log-normal via CLT       │  CLT  → Ref. [1] Tab. C.2
│           (μ_K4·M, σ²_K4·M)        │  Kappa4 moments
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 7. CSV + JSON sidecar exports      │  Parameter suffixes in
│    8. Plotly HTML reports          │  the column names
└────────────────────────────────────┘
```

---

## 4. Technical choices and rationale

### 4.1 Why Kappa4 rather than the standard's LAP tree?

Standard [1] (Figure C.5) uses a **decision tree** between 7 distributions (GUM, FRE, WBN, WB2P, WB3P, LN2P, LN3P) depending on the sample size n. Paper [2] proposes Kappa4 as a **single distribution** because it degenerates asymptotically:
- k > 0 → EV3 / bounded Weibull family
- k → 0, h = 0 → Gumbel (EV1)
- k < 0 → EV2 / Fréchet family
- h > 0 → generalized Pareto family

This eliminates the LAP discontinuity from the threshold-based choice on n and provides a **unified branch** for both the SRE and the SDF (Kappa4 on the D_p, [1] Table C.2).

**Identified limitation** ([3] Figure 10): Kappa4 fluctuates ±135% relative to the analytical solution on a synthetic Gaussian signal. The program therefore also provides, in parallel, the "Rayleigh" SRX estimate and the Bendat-Lalanne SDF for cross-checking.

### 4.2 Identification method: `direct_pwm_analytic`

Three approaches exist in the literature for identifying (k*, h*, α*, ξ*):
1. **Mielke polynomial** — table of order-6 polynomials for τ₄(τ₃) at fixed h ([2] Tables 2.1–2.2).
2. **Maximum likelihood** — costly and unstable.
3. **Analytical L-moments + fsolve (warm-start)** — *retained in V3*. Computes the theoretical (τ₃, τ₄) via Hosking's g_r functions ([2] eq. 20.1–20.3) then solves the 2×2 system:
```
   τ3_computed(k, h) = τ3_empirical
   τ4_computed(k, h) = τ4_empirical
```
via `scipy.optimize.fsolve`, started at (k=0.1, h=0.1).

**Advantages**:
- no polynomial table to maintain,
- controllable xtol (`KAPPA4_ANALYTIC_XTOL`, default 1.49e-8),
- residual checked a posteriori (rejected if > 1e-6).

### 4.3 SDOF via FOH (First-Order Hold) + lfilter

In accordance with [1] §C.3 (p. 76), the recursive **Smallwood 1981** coefficients are used to convert the continuous signal into the equivalent 2nd-order IIR filter — **exact at the sampling rate** under the FOH assumption (linear interpolation of the excitation between samples), unlike ZOH which underestimates the response above fs/10.

Using `scipy.signal.lfilter` with initial conditions `lfilter_zi * y[0]` avoids the start-up transient.

### 4.4 SDF rainflow: strict ASTM E1049, no rearrangement

The `rearrange=True` option (prior rearrangement from the absolute peak, DSF / MIL-STD-810 type) artificially closes the major cycle and **overestimates the damage** by a few percent — all the more so as b is large. **strict ASTM E1049** (`rearrange=False`), equivalent to the `iamlikeme/rainflow` package bit-for-bit. Unclosed residuals are counted as **half-cycles** (n_i = 0.5).

Adopted convention: **amplitude** (σ_a = range/2), consistent with Basquin in the form `N · σ_aᵇ = C`.

### 4.5 SDF applied to z(t), not to the pseudo-acceleration

[1] §C.2 p. 71, Figure C.1:
- σ(t) = K · z(t), K = 1 by convention (not K = (2πf₀)²)
- The rainflow must therefore be applied to **z(t)** (relative displacement, in m).


### 4.6 Damage granularity = block T_b

V2 computed `Σ_classes max(D_bloc)^b` — incorrect: Miner is additive. Sums the D_p,bloc then fits Kappa4 on the **distribution of the D_p** (cf. [1] Table C.2 p. 85). The long-duration projection uses μ_K4·M and σ²_K4·M (CLT → log-normal).

### 4.7 `multiprocessing.shared_memory` parallelization

The loop over f₀ is embarrassingly parallel. The excitation signal (typically >10 M points × 8 bytes = 80 MB) is placed in `shared_memory` to avoid serializing it for each task. The Numba JIT cache is precompiled before the fork so it is shared between workers.

### 4.8 Analytical PSD SRE & SRX — why two α levels

The PSD branch (`calculer_sre_analytique`) implements the two Gaussian narrow-band analytical estimators of [4] PR NORMDEF 0101 §5.4.2-5.4.3:

- **SRE** (§5.4.2) — `(2π·f₀)²·z_eff·√(2·ln(n₀⁺·T))` — mean peak expected over T (limit α → 1/(n₀⁺·T) of [5.2]).
- **SRX(α)** (§5.4.3 eq. [5.2]) — `(2π·f₀)²·z_eff·√(-2·ln(1-(1-α)^(1/(n₀⁺·T))))` — peak at explicit exceedance risk α.

Two α levels are computed in parallel to cover the two **distinct engineering use cases** of [4] §5.4.3:

1. **Low α (~1%) — Design**: high envelope. The structure must withstand a peak that the response has a 1% probability of exceeding during T.
2. **High α (~99%) — Comparison vs shock**: if the SRX(99%) of the random signal exceeds the SRC of a shock, it is demonstrated with ≥99% probability that the vibration is more severe than the shock (avoidable shock test, cf. [4] fig. 5.3).

The choice of the **PSD method rather than the time-domain method** (`z_eff = std(z(t))` on a simulated SDOF, as in the `SRX_0.1.py` prototype) is because this branch is *complementary* to the MBD-Kappa4 branch, which is itself already time-domain, per block. The stationary-Gaussian analytical estimator is consistent with [4] §5.4.5 (analytical method) and faster; it converges to the time-domain result when the stationarity assumption holds.

The asymptotic Gumbel ([4] eq. [5.6]) and Poisson ([5.7]) models were replaced by the non-asymptotic model [5.2]: according to [5] Colin (MI0460), they only become comparable to the non-asymptotic model for `n₀⁺·T > 1000`, and Gumbel is moreover very conservative for short durations.

**Opposite convention and automatic alignment**: NORMDEF [4] §5.4.3 uses α as an **exceedance probability**, whereas NF X50-144-3 [1] §C.9 (and `ALFA_PROJECTION` in this code) uses the **non-exceedance probability**. To guarantee a homogeneous comparison between the **projected MBD SRE** and the **projected SRX α_low** over the same duration `T_proj`, the code automatically computes `ALPHA_SRX_LOW = 1 - ALFA_PROJECTION` (e.g. ALFA_PROJECTION=0.9 ⇒ ALPHA_SRX_LOW=0.1, same 10% risk on both sides). Only `ALPHA_SRX_HIGH` remains to be set independently.

### 4.9 BLAS limited to 1 thread per worker

Enforced **before importing numpy** via the environment variables `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc. — otherwise each worker tries to use all cores (oversubscription, 2-5× slowdown).

---

## 5. User guide

### 5.1 Installation

```bash
# Python 3.10+
pip install numpy scipy scikit-learn pandas tqdm plotly numba psutil
```

`psutil` is optional (used to detect physical cores when `N_WORKERS=None`).

### 5.2 Input CSV file format

Two columns: **time (s)** and **acceleration (m/s²)**:
```
... header lines (count = CSV_SKIP_ROWS) ...
0.000000000;0.0345
0.000078125;0.0382
0.000156250;0.0411
...
```
- Delimiter: `;` or `,` (parameter `CSV_DELIMITER`)
- Decimal: auto-detected (`,` or `.`)
- Encoding: auto-detected (utf-8, latin-1, windows-1252, iso-8859-1, cp1252)
- Number of header lines to skip: `CSV_SKIP_ROWS`
- The sampling frequency `fs` is computed as `1/mean(diff(t))`.

### 5.3 Configuration

Edit the constants at the top of the file `mbd_simple-multi-process_v3_4.py` (SECTION 1), then:

```bash
python mbd_simple-multi-process_v3_4.py
```

The program produces an output folder (`OUTPUT_FOLDER`, default `mbd_simple_output/`) containing the CSVs, the parameter JSON sidecar, and the HTML reports.

### 5.4 Execution modes

- **Multiprocess** (default, recommended): `USE_MULTIPROCESS = True`
- **Sequential** (debug, profiling): `USE_MULTIPROCESS = False`

### 5.4bis Statistical fitting law — `LOI_AJUSTEMENT`

Selects the *a priori* law inferred via L-moments for the SRE/SDF quantile:

- `'kappa4'` (default) — Hosking's 4-parameter law, unchanged V3 behavior.
- `'rayleigh_gen'` — **generalized Rayleigh law** (Kundu & Raqab,
  `F(x;α,λ)=(1−e^(−(λx)²))^α`), exact model of Gaussian excitation.
  Fitted via **modified L-moments (MLME, Kundu & Raqab §6 eq. 18-20)**:
  the transform `Y=X²` follows a generalized exponential `GE(α,λ²)` with
  closed-form L-moments (digamma `ψ`); the shape `α` is the root of
  `[ψ(2α+1)−ψ(α+1)]/[ψ(α+1)−ψ(1)] = l₂/l₁` (via `brentq`) and the scale
  `λ = √([ψ(α+1)−ψ(1)]/l₁)`. **Exact analytical PPF**. According to the CFM 2025 article (Clou/Lelan), more
  stable than Kappa4 across degrees of freedom for Gaussian and
  non-Gaussian signals. Family distinct from Kappa4 (the two are opposed in
  the article). The selected law appears in the HTML parameter box and in a
  `loi` column of the fit CSV.

### 5.4ter Demo mode — `mbd_demo_v1.py`

A short program reusing `main()` from the main module. Generates a Gaussian
or non-Gaussian signal (ZMNL Hermite method from the CFM 2025 article §4.3,
driven by `KURTOSIS`/`SKEWNESS`) then runs the full MBD computation with the
same visuals. `python mbd_demo_v1.py` (config at the top of the file).

### 5.4quater Unit tests — `tests_unitaires/`

The `tests_unitaires/` folder contains one test per critical function. Each
test builds data whose exact answer is known in advance (closed-form theory
or controlled construction), calls the function and compares the result to
the reference, then produces a PNG with a PASS/FAIL banner and a plain
explanation (function role, what is proven, success criterion).

Functions covered (one `test_*.py` file each):

| Test | Function verified |
|------|-------------------|
| `test_extraire_caracteristiques.py` | per-block feature extraction (SECTION 4) |
| `test_reponse_sdof.py` | SDOF FOH Smallwood response (SECTION 5) |
| `test_kappa4.py` | analytical L-moment Kappa4 fit (SECTION 6) |
| `test_lmoments.py` | empirical L-moment computation (SECTION 6) |
| `test_rayleigh_gen.py` | generalized Rayleigh MLME law (SECTION 6) |
| `test_sre_analytique.py` | PSD-based SRE / SRX / SDF (SECTION 7) |
| `test_rainflow.py` | ASTM E1049 rainflow counting (SECTION 8) |
| `test_projection.py` | long-duration EVT / CLT projection (SECTION 9) |
| `test_quality_gate_iid.py` | IID Quality Gate (SECTION 9bis) |

The `_th.py` module gathers the shared plotting utilities.

```bash
python tests_unitaires/run_all.py
```

`run_all.py` runs every `test_*.py`, aggregates the results into
`tests_unitaires/_resultats/index.html` (one PNG per test) and returns exit
code `0` if all tests pass, `1` otherwise (usable in continuous
integration).

### 5.4quinquies HTML chart scales

The HTML reports offer, per subplot, interactive
**X-lin / X-log / Y-lin / Y-log** buttons. The parameter box is now a
**collapsible** static block at the top of the page (no longer obscures the chart).

### 5.5 Kappa4 diagnostics

Enable `KAPPA4_DEBUG_ANALYTIC = True` to export a CSV `Kappa4_Debug_*.csv` containing, for each (f₀, class): empirical L-moments, fsolve warm-start, optimizer residuals, g₁/g₂, exit_reason.

### 5.6 Typical performance

On a 36 Ms signal (≈ 5 min at 12.8 kHz), 500 frequencies, 10 workers: **2–5 min** per run with K=1, **5–15 min** with K=8.

---

## 6. User parameters — full description

### 6.1 Input file

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `CSV_FILEPATH` | str | (local path) | Absolute path to the input CSV |
| `CSV_SKIP_ROWS` | int | 10 | Header lines to skip — ≥ 0 |
| `CSV_DELIMITER` | str | `";"` | Delimiter — typically `";"` (FR) or `","` (US) |

### 6.2 SDOF parameters — Ref. [1] §C.3, [2] eq. 3

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `Q` | float | 10 | **Quality factor** Q = 1/(2ξ). Usual range **5–50**; 10 = standard for a mechanical reference of embedded equipment (≈ ξ=5%). |
| `TB` | float | 1.00 | **Block duration T_b** (s). MBD block step. Must be ≫ 1/f_min but ≪ total duration. Range **0.05–10 s**. |

### 6.3 Frequency spectrum

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `F0_MIN` | float | 1 | Min frequency (Hz) — ≥ 1/T_b recommended |
| `F0_MAX` | float | 500 | Max frequency (Hz) — < fs/4 (anti-aliasing) |
| `DELTA_F0` | float | 1 | Frequency step (Hz). Determines `num_f0 = (F0_MAX-F0_MIN)·DELTA_F0 + 1` |

### 6.4 K-Means classification

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `N_CLUSTERS` | int | 1 | Number of a priori classes. **1 = single-class** (no partition). Range 1–10; beyond that, per-class statistics become too sparse. |
| `AUTO_SELECT_K` | bool | False | If True, picks K within `K_RANGE` by maximizing the Silhouette score |
| `K_RANGE` | range | `range(3,9)` | Range tested for optimal K |
| `MIN_SAMPLES_PER_CLUSTER` | int | 40 | If a class < this threshold, K is automatically decremented. Must stay ≥ `MIN_POINTS_KAPPA4` (40) otherwise the Kappa4 fits fail |
| `FEATURE_FLAGS` | dict | (see code) | Booleans to enable/disable each feature: `mean`, `variance`, `skewness`, `kurtosis`, `rms`, `mav`, `crest_factor`, `autocorr_lag1`, `zcr`, `dominant_freq`, `spectral_centroid`, `spectral_spread` |

### 6.5 Target quantile (PPF) — Ref. [1] §C.7, [2] Tab. 1

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `PROBABILITE_CIBLE` | float | 0.51 | Reference probability if Cunnane disabled. Strict range **(0,1)**. The standard recommends α=0.1 (10%, low criticality), 0.01 (medium criticality), 0.001 (high criticality) ⇒ `PROBABILITE_CIBLE = 1 - α` |
| `OPTION_CUNNANE` | bool | True | If True, p_eff = (N - a) / (N + 1 - 2a) instead of PROBABILITE_CIBLE. Recommended by [1] §C.7 |
| `CUNNANE_A` | float | 0.4 | Cunnane constant. **0.4** = hydrological standard (ν=0.4 in the standard). Range 0.0–0.5 |

### 6.6 SRE / SRX from PSD — Ref. [4] §5.4.2 and §5.4.3

Analytical branch for comparison with MBD-Kappa4, computed from the Welch PSD
of the signal and the integration `z_eff² = ∫ Pxx(f)·|Fd(f,f₀,Q)|² df`.
Gaussian narrow-band assumption: `n₀⁺ ≈ f₀` ([4] §5.4.3, right after [5.2]).

- **SRE** (Extreme Response Spectrum) = `(2π·f₀)²·z_eff·√(2·ln(n₀⁺·T))` — special case of [5.2] with α → 1/(n₀⁺·T); mean peak expected over T.
- **SRX(α)** = `(2π·f₀)²·z_eff·√(-2·ln(1-(1-α)^(1/(n₀⁺·T))))` — equation [5.2] of [4] §5.4.3.

Two α levels are computed simultaneously, corresponding to the two engineering use cases documented in [4] §5.4.3:

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `ALPHA_SRX_LOW`  | float | **derived** | **Automatically DERIVED**: `ALPHA_SRX_LOW = 1 - ALFA_PROJECTION`. Guarantees exceedance-risk homogeneity with the projected MBD SRE (which uses the opposite convention *non-exceedance probability*, [1] §C.9). Set via `ALFA_PROJECTION`. Use case: **design** high envelope. |
| `ALPHA_SRX_HIGH` | float | 0.99 | **High** risk (≈ 99%) → **comparison vs SRC of a shock**: if SRX(99%) > SRC, the random vibration is more severe than the shock with ≥99% probability ([4] fig. 5.3). Usual range 0.9–0.999. Independently adjustable. |

### 6.7 SDF — Basquin / Miner law

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `SDF_ENABLED` | bool | True | Enables the SDF computation (rainflow + Bendat). Disable for ~30% gain on very long signals |
| `SDF_B` | float | 8.0 | **Basquin slope b** (N·σ_a^b = C). Usual range for metals **3–14**: aluminum 3-5, steel 5-8, welds 3-4, composites 8-14. b=8 = "hard material" default cf. [2] |
| `SDF_C` | float | 1.0 | **Basquin constant C** — fixed at 1 by default (relative analysis; the absolute SDF value depends on the application) |

### 6.8 Long-duration projection — Ref. [1] §C.8-C.9

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `ENABLE_PROJECTION` | bool | True | Enables the T_v projection |
| `DUREE_PROJECTION` | float | 36_000_000 | T_v (s). 36×10⁶ s ≈ 1 year of use (or 10,000 km at typical mean speed). Range **T_measure × 10² to T_measure × 10⁸** |
| `ALFA_PROJECTION` | float | 0.9 | Projected exceedance risk (1−α in the standard). 0.9 = 90% non-exceeded. Range **0.5–0.999** |

### 6.8bis IID Quality Gate — Ref. specification §3-5

Post-processing validation building block (loose coupling, vectorized
numpy/scipy). Checks the temporal-block independence assumption — required
by Kappa-4 inference via L-moments — before trusting the SRE/SDF. For each
f₀: lag-1 autocorrelation of **Spearman on the ranks** (robust to Kappa-4
extremes) + **Wald-Wolfowitz runs test** (binarization vs the median).
Global computation per f₀ **and** per K-Means class.
Verdict 🟢 GO / 🟡 WARNING (isolated band, often low frequencies) /
🔴 NO-GO (generalized dependence → corrective message requiring an increase
of `TB`). The run is never interrupted: the offending f₀ are tagged with
reduced confidence (visible on hover in the main report).

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `IID_GATE_ENABLED` | bool | True | Enables the check. `False` ⇒ behavior strictly identical to the legacy (no column / section added) |
| `IID_RHO_MAX` | float | 0.2 | Acceptable \|ρ\| Spearman lag-1 threshold. Usual range 0.1–0.3 |
| `IID_PVALUE_MIN` | float | 0.05 | P-value threshold of the runs test. Below this threshold ⇒ rejection of the independence assumption |
| `IID_FAIL_FRAC_MAX` | float | 0.05 | Max fraction of out-of-tolerance f₀ to stay 🟢 GO (5% of the spectrum) |
| `IID_NOGO_FRAC` | float | 0.30 | Beyond this fraction of failing f₀ ⇒ 🔴 NO-GO. Between `IID_FAIL_FRAC_MAX` and this value ⇒ 🟡 WARNING |
| `IID_MIN_N` | int | 20 | Minimum sample size for a test to be deemed reliable (otherwise metrics NaN, column ignored in the verdict) |

### 6.9 Numerical / runtime

| Parameter | Type | Default | Range / Notes |
|-----------|------|---------|---------------|
| `RANDOM_SEED` | int | 53 | RNG seed (KMeans, etc.) — reproducibility |
| `MIN_POINTS_KAPPA4` | int | 40 | Min threshold to attempt a Kappa4 fit. < 40 → fit refused (L-moments instability). [1] §C.9 requires at least n=50 for LAR to converge |
| `OUTPUT_FOLDER` | str | `"mbd_simple_output"` | Output folder (created if absent) |
| `USE_MULTIPROCESS` | bool | True | Enables the multiprocess pool on the f₀ loop |
| `N_WORKERS` | int / None | 10 | Number of workers. `None` = `psutil.cpu_count(logical=False)` or `os.cpu_count()-1` |
| `KAPPA4_METHOD` | str | `'direct_pwm_analytic'` | Only method available in V3 |
| `KAPPA4_ANALYTIC_XTOL` | float | 1.49e-8 | fsolve tolerance. Range 1e-12 to 1e-4. Tighter = more precise but slower |
| `KAPPA4_DEBUG_ANALYTIC` | bool | False | If True, exports the Kappa4_Debug_*.csv diagnostic CSV |

---

## 7. Output files

All in `OUTPUT_FOLDER`, suffixed with a **descriptive tag** (e.g. `v3p2_f1-500_Q10_Tb1p28_b8_T36Ms_a0p9_20260503_155038`) that encodes all physically significant parameters.

| File | Content |
|------|---------|
| `params_*.json` | JSON sidecar: all run parameters (build_run_meta) |
| `SRE_*.csv` | SRC, Kappa4 SRE, PSD SRE (NORMDEF §5.4.2), SRX α_low, SRX α_high (NORMDEF §5.4.3 eq. [5.2]), + IID Quality Gate columns (ρ lag-1, runs p-value, status) per frequency |
| `SDF_*.csv` | Global rainflow SDF, MBD-AnnexC SDF (Σ D_bloc), spectral Bendat SDF, mean Kappa4-Dmg RMSE/MSDI, + damage-branch IID Quality Gate columns |
| `SDF_Kappa4_Fit_*.csv` | Kappa4 fit diagnostics on D_bloc: ξ, α, k, h, t3, t4, RMSE, MSDI per (f₀, class) |
| `IID_QualityGate_*.csv` | (if `IID_GATE_ENABLED`) IID diagnostics per (f₀, class) — `Classe=-1` = global all classes: Spearman lag-1 ρ, runs-test p-value, n, status, SRE and SDF branches |
| `SRE_Projection_*.csv` | Projected SRE, M, projected SDF (empirical CLT / K4-LogN / PSD) |
| `Kappa4_Debug_*.csv` | (optional) fsolve trace for each fit |
| `Rapport_*.html` | Main Plotly report: 4 charts (SRC+SRE / 5-curve comparison / SDF / Projection); IID status in the parameter box, IID confidence on SRE hover |
| `Rapport_Details_*.html` | Diagnostic report: τ₃/τ₄, CDF per class (f₀ dropdown), SRE+MSDI comparison, projected SDF comparison, + IID ρ/p-value spectra vs f₀ |

**CSV column suffix convention**: each quantity carries the parameters on which it physically depends.
- SRC depends on Q, T_b → `SRC_Q10_Tb1p28s`
- Kappa4 SRE depends on Q, T_b, P_target, T_meas → `SRE_Kappa4_Q10_Tb1p28s_P0p51_Tmes281p25s`
- PSD SRE (NORMDEF §5.4.2) depends on Q, T_b, T_meas → `SRE_DSP_NormDef_Q10_Tb1p28s_Tmes281p25s`
- SRX α_low / α_high (NORMDEF §5.4.3) add the risk α → `SRX_alpha_low_Q10_Tb1p28s_aL0p01_Tmes281p25s` / `SRX_alpha_high_Q10_Tb1p28s_aH0p99_Tmes281p25s`
- SDF depends on b, Q, T_b, T_meas → `SDF_Temporel_Rainflow_b8_Q10_Tb1p28s_Tmes281p25s`
- Projected SRE adds T_proj, α → `SRE_Projection_Q10_Tb1p28s_P0p51_T36Ms_a0p9`
- IID Quality Gate depends on Q, T_b (SDOF response) → `IID_rho_lag1_Q10_Tb1p28s`

This allows **stacking CSVs from runs with different parameters** without name collisions.

---

## 8. Code architecture (internal sections)

The file `mbd_simple-multi-process_v3_4.py` is a self-contained single file, organized into 13 numbered sections:

| Section | Lines (approx) | Role |
|---------|----------------|------|
| 1 | CONFIGURATION | User constants (edit here before launch) |
| 2 | LOGGING | `logging.basicConfig` |
| 3 | CSV IMPORT | `importer_signal_csv()` — auto-encoding |
| 4 | FEATURES & MAXIMA | `extraire_caracteristiques()`, `_trouver_diviseurs()` |
| 5 | SDOF FOH RESPONSE | `reponse_sdof()` — Smallwood recursive (std §C.3) |
| 6 | ANALYTICAL KAPPA4 | `_calculer_pwm`, `calculer_lmoments`, `_g_functions`, `_fit_loc_scale`, `_tau3_tau4_from_kh_analytic`, `ajuster_kappa4_pwm_analytic`, `kappa4_rmse_msdi`, `kappa4_ppf` |
| 7 | PSD SRE / SDF | `calculer_sre_analytique()` — Welch + Rayleigh + Gumbel + Bendat |
| 8 | SDF RAINFLOW | `_rainflow_extrema_numba`, `_rainflow_stack_damage_numba`, `_rainflow_damage`, `calculer_sdf_rainflow`, `calculer_sdf_per_bloc` |
| 9 | L-MOMENTS PROJECTION | `calculer_projection_lmoments`, `calculer_projection_sdf_tcl`, `calculer_projection_dmg_kappa4`, `_kappa4_mean_var` |
| 9bis | IID QUALITY GATE | `quality_gate_iid`, `_iid_spearman_lag1`, `_iid_runs_pvalue`, `_iid_verdict`, `agreger_quality_gate` — loose coupling, vectorized |
| 10 | f₀ PROCESSING | `traiter_f0()` — full pipeline for one frequency; multiprocessing workers |
| 11 | CSV EXPORTS | `exporter_csv_sre/sdf/projection/sdf_kappa4_fit/iid/debug_analytic`, suffixing helpers |
| 12 | HTML REPORTS | `generer_html()`, `generer_html_details()` |
| 13 | MAIN | `main()`, `main_mp()`, entry point |

