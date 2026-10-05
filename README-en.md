# MyMBD — Disjoint Block Method ("MBD") / Kappa4 & Generalized Rayleigh / SRE-SRX / SDF vibration analysis tool

> **Author**: Guillaume LE ROUSSEAU
> **Documented main program**: `mbd_simple-multi-process_v3_6.py`
> **Current version**: V3.6
> **Language**: Python 3.10+
> **Domain**: vibration testing — mechanical environment tailoring

> **Acronyms kept in French, as in the program and its outputs**:
> MBD = *Méthode des Blocs Disjoints* (disjoint block method) ·
> SRC = shock response spectrum · SRE = extreme response spectrum ·
> SRX = response spectrum at exceedance risk · SDF = fatigue damage spectrum ·
> 1-DDL = single degree of freedom (SDOF) · TCL = central limit theorem ·
> IID = independent and identically distributed.


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


---

## 1. Purpose and scope

From a measured acceleration signal (vehicle test, test rig, road-load recording), the program computes **three spectral quantities** as functions of a natural frequency f₀:

- **SRC(f₀)** — shock response spectrum: maximum of the response of a single-degree-of-freedom (SDOF) oscillator;
- **SRE(f₀)** — extreme response spectrum: upper quantile of the per-block maxima, MBD method;
- **SDF(f₀)** — fatigue damage spectrum: Basquin law, Miner summation, rainflow counting.

It also provides their **projection to the target service life** (T_proj, default 36×10⁶ s = 10,000 h).

The computation applies the uncorrelated **Disjoint Block Method (MBD)** of standard **NF X50-144-3 (2021), Annex C**, with the contribution of **B. Colin (COFREND 2023)** [2]: a **4-parameter Kappa4 distribution (Hosking)** is used as a single law, instead of the standard's decision tree between 7 laws. A second law is available: the **2-parameter generalized Rayleigh** (A. Clou & P. Lelan, DGA TT / CFM 2025) [3], with a narrower domain but a more stable estimate from one frequency to the next.

Around this core, the program adds:

- a **long-duration projection**: extreme value theory for the SRE (F^M), central limit theorem for the SDF;
- a **K-Means classification** of the excitation blocks, to handle a non-stationary signal as locally stationary classes, and the **synthesis** of those classes;
- two **validity checks** whose outcome is explained in plain language: stationarity of the signal, and independence of the blocks (IID contract) required by the method;
- a **strict ASTM E1049-85 rainflow counting**, accelerated with Numba;
- an **analytical comparison branch** from the power spectral density: SRE and SRX of PR NORMDEF 0101, Bendat-Lalanne SDF.

---

## 2. Normative framework and reference documents

### Main references

| # | Document | Local file | Role |
|---|----------|------------|------|
| **[1]** | **NF X50-144-3 (2021)** — *Demonstration of resistance to mechanical environments, Part 3: Tailoring* | not in the repository (AFNOR document) | Normative framework, Annex C (MBD method) |
| **[2]** | **B. Colin (Nexter Systems / COFREND 2023)** — *Maintenance prévisionnelle des équipements critiques, embarqués sur systèmes d'armes terrestres*, e-Journal of NDT, doi:10.58286/28496 | `_docs_MBD-Kappa4_MBD&KAPPA4_ME3E2_B_Colin.pdf` | Kappa4 law via L-moments, projection, stochastic synthesis |
| **[3]** | **A. Clou & P. Lelan (DGA TT / CFM 2025)** — *Development of statistical methods for vibration analysis* | `_docs_MBD-Rayleigh_ASTE_2025-12-08_Article_CFM_2025_CLOU_LELAN.pdf` | Kappa4 limitations, generalized Rayleigh law |
| **[4]** | **PR NORMDEF 0101 (DGA 2009)** — *Tailoring of mechanical environment tests* | `_docs-SRX_prnormdef0101pcemv12versionaste.pdf` (excerpt p. 31-39: `_docs-SRX_extract_31-39_…pdf`) | SRE (§5.4.2) and SRX (§5.4.3, eq. [5.2]) |
| **[5]** | **B. Colin (MI0460, 2008)** — *Definition of a response spectrum at exceedance risk (SRX)* | `_docs-SRX_Colin_mi0460-2008.pdf` | Origin of the non-asymptotic SRX model |
| **[6]** | **Kundu & Raqab** — *Generalized Rayleigh Distribution: Different Methods of Estimations* | `_docs_MBD-Rayleigh_KUNDUetRAQABpaper96.pdf` | Generalized Rayleigh estimation by modified L-moments |
| **[7]** | **W. Asquith** — *Distributional Analysis with L-moment Statistics* | `_docs-Distributional_Analysis_with_L-moment_Statistics_…pdf` | Reference book on L-moments |

### Key pages

**[1] NF X50-144-3 — Annex C:**
- §C.2 (Figure C.1): σ(t) = K · z(t), K = 1 by convention;
- §C.3: SDOF response by Smallwood's recursive formulas;
- §C.5–C.7: per-block samples, goodness-of-fit tests, Cunnane plotting position (ν = 0.4), MSDI;
- §C.8–C.9: extrapolation coefficient M, criteria M > 100 (extreme values) and M > 50 (central limit theorem);
- §C.10: stochastic synthesis of the classes.

**[2] Colin 2023:**
- eq. 1: Basquin law N·σᵇ = C; Table 1: risk α by criticality (10% / 1% / 0.1%);
- eq. 8–15 and 24–27: L-moments, probability-weighted moments, unbiased estimators;
- eq. 16–20: Kappa4 L-moments through the g_r functions; Figure 11: validity domain of (k, h);
- eq. 28–34: fitting procedure (h*, k*, α*, ξ*); eq. 30–31: distribution and quantile functions;
- eq. 35–36: move to the global model (F^M, damage sum);
- §4.2, eq. 37–38: K-Means classification and stochastic synthesis of the classes.

**[4] PR NORMDEF 0101:** §5.4.2 (SRE, mean peak over T), §5.4.3 (SRX, equations [5.2] and [5.3], figures 5.2 and 5.3).

### Secondary references (algorithms)

- **ASTM E1049-85 (2017)** and **AFNOR A03-406** — rainflow counting.
- **Hosking, J.R.M. (1994)** — *The four-parameter kappa distribution*, IBM J. Res. Dev. 38(3):251–258.
- **Hosking & Wallis (1997)** — *Regional frequency analysis: an approach based on L-moments*, Cambridge University Press.
- **Smallwood, D.O. (1981)** — *An improved recursive formula for calculating shock response spectra*, Shock & Vibration Bulletin 51.
- **Cunnane, C. (1978)** — *Unbiased plotting positions — A review*, J. Hydrology 37.
- **Lalanne, C.** — *Mechanical Vibration and Shock*, vol. 3 (*Random Vibration*) and vol. 4 (*Fatigue Damage*).

### Libraries used

- **NumPy / SciPy** — numerical computation (`lfilter`, `welch`, `fsolve`, `brentq`, `scipy.stats.kappa4`, `lognorm`);
- **scikit-learn** — `KMeans`, `StandardScaler`, `silhouette_score`;
- **Numba** — rainflow compilation;
- **pandas** — CSV reading and writing;
- **plotly** — interactive HTML reports;
- **tqdm** — progress bars; **psutil** (optional) — number of physical cores;
- **matplotlib** — unit tests only.

---

## 3. Detailed program workflow

```
┌───────────────────────────────────────────────────────────────────┐
│  INPUT: CSV signal  (t, ẍ)                                         │
└───────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 1. CSV import                      │  importer_signal_csv()
│    encoding and decimal detected   │  TRIM_DEBUT_S: trims the start
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 1bis. Stationarity check           │  detecter_non_stationnarite()
│    f₀ probes + per-block RMS CV    │  → STATIONNAIRE / NON_STATIONNAIRE
│    + Gaussian character            │    / DOUTEUX, recommendation
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 2. Split into T_b blocks           │  extraire_caracteristiques()
│    block = round(T_b·fs) samples   │  _taille_bloc()
│    + per-block features            │  incomplete signal tail ignored
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 3. K-Means classification          │  K = 1 if the signal is stationary
│    of the excitation blocks        │  otherwise Silhouette score
│                                    │  → clusters[n_blocs]
└────────────────────────────────────┘
              │
              ▼  for each f₀ ∈ [F0_MIN .. F0_MAX]  (parallel)   traiter_f0()
┌────────────────────────────────────┐
│ 4A. SDOF response z(t)             │  reponse_sdof()
│     recursive Smallwood, primed    │  2nd-order IIR filter
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4B. Pseudo-acceleration (2πf₀)²·z  │  SRC = max over the signal
│     Per-block maxima → {Z_max}     │  SRE branch
│     Per-block rainflow → {D_p}     │  damage branch (on z)
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4C. IID contract                   │  quality_gate_iid()
│     lag-1 Spearman ρ               │  global and per class
│     + runs test                    │
└────────────────────────────────────┘
              │
              ▼  for each class
┌────────────────────────────────────┐
│ 4D. Law fitting                    │  ajuster_loi()
│     L-moments → Kappa4 (2 steps)   │  kappa4_from_lmoments()
│     or generalized Rayleigh        │  ajuster_rayleigh_gen()
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4E. Per-class quantile             │  loi_ppf() at Cunnane's p
│     + quality: RMSE, MSDI          │  loi_rmse_msdi()
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4F. Class synthesis                │  SRE = quantile of Π F_j^{N_j}
│     (signal duration)              │  SDF = Σ per-block damages
│                                    │  excluded classes recorded
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4bis. IID contract verdict         │  agreger_quality_gate()
│                                    │  → GO / WARNING / NO-GO
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 5. Analytical SRE / SRX / SDF      │  calculer_sre_analytique()
│    from the spectral density       │  Welch + NORMDEF + Bendat
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 6. Long-duration projection        │  M_j = Occ(j)·T_proj / T_b
│    SRE: quantile of Π F_j^{M_j}    │  quantile_produit_classes()
│    SDF: sum of M blocks (CLT)      │  calculer_projection_sdf_tcl()
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 7. CSV + JSON exports              │  SECTION 11
│ 8. HTML reports                    │  SECTION 12, with the explained
│                                    │  "Contrat IID" block
└────────────────────────────────────┘
```

---

## 4. Technical choices and rationale

### 4.1 Why Kappa4 rather than the standard's tree of laws?

Standard [1] picks among 7 laws depending on the sample size. Paper [2] proposes Kappa4 as a **single law**: it contains the classical laws as special cases (h = −1 generalized logistic, h = 0 generalized extreme value GEV, h = 1 generalized Pareto) and covers a large part of the L-moment diagram (τ3, τ4). The same treatment serves the maxima (SRE) and the per-block damages (SDF).

**Known limitation** ([3], Figure 10): on a Gaussian signal, the Kappa4 SRE fluctuates strongly from one frequency to the next. The program therefore also provides the generalized Rayleigh law and the analytical branch (SRX, Bendat) for cross-checking — see also §9.

### 4.2 Kappa4 identification: two steps and domain check

Function `kappa4_from_lmoments`. The input is the quadruplet (L1, L2, τ3, τ4) computed on the sample from the unbiased probability-weighted moments ([2] eq. 24-27).

**Step 1 — the shape (k, h)** depends only on the ratios τ3, τ4. The system

```
τ3(k, h) = measured τ3        τ4(k, h) = measured τ4
```

is solved, where τ3(k, h) and τ4(k, h) are the analytical expressions of [2] (eq. 18-19) through Hosking's g_r functions (eq. 20.1 to 20.3), using `scipy.optimize.fsolve`. This is the continuous equivalent of the paper's procedure (eq. 28: minimum distance over a family of polynomials at discrete h): inside the domain the minimum distance is zero and both procedures coincide. Three safeguards:

1. `fsolve` convergence **and** recomputed residual below `KAPPA4_RESIDUAL_TOL`, otherwise retry from other starting points (`KAPPA4_RETRY_WARM_STARTS`);
2. **h domain**: a root h < `KAPPA4_H_MIN` (−1) is rejected — below that bound, several (k, h) pairs reproduce the same (τ3, τ4) with different tails;
3. **point outside the domain**: if (τ3, τ4) lies above the h = −1 curve, of equation τ4 = (1 + 5τ3²)/6, no admissible Kappa4 reproduces it. The retained law is the **boundary** one: h* = −1, at the point of the curve **nearest in Euclidean distance** ([2] eq. 28-29), obtained in closed form (root of a cubic). The case is tagged `fail_reason = 'ok_h_borne'`.

**Step 2 — location and scale** follow in closed form ([2] eq. 33-34):

```
α = k·L2 / (g1 − g2)          ξ = L1 − (α/k)·(1 − g1)
```

with the k → 0 limit (expansion of the g_r) for the laws on the k = 0 line (Gumbel, logistic, exponential).

The quantile and distribution functions are those of [2] (eq. 30-31). `scipy.stats.kappa4` is used everywhere except near k = 0 with h ≤ 0, where it returns NaN: the program then switches to its own exact formulas.

### 4.3 SDOF response: Smallwood and priming

In accordance with [1] §C.3 and [2] §4.1, the response z(t) is computed with **Smallwood's (1981)** recursive filter, exact for an excitation that is linear between two samples.

**Priming** (`SDOF_AMORCAGE`, on by default). The natural initial state of a recursive filter is that of a constant excitation equal to the first value of the signal: the oscillator then starts from a static displacement and releases a spurious free oscillation, of amplitude ≈ |x[0]| in pseudo-acceleration, decaying with the time constant τ = Q/(π·f₀) (0.64 s at 5 Hz for Q = 10). At low frequencies this artefact sets the maximum of the first block, hence the SRC. With priming, the filter is first run over the time-reversed start of the signal for `SDOF_AMORCAGE_N_TAU` time constants; the output keeps the length of the input.

### 4.4 Rainflow: strict ASTM E1049

Stack-based counting, equivalent to the Downing-Socie 4-point algorithm. Unclosed residuals count as half-cycles. **Amplitude** convention: σ_a = range/2, consistent with Basquin in the form N·σ_aᵇ = C.

### 4.5 Damage is computed on z(t)

[1] §C.2: σ(t) = K·z(t), K = 1. The rainflow is applied to the **relative displacement** z(t), not to the pseudo-acceleration (reserved for the SRC and SRE).

### 4.6 Damage granularity: the T_b block

The rainflow is run inside each block: this is the variable D_p of the method ([2] §4.1). The MBD SDF is the **sum** of the D_p ([2] eq. 36). It is slightly lower than the rainflow over the whole signal, which also sees the large cycles spanning several blocks.

### 4.7 Parallelization

The loop over f₀ is spread over `N_WORKERS` processes. The signal is placed in shared memory (`multiprocessing.shared_memory`) so that it is not copied for each task.

### 4.8 Analytical SRE and SRX: two risk levels

The `calculer_sre_analytique` branch integrates the spectral density (Welch) against the oscillator's transfer function, then applies [4]:

- **SRE** (§5.4.2): `(2π·f₀)²·z_eff·√(2·ln(n₀⁺·T))`;
- **SRX(α)** (§5.4.3, eq. [5.2]): `(2π·f₀)²·z_eff·√(−2·ln(1 − (1 − α)^(1/(n₀⁺·T))))`, with n₀⁺ ≈ f₀.

Two α levels are computed: low α for **design** (upper envelope), high α (0.99) for **comparison with a shock**.

**Opposite conventions**: [4] uses α as an *exceedance* probability, [1] and `ALFA_PROJECTION` as a *non-exceedance* probability. The code aligns both: `ALPHA_SRX_LOW = 1 − ALFA_PROJECTION`.

This branch assumes a **stationary Gaussian** signal. It is the reference in that case, and underestimates the tail otherwise.

### 4.9 One BLAS thread per process

The variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc. are set to 1 **before** numpy is imported, otherwise each process would try to use every core.

### 4.10 Long-duration projection of the SRE

The maximum over the service life is the largest of M block maxima assumed independent ([2] eq. 35): F_Zsup = F_Zmax^M, with M = Occ(j)·T_proj/T_b.

1. **`'puissance'`** (default): projected SRE = quantile of the fitted law at probability α^(1/M). This is the direct application of eq. 35.
2. **`'gev_domaines'`**: the law is re-fitted in the GEV family (h fixed to 0) and projected by max-stability, in closed form. The sign of k designates the domain — Gumbel (k ≈ 0), Fréchet (k < 0), negative Weibull (k > 0) — exported in the `GEV_domaine` column. See the reservation in §9.

### 4.11 Block splitting and the coefficient M

A block **always** holds `round(T_b·fs)` samples (`_taille_bloc`). If the signal length is not a multiple of it, the incomplete tail — less than one block — is ignored. The block duration is a datum of the method: it sets M = T_proj / T_b and must not depend on the signal length. The effective duration (integer size / fs) is the one used in M; it is written to the log, to the report parameter box and to `params_*.json` (`Tb_effectif_s`, `n_blocs`).

### 4.12 Stationarity check

Computed before the frequency loop (`detecter_non_stationnarite`). The method assumes block maxima that are independent and share the same law; two causes of violation call for **opposite** remedies:

- T_b too short compared with the oscillator's memory → **lengthen T_b**;
- slow envelope of the signal (regime change, drift) → **classify the blocks**.

Three indicators:

| Indicator | Measure | Threshold |
|---|---|---|
| (a) probes | fraction of probe frequencies whose per-block maxima fail the IID contract; the probes are placed at f₀ ≥ 3·Q/(π·T_b), where the oscillator's memory alone cannot explain a failure | `STATIONNARITE_FRAC_ECHEC` |
| (b) energy | coefficient of variation of the excitation RMS from block to block | `STATIONNARITE_CV_RMS_MAX` |
| (c) shape | kurtosis and skewness of the signal (Gaussian character) | `STATIONNARITE_KURT_TOL`, `STATIONNARITE_SKEW_TOL` |

Verdict: **NON_STATIONNAIRE** if (a) and (b) both exceed their threshold; **STATIONNAIRE** if neither does; **DOUTEUX** (doubtful) otherwise. The verdict informs the user; the only automatic decision is K = 1 when `AUTO_SELECT_K` and `AUTO_K_SELON_STATIONNARITE` are on and the verdict is STATIONNAIRE (the Silhouette score, undefined for K = 1, would otherwise split a homogeneous signal).

### 4.13 Class synthesis

The classes are situations occurring in series ([2] Figure 15). The maximum over the duration is the largest of the maxima of all classes:

```
P(Z_sup ≤ z) = Π_j F_j(z)^{M_j}                    ([1] §C.10, [2] eq. 37)
```

The α quantile of this product is obtained by solving Σ_j M_j·ln F_j(z) = ln α (`quantile_produit_classes`, `_quantile_produit`), with an ln F that stays accurate near 1 (`_loi_logcdf`).

- **In projection**: M_j = Occ(j)·T_proj/T_b, α = `ALFA_PROJECTION`.
- **At signal duration**: M_j = number of measured blocks of the class, and the target level is that of the single-class quantile applied to all N blocks, Π F_j^{N_j} = p^N — which gives back exactly F(z) = p when there is a single class.

The `SYNTHESE_CLASSES = 'max'` mode (maximum of the per-class quantiles) remains available; it is always lower than or equal to the product, hence non-conservative.

**Excluded classes.** A class that holds blocks but has no usable law (fewer than `MIN_POINTS_KAPPA4` blocks, failed fit) cannot enter the synthesis. The spectrum is then **underestimated**. The case is reported in the log, in the HTML report parameter box ("⚠ Classes exclues" line) and in the `Classes_exclues` / `Occurrence_exclue` CSV columns.

For damage, the synthesis is the **sum** of the class contributions (convolution of densities, [2] eq. 38): means and variances add up.

### 4.14 IID contract and its commentary

The method assumes that the per-block maxima (and per-block damages) are **independent and identically distributed**: this is what allows the L-moment fit and the F^M projection ([2] eq. 35-36). Two tests per frequency, on the series of blocks in time order (`quality_gate_iid`):

1. **Spearman** correlation between a block and the next, computed on the ranks — fails if |ρ| > `IID_RHO_MAX`;
2. Wald-Wolfowitz **runs test** against the median — fails if p-value < `IID_PVALUE_MIN`.

Global verdict (`agreger_quality_gate`) from the fraction of failing frequencies: GO, WARNING, NO-GO. The computation is never interrupted.

The verdict alone says neither which test fails, nor where, nor why. Both HTML reports therefore contain a **"Contrat IID — explication des résultats"** block (`commenter_quality_gate`):

| Heading | Content |
|---|---|
| Verdict | status, number of failing frequencies, decision rule |
| Data tested | number of blocks, effective block duration |
| Result test by test | share of the correlation and of the runs test, extreme values, sign of ρ |
| Location of failures | contiguous frequency bands |
| Share attributable to chance | fraction of failing frequencies produced by **perfectly independent** blocks (both tests have a false-alarm rate), compared with the observed fraction |
| Probable cause | oscillator memory (τ = Q/(π·f₀) > T_b/3), slow envelope of the signal, or statistical fluctuation |
| Consequence, action | what it changes for the SRE; lengthen T_b, classify, or do nothing |
| Per class | if K > 1: independence within each class, the one that conditions the fit |
| Stationarity probes | outcome of the probes of §4.12 |

The block is written in French, like the rest of the reports.

---

## 5. User guide

### 5.1 Installation

```bash
# Python 3.10+
pip install numpy scipy scikit-learn pandas tqdm plotly numba psutil
pip install matplotlib        # unit tests only
```

### 5.2 Input CSV file format

Two columns: **time (s)** and **acceleration (m/s²)**.

```
... header lines (count = CSV_SKIP_ROWS) ...
0.000000000;0.0345
0.000078125;0.0382
```

- Delimiter: `CSV_DELIMITER` parameter (`;` or `,`).
- Decimal separator (`,` or `.`) and encoding: detected automatically.
- Unreadable lines are dropped.
- The sampling frequency is computed as `1/mean(diff(t))`.

### 5.3 Running a computation

Edit the constants at the top of `mbd_simple-multi-process_v3_6.py` (SECTION 1), then:

```bash
python mbd_simple-multi-process_v3_6.py
```

Each run creates its own **subfolder** inside `OUTPUT_FOLDER`:

```
<YYYYMMDD_HHMMSS>_<file name ≤35 chars>_f<min>-<max>_Q<Q>_Tb<Tb>_b<b>_T<Tproj>_a<alfa>
e.g.: 20261002_224557_signal_asymlaplace_f5-400_Q10_Tb1p28_b8_T36Ms_a0p9/
```

### 5.4 Reading the results: where to start

1. Open `Rapport_*.html`. The **"Paramètres du calcul"** box recalls the settings, the block splitting obtained, the stationarity verdict and its recommendation.
2. Read the **"Contrat IID"** block right below: it says whether the results call for a reservation, and which one.
3. Chart 2: compare the MBD SRE (solid line = signal duration, dashes = projected) with the analytical SRE and the SRX.
4. If in doubt about a frequency, open `Rapport_Details_*.html`: L-moment diagram, empirical and fitted distributions per class (frequency selector), MSDI indicator.

### 5.5 Which configuration for which signal?

| Signal | Recommended setting |
|---|---|
| Stationary | `N_CLUSTERS = 1`, or `AUTO_SELECT_K = True` (K = 1 will be retained) |
| Non-stationary (regimes, drift) | `AUTO_SELECT_K = True`; with no feature enabled, rms + kurtosis + crest_factor are used by default; `SYNTHESE_CLASSES = 'produit'` |
| DOUTEUX verdict | run both routes and compare the projected SREs |
| Low frequencies failing the IID contract | lengthen T_b: aim for T_b ≥ 3·Q/(π·F0_MIN) |
| Synthetic signal with a start-up transient | `TRIM_DEBUT_S` > 0 |

### 5.6 Execution modes

- **Multiprocess** (default): `USE_MULTIPROCESS = True`, `N_WORKERS` at most equal to the number of physical cores.
- **Sequential** (debugging): `USE_MULTIPROCESS = False`.

### 5.7 Fitting law — `LOI_AJUSTEMENT`

- `'kappa4'` (default) — Hosking's 4-parameter law, §4.2.
- `'rayleigh_gen'` — **generalized Rayleigh** F(x; α, λ) = (1 − e^(−(λx)²))^α. Fitted by modified L-moments ([6] eq. 18-20): Y = X² follows a generalized exponential whose L-moments are expressed with the digamma function; the shape α is the root of `[ψ(2α+1) − ψ(α+1)] / [ψ(α+1) − ψ(1)] = l₂/l₁`. Exact analytical quantile.

### 5.8 Demo mode — `mbd_demo_v1.py`

A short program that generates a signal of **known** nature, then runs the full computation of the main module: same reports, same CSVs. `python mbd_demo_v1.py`, configuration at the top of the file.

| Section | Content |
|---|---|
| A — signal | `MODE_SIGNAL`: `'stationnaire'` (flat PSD + Hermite transformation, driven by kurtosis and skewness), `'phases'` (stationary segments placed end to end), `'enveloppe'` (signal modulated by a slow RMS envelope) |
| B — classification | `N_CLUSTERS`, `AUTO_SELECT_K`, `AUTO_K_SELON_STATIONNARITE`, `K_RANGE`, `MIN_SAMPLES_PER_CLUSTER`, `SYNTHESE_CLASSES`, `FEATURE_FLAGS` |
| B bis — stationarity | `STATIONNARITE_*` |
| C — computation | law, Q, T_b, f₀ spectrum, Cunnane, SRX, SDF, projection, IID contract, priming, `KAPPA4_H_MIN` |
| D — execution | `MODULE_PATH`, output folder, multiprocess |

The demo is a way to watch the checks react: a `'stationnaire'` signal must yield the STATIONNAIRE verdict and K = 1; `'phases'` and `'enveloppe'` must yield NON_STATIONNAIRE. It stores in the run folder the generated signal and a `RESUME_demo.txt` (configuration, achieved statistics, checks observed), and prints a console summary.

The demo applies its settings at module level and registers the main module in `sys.modules`: `USE_MULTIPROCESS = True` also works on Windows.

### 5.9 Unit tests — `tests_unitaires/`

```bash
python tests_unitaires/run_all.py
```

Each test builds data whose **correct answer is known in advance** (exact theory, controlled construction, or computation by independent code), calls the function and compares. It produces a figure: plain-language explanation on the left, test data in the middle, verification on the right, PASS/FAIL banner. `run_all.py` assembles `tests_unitaires/_resultats/index.html` (summary table then one card per test, in computation order) and returns exit code 0 if everything passes, 1 otherwise. The explanations are written in French.

| Step | Test | Function(s) verified | Reference |
|---|---|---|---|
| Import, splitting | `test_importer_signal_csv` | `importer_signal_csv` | signal written under 3 file conventions |
| | `test_extraire_caracteristiques` | `extraire_caracteristiques`, `_taille_bloc` | imposed means and maxima; block duration for 6 signal lengths |
| | `test_features_blocs` | the 12 features | exact expressions for a sine |
| SDOF response | `test_reponse_sdof` | `reponse_sdof` | amplification Q, transfer function, `scipy.signal.lsim` solver |
| | `test_reponse_sdof_amorcage` | `SDOF_AMORCAGE` option | exact steady state |
| Laws | `test_lmoments` | `calculer_lmoments` | exact L-moments of 3 laws |
| | `test_kappa4_lmoments` | `kappa4_from_lmoments`, `_tau3_tau4_from_kh_analytic`, `_fit_loc_scale` | numerical integration of the quantile, 42 laws |
| | `test_kappa4` | `ajuster_kappa4`, `kappa4_ppf` | samples of known laws, k = 0 cases |
| | `test_kappa4_bord_domaine` | h domain, `_kappa4_sur_borne_h` | nearest point by brute force |
| | `test_kappa4_cdf_ppf` | `_kappa4_cdf_exact`, `_kappa4_ppf_exact`, `_loi_logcdf` | formulas of [2] eq. 30-31 |
| | `test_msdi` | `_msdi_queue_droite`, `kappa4_rmse_msdi` | MSDI = 100·ε² |
| | `test_rayleigh_gen` | `ajuster_rayleigh_gen`, `rayleigh_gen_ppf` | exact inversion of the law |
| Analytical, damage | `test_sre_analytique` | `calculer_sre_analytique` | linearity; Miles' formula |
| | `test_rainflow` | `calculer_sdf_rainflow` | sine counted by hand |
| | `test_sdf_par_bloc` | `calculer_sdf_per_bloc` | exact damage of each block |
| Projection, synthesis | `test_projection` | `calculer_projection_lmoments` | α^(1/M) identity |
| | `test_projection_gev` | `ajuster_gev_lmoments`, `calculer_projection_gev_domaines` | `scipy.stats.genextreme` |
| | `test_projection_dommage` | `calculer_projection_sdf_tcl`, `calculer_projection_dmg_kappa4` | exact sum of Gamma laws |
| | `test_synthese_classes` | `quantile_produit_classes`, `_quantile_produit` | simulation of 20,000 service lives |
| Checks | `test_quality_gate_iid` | `quality_gate_iid`, `_iid_verdict` | independent and correlated series |
| | `test_contrat_iid` | `commenter_quality_gate`, `agreger_quality_gate`, `_iid_taux_hasard`, `_iid_bandes` | 4 constructed situations |
| | `test_stationnarite` | `detecter_non_stationnarite` | 4 signals of known nature |
| Full chain | `test_traiter_f0` | `traiter_f0` | imposed classes, product equation |
| | `test_bout_en_bout` | `main` | full run on a temporary CSV, 9 checks |

The `_th.py` module holds the shared tooling (program loading, page layout, independent Kappa4 references).

### 5.10 Fit diagnostics

`KAPPA4_DEBUG_ANALYTIC = True` exports `Kappa4_Debug_*.csv`: for each (f₀, class), L-moments, starting point, solver residuals, g₁ and g₂, exit reason.

### 5.11 Computation time

Measured on a 600 s signal at 12.8 kHz (7.68 million points), 396 frequencies, one class, SDF disabled, 10-physical-core machine: **28 s in multiprocess (10 processes)**, 56 s sequentially, CSV reading included. Both modes give strictly identical results. The SDF (rainflow) lengthens the computation.

---

## 6. User parameters — full description

The "default" values are those of the delivered file.

### 6.1 Input file

| Parameter | Default | Notes |
|-----------|---------|-------|
| `CSV_FILEPATH` | (local path) | Path of the input CSV |
| `CSV_SKIP_ROWS` | 10 | Header lines to skip |
| `CSV_DELIMITER` | `";"` | Column delimiter |
| `TRIM_DEBUT_S` | 0.0 | Duration (s) removed from the start of the signal before any computation |

### 6.2 Reference oscillator and block

| Parameter | Default | Notes |
|-----------|---------|-------|
| `Q` | 10 | Quality factor Q = 1/(2ξ). Usual range 5–50 |
| `TB` | 1.28 | Block duration T_b (s). Usual range 0.05–10 s. Aim for T_b ≥ 3·Q/(π·F0_MIN) and enough blocks (≥ 40 per class) |

### 6.3 Frequency spectrum

| Parameter | Default | Notes |
|-----------|---------|-------|
| `F0_MIN` | 5 | Minimum natural frequency (Hz) |
| `F0_MAX` | 400 | Maximum natural frequency (Hz). Stay below fs/18 for an amplitude error < 1% (§9) |
| `DELTA_F0` | 1 | Frequency step (Hz) |

### 6.4 Classification and class synthesis

| Parameter | Default | Notes |
|-----------|---------|-------|
| `N_CLUSTERS` | 1 | Number of classes imposed when `AUTO_SELECT_K = False`. Requires at least one enabled feature |
| `AUTO_SELECT_K` | False | True: K = 1 if the signal is judged stationary, otherwise K maximizing the Silhouette score |
| `AUTO_K_SELON_STATIONNARITE` | True | Ties `AUTO_SELECT_K` to the stationarity verdict |
| `K_RANGE` | `range(2, 7)` | Range tested for K |
| `MIN_SAMPLES_PER_CLUSTER` | 40 | If a class has fewer blocks, K is decremented. Keep ≥ `MIN_POINTS_KAPPA4` |
| `SYNTHESE_CLASSES` | `'produit'` | `'produit'` or `'max'` — §4.13 |
| `FEATURE_FLAGS` | all False | `mean`, `variance`, `skewness`, `kurtosis`, `rms`, `mav`, `crest_factor`, `autocorr_lag1`, `zcr`, `dominant_freq`, `spectral_centroid`, `spectral_spread` |

### 6.5 Stationarity check

| Parameter | Default | Notes |
|-----------|---------|-------|
| `STATIONNARITE_ENABLED` | True | False: no diagnostic |
| `STATIONNARITE_N_SONDES` | 8 | Number of probe frequencies. Range 4–20 |
| `STATIONNARITE_FRAC_ECHEC` | 0.3 | Threshold of indicator (a). Range 0.1–0.5 |
| `STATIONNARITE_CV_RMS_MAX` | 0.15 | Threshold of indicator (b). Range 0.05–0.40; raise towards 0.20–0.25 if T_b < 0.2 s |
| `STATIONNARITE_KURT_TOL` | 0.5 | Tolerance on \|kurtosis − 3\| for "Gaussian" |
| `STATIONNARITE_SKEW_TOL` | 0.3 | Tolerance on \|skewness\| for "Gaussian" |

### 6.6 Quantile at signal duration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `PROBABILITE_CIBLE` | 0.9 | Non-exceedance probability of the maximum of **one block**, used if `OPTION_CUNNANE = False`. Risk α of [2] Table 1: 0.9 / 0.99 / 0.999 |
| `OPTION_CUNNANE` | True | p = (N − a)/(N + 1 − 2a): highest quantile reachable with N measured blocks |
| `CUNNANE_A` | 0.4 | Cunnane constant (ν in the standard) |

### 6.7 SDF

| Parameter | Default | Notes |
|-----------|---------|-------|
| `SDF_ENABLED` | False | Enables the damage computation (rainflow and Bendat) |
| `SDF_B` | 8.0 | Basquin slope b. [2]: 4, 5 or 8 for electronic, optronic, mechanical equipment |
| `SDF_C` | 1.0 | Basquin constant (relative analysis) |

### 6.8 Long-duration projection

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ENABLE_PROJECTION` | True | Enables the projection |
| `DUREE_PROJECTION` | 36,000,000 | T_proj (s) = 10,000 h |
| `ALFA_PROJECTION` | 0.90 | Projected **non-exceedance** probability |
| `METHODE_PROJECTION` | `'puissance'` | `'puissance'` or `'gev_domaines'` — §4.10 |
| `ALPHA_SRX_HIGH` | 0.99 | High SRX risk (comparison with a shock) |
| `ALPHA_SRX_LOW` | derived | = 1 − `ALFA_PROJECTION` |
| `LOI_AJUSTEMENT` | `'kappa4'` | `'kappa4'` or `'rayleigh_gen'` |

### 6.9 IID contract

| Parameter | Default | Notes |
|-----------|---------|-------|
| `IID_GATE_ENABLED` | True | Enables the check and its commentary |
| `IID_RHO_MAX` | 0.2 | Threshold on Spearman's \|ρ\| |
| `IID_PVALUE_MIN` | 0.05 | P-value threshold of the runs test |
| `IID_FAIL_FRAC_MAX` | 0.05 | Fraction of failing frequencies up to which the verdict stays GO |
| `IID_NOGO_FRAC` | 0.30 | Fraction from which the verdict is NO-GO |
| `IID_MIN_N` | 20 | Minimum number of blocks to run the tests |

### 6.10 Miscellaneous and execution

| Parameter | Default | Notes |
|-----------|---------|-------|
| `RANDOM_SEED` | 53 | Seed (K-Means, chance rate of the IID contract) |
| `MIN_POINTS_KAPPA4` | 40 | Minimum number of blocks to fit a law |
| `OUTPUT_FOLDER` | `"mbd_simple_output"` | Root output folder |
| `USE_MULTIPROCESS` | True | Frequency loop in parallel |
| `N_WORKERS` | 10 | Number of processes; `None` = physical cores |

### 6.11 Expert parameters

Gathered in the "PARAMÈTRES EXPERT" map that follows SECTION 1.

| Parameter | Default | Role |
|-----------|---------|------|
| `KAPPA4_ANALYTIC_XTOL` | 1.49e-8 | `fsolve` tolerance |
| `KAPPA4_L2_MIN_FOR_FIT` | 1e-10 | Below this L2, data deemed constant: fit refused |
| `KAPPA4_RESIDUAL_TOL` | 1e-6 | Maximum accepted squared residual on (τ3, τ4) |
| `KAPPA4_WARM_START_INITIAL` | (0.1, 0.1) | Starting point (k, h) of the first attempt |
| `KAPPA4_RETRY_ENABLED`, `KAPPA4_RETRY_WARM_STARTS`, `KAPPA4_RETRY_MAXFEV` | True, 4 starts, 2000 | Retries when the solver does not converge |
| `KAPPA4_H_MIN` | −1.0 | Lower bound of h; `None` = no check |
| `KAPPA4_DOMAIN_RETRY_STARTS` | 4 starts | Starting points within h ∈ [−1, 0) after an out-of-domain root is rejected |
| `KAPPA4_DEBUG_ANALYTIC` | False | Exports `Kappa4_Debug_*.csv` |
| `SDOF_AMORCAGE`, `SDOF_AMORCAGE_N_TAU` | True, 8.0 | Oscillator priming — §4.3 |
| `GEV_GUMBEL_K_TOL` | 0.01 | \|k\| threshold of the "gumbel" label (diagnostic) |
| `KAPPA4_MEAN_VAR_GRID` | 4096 | Integration grid for the moments of the law (damage) |
| `PROBA_CLIP_EPS` | 1e-7 | Bounds of the Cunnane probability |
| `MIN_ECH_PAR_BLOC` | 10 | Minimum number of samples per block |

---

## 7. Output files

All in the run subfolder, suffixed with a tag (`v3p5pub_f5-400_Q10_Tb1p28_b8_T36Ms_a0p9_<date>`).

| File | Content |
|------|---------|
| `params_*.json` | All run parameters, `Tb_effectif_s`, `n_blocs`, stationarity diagnostic |
| `Stationnarite_*.csv` | One summary line (verdict, indicators), then one line per probe |
| `SRE_*.csv` | SRC, MBD SRE, analytical SRE, low and high α SRX, IID contract columns. If K > 1: `Synthese_classes`, `SRE_Kappa4_max_classe_*`, `Classes_exclues`, `Occurrence_exclue` |
| `SRE_Projection_*.csv` | Projected SRE, projected analytical SRE and SRX, M, projected SDFs (if SDF enabled). `GEV_domaine` with the GEV method. If K > 1: same synthesis columns |
| `IID_QualityGate_*.csv` | Per (f₀, class): ρ, p-value, number of blocks, status. `Classe = −1` is the series of all blocks |
| `SDF_*.csv` | (if SDF enabled) rainflow over the whole signal, sum of per-block damages, Bendat, fit quality on D_p |
| `SDF_Kappa4_Fit_*.csv` | (if SDF enabled) parameters of the law fitted on the D_p, per (f₀, class) |
| `Kappa4_Debug_*.csv` | (if `KAPPA4_DEBUG_ANALYTIC`) solver trace |
| `Rapport_*.html` | Spectral summary: (1) SRC and SRE; (2) SRE / SRX comparison, measurement and projection; (3) SDF; (4) measured and projected SRE |
| `Rapport_Details_*.html` | Diagnostics: (1) (τ3, τ4) diagram; (2) per-class distributions, frequency selector; (3) SRE / SRX comparison and MSDI; (4) projected SDF; (5) IID contract vs f₀ |

At the top of both reports, three collapsible blocks: **run parameters**, **IID contract**, **curve reading guide**.

**CSV column suffixes**: each quantity carries the parameters it depends on, which allows stacking CSVs from different runs.

- `SRC_Q10_Tb1p28s`
- `SRE_Kappa4_Q10_Tb1p28s_P0p9_Tmes599p999s`
- `SRX_alpha_low_Q10_Tb1p28s_aL0p1_Tmes599p999s`
- `SRE_Projection_Q10_Tb1p28s_P0p9_T36Ms_a0p9`
- `IID_rho_lag1_Q10_Tb1p28s`

**Fit statuses** (SRE curve hover text, debug CSV): `ok`, `ok_retry` (obtained from a fallback starting point), `ok_h_borne` (boundary law of the domain), `n_lt_min` (too few blocks), `fit:<reason>` (failure).

---

## 8. Code architecture (internal sections)

`mbd_simple-multi-process_v3_6.py` is a single self-contained file.

| Section | Role | Main functions |
|---------|------|----------------|
| 1 | Configuration, then expert parameter map | — |
| 2 | Logging | `logging.basicConfig` |
| 3 | CSV import | `importer_signal_csv` |
| 4 | Splitting and features | `_taille_bloc`, `extraire_caracteristiques` |
| 5 | SDOF response | `reponse_sdof` |
| 6 | Kappa4 | `_calculer_pwm`, `calculer_lmoments`, `_g_functions`, `_g_deriv_k0`, `_fit_loc_scale`, `_tau3_tau4_from_kh_analytic`, `_kappa4_h_hors_domaine`, `_kappa4_au_dessus_glo`, `_kappa4_sur_borne_h`, `kappa4_from_lmoments`, `ajuster_kappa4`, `_msdi_queue_droite`, `kappa4_rmse_msdi`, `_kappa4_ppf_exact`, `_kappa4_cdf_exact`, `kappa4_ppf` |
| 6bis | Generalized Rayleigh and dispatch | `ajuster_rayleigh_gen`, `rayleigh_gen_ppf`, `rayleigh_gen_rmse_msdi`; `ajuster_loi`, `loi_ppf`, `loi_rmse_msdi` |
| 6ter | GEV projection | `_gev_domaine`, `ajuster_gev_lmoments`, `calculer_projection_gev_domaines` |
| 7 | Analytical branch | `calculer_sre_analytique` |
| 8 | Rainflow | `_rainflow_damage`, `calculer_sdf_rainflow`, `calculer_sdf_per_bloc` |
| 9 | Projection | `calculer_projection_lmoments`, `calculer_projection_sdf_tcl`, `calculer_projection_dmg_kappa4`, `_kappa4_mean_var` |
| 9bis | IID contract | `quality_gate_iid`, `_iid_verdict`, `agreger_quality_gate`, `_iid_taux_hasard`, `_iid_bandes`, `commenter_quality_gate` |
| 9ter | Stationarity | `detecter_non_stationnarite`, `exporter_csv_stationnarite` |
| 9quater | Class synthesis | `_loi_logcdf`, `quantile_produit_classes`, `_quantile_produit` |
| 10 | Processing of one frequency | `traiter_f0`, worker processes |
| 11 | CSV exports | `build_run_meta`, `exporter_csv_*`, `_cols_synthese_classes`, `_bilan_classes_exclues` |
| 12 | HTML reports | `generer_html`, `generer_html_details`, `_ecrire_html_avec_cadre` |
| 13 | Main program | `main`, `main_mp` |

---

## 9. Known pitfalls and precautions

**Sensitivity of the projection.** The SRE projected to 10,000 h depends strongly on the shape parameter k when it is close to zero: its sign separates a bounded tail from a heavy one. On a 600 s signal, changing only the block splitting (468 blocks instead of 363) moved the projected SRE by more than 20% at 7% of the frequencies, while the SRE at signal duration moved by less than 4%. A saw-toothed projected spectrum is the mark of this sensitivity, not a property of the signal. Cross-check with the generalized Rayleigh law and with the analytical SRX.

**IID contract — read the commentary, not just the verdict.**
- The GO threshold (5% of failing frequencies) is of the same order as the false-alarm rate of the two tests: 4 to 6% beyond 300 blocks, about 9% for 90 blocks, 22% for 40 blocks. A perfectly independent signal may come out as WARNING. The "share attributable to chance" heading sorts this out.
- When K > 1, the verdict bears on the series of all blocks, classes pooled: it is expected to fail on a signal with regimes. The "per class" heading is the one that tells whether the fit is valid.

**T_b and low frequencies.** Below f₀ = 3·Q/(π·T_b), the response of a block spills over into the next. With Q = 10 and T_b = 1.28 s, this concerns f₀ < 7.5 Hz. Remedy: T_b ≥ 3·Q/(π·F0_MIN), at the cost of fewer blocks.

**Excluded classes.** The "⚠ Classes exclues" line flags an underestimated spectrum; the discarded class may be the most severe regime. The remedy lies with the data: a longer signal or fewer classes. `MIN_SAMPLES_PER_CLUSTER` ≥ `MIN_POINTS_KAPPA4` avoids the case for classes that are too small.

**`N_CLUSTERS` > 1 with no feature.** With `AUTO_SELECT_K = False` and every `FEATURE_FLAGS` entry at False, the run falls back to a single class with no specific warning. Enable at least `rms`.

**`gev_domaines` option.** It re-fits a GEV (h = 0) on the maxima, whereas [2] designates the domain of attraction by the k* of the fitted Kappa4. The two k values may differ, up to a change of sign; on trial laws, the gap on the projected SRE ranged from −55% to several hundred %. Use it as a cross-check, not as a reference.

**Uniqueness of the Kappa4.** The condition h ≥ −1 is not sufficient everywhere: for τ3 above about 0.28, near the h = −1 curve, two (k, h) pairs reproduce exactly the same (τ3, τ4). The solver returned the root with the largest h in every trial, but nothing guarantees it. Only concerns strongly skewed maxima.

**Projected SDF "law on D_bloc".** The moments of the fitted law are integrated on a truncated grid: for a heavy tail (k < 0), mean and variance are underestimated. The projection from empirical moments (`SDF_Proj_TCL_empirique`) does not have this bias.

**F0_MAX and sampling frequency.** Smallwood's filter assumes the excitation is linear between two samples; the response amplitude is reduced by about (π·f/fs)²/3, i.e. 3% at fs/10 and 1% at fs/18.

**End of the signal.** The last samples that do not fill a block are ignored in the MBD branch (the SRC and the analytical branch cover the whole signal).

**Column name.** The `SRE_Kappa4_*` column keeps that name when the chosen law is the generalized Rayleigh.

**CSV.** A `CSV_SKIP_ROWS` that is too small does not raise an error: the unreadable header lines are simply dropped. Check in the log the number of points and the duration read.

---

## 10. Improvement suggestions

1. **Uniqueness of the Kappa4**: add Hosking's (1994) constraint on the (k, h) pair to the domain check, and test explicitly k > −1 and h·k > −1 ([2] Figure 11).
2. **GEV projection**: project with the k* of the fitted Kappa4, without re-fitting at h = 0, in line with [2] §4.1.
3. **IID verdict**: evaluate it class by class when K > 1, and tie the GO threshold to the chance rate computed for the number of blocks.
4. **Moments of the law for damage**: mean = L1 by construction; variance in closed form through g₁(2k, h), refused when it does not exist (k ≤ −0.5).
5. **Uncertainty of the projection**: confidence interval by resampling the blocks, to display the scatter of the projected SRE.
6. **Classification**: principal component analysis of the features before K-Means ([2] §4.2).
7. **Criteria of the standard**: warn when M < 100 (extreme values) or M < 50 (central limit theorem).
8. **Excluded classes**: bound the SRE at signal duration from below by the largest measured maximum of the discarded class.
