# MyMBD — Outil d'analyse vibratoire Méthode des Blocs Disjoints "MBD" / Kappa4 & Rayleigh "généralisée" / SRE-SRX / SDF

> **Auteur** : Guillaume LE ROUSSEAU
> **Programme principal documenté** : `mbd_simple-multi-process_v3_4.py`
> **Version courante** : V3.4
> **Langage** : Python 3.10+
> **Domaine** : essais vibratoires — personnalisation d'environnement mécanique

---

## Sommaire

1. [Présentation et finalité](#1-présentation-et-finalité)
2. [Cadre normatif et références documentaires](#2-cadre-normatif-et-références-documentaires)
3. [Workflow détaillé du programme](#3-workflow-détaillé-du-programme)
4. [Choix techniques et justifications](#4-choix-techniques-et-justifications)
5. [Guide d'utilisation](#5-guide-dutilisation)
6. [Paramètres utilisateur — description complète](#6-paramètres-utilisateur--description-complète)
7. [Fichiers de sortie](#7-fichiers-de-sortie)
8. [Architecture du code (sections internes)](#8-architecture-du-code-sections-internes)
9. [Pièges connus et précautions](#9-pièges-connus-et-précautions)
10. [Suggestions d'amélioration](#10-suggestions-damélioration)

---

## 1. Présentation et finalité

Le programme calcule, à partir d'un signal d'accélération mesuré (typiquement issu d'un essai sur véhicule, banc, ou enregistrement de roulage), **trois grandeurs spectrales** dépendant d'une fréquence propre f₀ :

- **SRC(f₀)** — Spectre de Réponse au Choc (max de la réponse SDOF)
- **SRE(f₀)** — Spectre de Réponse Extrême (quantile haut, méthode MBD Kappa4)
- **SDF(f₀)** — Spectre de Dommage par Fatigue (loi de Basquin + Miner, comptage rainflow)

Et leur **projection à la durée de vie cible** (T_proj typiquement 36×10⁶ s ≈ 1 an).

Le pipeline implémente la méthode **MBD non corrélée** (Méthode du Bloc de Durée) de la norme **NF X50-144-3 (édition 2021), Annexe C**, avec l'amélioration proposée par **B. Colin (COFREND 2023)** consistant à utiliser une **loi Kappa4 (Hosking, 4 paramètres)** comme loi paramétrique unique — Kappa4 dégénère analytiquement en GUM/FRE/WBN/EV1/EV2/EV3 selon les valeurs de (k, h), ce qui évite l'arbre de décision LAP de la norme (Figure C.5). Un autre mode utilise une loi paramétrique "Rayleigh généralisée" couvrant un domaine restreint (A. Clou & P. Lelan , DGA TT / CFM 2025).

L'approche est complétée par :
- une **projection longue durée** par TVE (théorie des valeurs extrêmes) pour le SRE et par TCL (théorème central limite, loi log-normale) pour le SDF ;
- une **classification K-Means** des blocs d'excitation basée sur de multiples métriques classiques dans le but de traiter les signaux non stationnaires en classes localement stationnaires ;
- un **comptage rainflow ASTM E1049-85 strict** (algorithme par pile de Downing-Socie 4-points), accéléré Numba ;
- un **calcul SRE/SRX/SDF analytique depuis la DSP** (Welch + intégration analytique `∫ Pxx·|Fd|² df` → **SRE narrow-band gaussien** suivant **PR NORMDEF 0101 §5.4.2** et **SRX à risque de dépassement α** suivant **§5.4.3 équation [5.2]** ; Bendat-Lalanne narrow-band pour le SDF) à des fins de comparaison.

---

## 2. Cadre normatif et références documentaires

### Références principales (PDF locaux)

| # | Document | Fichier local | Rôle |
|---|----------|---------------|------|
| **[1]** | **NF X50-144-3 (2021)** — *Démonstration de la tenue aux environnements mécaniques, Partie 3 : Personnalisation* | `[NF X50 144-3] 2022.pdf` | Cadre normatif Annexe C (méthode MBD) |
| **[2]** | **B. Colin (KNDS / COFREND 2023)** — *Maintenance prévisionnelle des équipements critiques embarqués sur systèmes d'armes terrestres* | `MBD&KAPPA4_ME3E2_B_Colin.pdf` | Formules Kappa4 par L-moments analytiques |
| **[3]** | **Clou & Lelan (DGA TT / CFM 2025)** — *Development of statistical methods for vibration analysis — Application to land vehicles* | `2025-12-08_Article_CFM_2025_CLOU_LELAN.pdf` | Critères SSI, limites Kappa4, Rayleigh généralisée |
| **[4]** | **PR NORMDEF 0101 (DGA 2009)** — *Norme Défense — Personnalisation des essais en environnement mécanique* | `_docs-SRX_extract_31-39_prnormdef0101pcemv12versionaste.pdf` | §5.4.2 (SRE), §5.4.3 (SRX) eq. [5.2]/[5.3], fig. 5.3 |
| **[5]** | **B. Colin (DGA, MI0460 / 2008)** — *Définition d'un Spectre de Réponse à risque de dépassement (SRX) — Modèle non asymptotique* | `_docs-SRX_Colin_mi0460-2008.pdf` | Origine du modèle SRX non-asymptotique vs Gumbel/Poisson |

### Pages clés à consulter

**[1] NF X50-144-3 — Annexe C (pages 70–89 du PDF) :**
- p. 71 (Figure C.1) : **σ(t) = K · z(t)** — relation contrainte/déplacement, K=1 forfaitaire
- p. 74 (Figure C.3) : synoptique complet MBD (8 étapes)
- p. 76 (§C.3) : transformée en z FOH — formules récursives Smallwood
- p. 78–80 (§C.5–C.7) : choix LAP, MMP, KS-M / MSDI / Cunnane (ν=0.4)
- p. 84 (§C.9) : critère M > 100 (TVE) / M > 50 (TCL)
- p. 88 (§C.11) : **SRE_α(f₀) = 4π²·f₀²·Z_sup,α(f₀)** ; **SDF_α(f₀) = D_c,α**
- p. 89 (Tableau C.3) : expressions analytiques L-moments des LAP

**[2] Colin 2023 :**
- p. 3–4 : Basquin (eq 1) et critère SRE_PGQE^α < SRE_PE^α (eq 3)
- p. 4 (Tableau 1) : risque α selon criticité (10% / 1% / 0.1%)
- p. 10–11 : **équations 16–19 et 20.1–20.3** = formules Kappa4 (g_r, λ₁, λ₂, τ₃, τ₄)
- p. 12–13 : **équations 28–34** = procédure d'inférence (k*, h*, α*, ξ*)

**[3] Clou & Lelan 2025 :**
- p. 2 : définitions MRS et FDS
- p. 7 (§4.1) : critère SSI (Spectral Similarity Index), seuil recommandé 85%
- p. 11 (Figure 10) : fluctuations Kappa4 ±135% sur signal gaussien (limite identifiée)

**[4] PR NORMDEF 0101 (DGA 2009) :**
- p. 33 (§5.4.2) : définition SRE — pic moyen sur T de la réponse SDOF en pseudo-accélération `(2π·f₀)²·z_sup`
- p. 34–35 (§5.4.3) : **SRX**, équations [5.2] (non-asymptotique) et [5.3] (rapport SRX/SRE) ; figure 5.2 (rapport SRX/SRE vs n₀⁺·T)
- p. 36 (fig. 5.3) : illustration comparée SRC choc / SRE / SRX(1%) / SRX(99%) d'une vibration aléatoire

**[5] Colin (MI0460, 2008) :**
- discussion du modèle SRX non-asymptotique vs modèles asymptotiques Gumbel ([5.6]) et Poisson ([5.7]) — équivalence pour n₀⁺·T > 1000

### Références secondaires (algorithmes)

- **ASTM E1049-85 (2017)** — *Standard Practices for Cycle Counting in Fatigue Analysis* (algorithme rainflow strict, 4-point Downing-Socie). Implémenté ici en Numba.
- **AFNOR A03-406** — *Méthodes de comptage des cycles pour l'analyse en fatigue* (équivalent rainflow français).
- **Hosking, J.R.M. (1994)** — *The four-parameter kappa distribution*, IBM Journal of Research and Development, 38(3):251–258. Définition originale de la Kappa4.
- **Bendat, J.S. (1964)** ; **Lalanne, C. (2009)** — *Mechanical Vibration and Shock Analysis*, Vol. 4 *Fatigue Damage*, Wiley/ISTE. Approximation narrow-band gaussienne du SDF spectral.
- **Smallwood, D.O. (1981)** — *An improved recursive formula for calculating shock response spectra*, Shock & Vibration Bulletin 51. Formules FOH récursives utilisées en p. 76 de la norme [1].
- **Cunnane, C. (1978)** — *Unbiased plotting positions — A review*, Journal of Hydrology 37, 205–222. Plotting positions p_i:n = (i−a)/(n+1−2a), a=0.4.
- **Mielke, P.W. (1973)** — Polynômes d'ordre 6 pour τ₄(τ₃) à h fixé (référencés dans [2], Tableaux 2.1–2.2 ; remplacés ici par calcul analytique).
- **Lalanne, C. (2002)** — Mechanical Vibration and Shock, Volume 3 (formule SRX Rayleigh)
LALANNE C. (2002) — Mechanical Vibration and Shock, Volume 3: Random Vibration, Hermes Penton, 2002

### Bibliothèques scientifiques utilisées

- **NumPy / SciPy** — calcul numérique (`scipy.signal.lfilter`, `scipy.special.gamma/beta`, `scipy.optimize.fsolve`, `scipy.stats.kappa4`, `scipy.stats.lognorm`)
- **scikit-learn** — classification non-supervisée (`KMeans`, `StandardScaler`, `silhouette_score`)
- **Numba** — accélération JIT du rainflow (cache disque partagé entre workers)
- **pandas** — I/O CSV et exports
- **plotly** — rapports HTML interactifs
- **tqdm** — barres de progression

---

## 3. Workflow détaillé du programme

Le pipeline reproduit la **Figure C.3 de NF X50-144-3** ([1] p. 74) avec quelques aménagements (Kappa4 en lieu et place de l'arbre LAP, parallélisation par f₀).

```
┌───────────────────────────────────────────────────────────────────┐
│  ENTRÉE : signal CSV  (t, ẍ)  ─  fs ≈ 12,8 kHz typique            │
└───────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 1. Import CSV (auto-encodage)      │  importer_signal_csv()
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 2. Extraction features par bloc Tb │  extraire_caracteristiques()
│    (mean, var, skew, kurt, mav,    │  Découpage en n_blocs de
│     crest_factor, …)               │  taille = round(Tb·fs),
└────────────────────────────────────┘  ajusté à un diviseur de N
              │
              ▼
┌────────────────────────────────────┐
│ 3. Classification K-Means          │  Auto-K (silhouette) optionnel
│    Repli auto si MIN_SAMPLES non   │  → vecteur clusters[n_blocs]
│    respecté                        │
└────────────────────────────────────┘
              │
              ▼  pour chaque f0 ∈ [F0_MIN .. F0_MAX] (parallèle)
┌────────────────────────────────────┐
│ 4A. Réponse SDOF z(t) — FOH        │  reponse_sdof()
│     Smallwood récursif (norme §C.3)│  filtre IIR ordre 2
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4B. Pseudo-accélération            │  contrainte = (2πf₀)²·z(t)
│     pour calcul SRC / Z_ext        │  (m/s² — sert au SRE)
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4C. Maxima par bloc → {Z_ext(i)}   │  Branche SRE
│     Rainflow par bloc → {D_p(i)}   │  Branche dommage MBD
│     (sur z(t))                     │  ⚠ NF X50-144-3 §C.2
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4bis. Quality Gate IID             │  quality_gate_iid()
│   ρ Spearman lag-1 (sur rangs)     │  agreger_quality_gate()
│   + test des suites Wald-Wolfowitz │  → 🟢 GO / 🟡 WARNING /
│   par f₀ (global + par classe)     │     🔴 NO-GO (cahier §5)
└────────────────────────────────────┘
              │
              ▼  pour chaque classe i
┌────────────────────────────────────┐
│ 4D. Kappa4 sur Z_ext (SRE)         │  ajuster_kappa4()
│     Kappa4 sur D_p   (SDF)    │  → {ξ, α, k, h}
│     L-moments analytiques + fsolve │  Réf. [2] §28-34
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4E. Quantile Cunnane / cible       │  PPF Kappa4 à p_eff
│     SRE(f₀,classe) = PPF(p_eff)    │  Réf. [1] §C.7, [2] eq. 3
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4F. Synthèse classes               │  SRE(f₀) = max_i SRE(f₀,i)
│                                    │  SDF(f₀) = Σᵢ Σⱼ D_p,ij
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 5. SRE/SDF analytique depuis DSP   │  calculer_sre_analytique()
│    Welch → ∫|H(f)|²·G_xx df        │  Rayleigh + Bendat
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 6. Projection longue durée         │  M(j) = (T_v/T_b)·Occ(j)
│    SRE  : F_M(x) = F(x)^M          │  TVE  → Réf. [1] §C.9-C.10
│    SDF  : log-normale via TCL      │  TCL  → Réf. [1] Tab. C.2
│           (μ_K4·M, σ²_K4·M)        │  moments Kappa4
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 7. Exports CSV + JSON sidecar      │  Suffixes paramètres dans
│    8. Rapports HTML Plotly         │  les noms de colonnes
└────────────────────────────────────┘
```

---

## 4. Choix techniques et justifications

### 4.1 Pourquoi Kappa4 plutôt que l'arbre LAP de la norme ?

La norme [1] (Figure C.5) utilise un **arbre de décision** entre 7 lois (GUM, FRE, WBN, WB2P, WB3P, LN2P, LN3P) selon la taille d'échantillon n. Le papier [2] propose Kappa4 comme **loi unique** parce qu'elle dégénère asymptotiquement :
- k > 0 → famille EV3 / Weibull bornée
- k → 0, h = 0 → Gumbel (EV1)
- k < 0 → famille EV2 / Fréchet
- h > 0 → famille généralisée de Pareto

Cela élimine la discontinuité LAP du choix par seuil sur n et fournit une **branche unifiée** pour le SRE et pour le SDF (sur les D_p, [1] Tableau C.2).

**Limite identifiée** ([3] Figure 10) : la Kappa4 fluctue ±135% par rapport à la solution analytique sur signal gaussien synthétique. Le programme propose donc en parallèle les estimations SRX "Rayleigh" et le SDF Bendat-Lalanne pour cross-check.

### 4.2 Méthode d'identification : `direct_pwm_analytic`

Trois approches existent dans la littérature pour identifier (k*, h*, α*, ξ*) :
1. **Mielke polynomial** — table de polynômes ordre 6 pour τ₄(τ₃) à h fixé ([2] Tableaux 2.1–2.2).
2. **Maximum de vraisemblance** — coûteux et instable.
3. **L-moments analytiques + fsolve (warm-start)**. Calcule (τ₃, τ₄) théoriques par les fonctions g_r de Hosking ([2] eq. 20.1–20.3) puis résout le système 2×2 :
```
   τ3_calculé(k, h) = τ3_empirique
   τ4_calculé(k, h) = τ4_empirique
```
par `scipy.optimize.fsolve`, démarré sur (k=0.1, h=0.1).

**Avantages** :
- pas de table polynomiale à maintenir,
- xtol contrôlable (`KAPPA4_ANALYTIC_XTOL`, défaut 1.49e-8),
- résidu vérifié a posteriori (rejet si > 1e-6).

### 4.3 SDOF par FOH (First-Order Hold) + lfilter

Conformément à [1] §C.3 (p. 76), on utilise les coefficients récursifs **Smallwood 1981** pour passer du signal continu au filtre IIR ordre 2 équivalent — **exact à l'échantillonnage** sous l'hypothèse FOH (interpolation linéaire de l'excitation entre échantillons), contrairement au ZOH qui sous-estime la réponse au-dessus de fs/10.

L'utilisation de `scipy.signal.lfilter` avec conditions initiales `lfilter_zi * y[0]` évite le transitoire au démarrage.

### 4.4 SDF rainflow : ASTM E1049 strict, pas de réarrangement

L'option `rearrange=True` (réarrangement préalable depuis le pic absolu, type DSF / MIL-STD-810) ferme artificiellement le cycle majeur et **sur-estime le dommage** de quelques pourcents — d'autant plus que b est grand. **ASTM E1049 strict** (`rearrange=False`), équivalent au paquet `iamlikeme/rainflow` au bit près. Les résidus non fermés sont comptés en **demi-cycles** (n_i = 0.5).

Convention adoptée : **amplitude** (σ_a = range/2), cohérente avec Basquin sous la forme `N · σ_aᵇ = C`.

### 4.5 SDF appliqué sur z(t), pas sur la pseudo-accélération

[1] §C.2 p. 71, Figure C.1 :
- σ(t) = K · z(t), K = 1 forfaitaire
- Le rainflow appliqué sur **z(t)** (déplacement relatif, en m).

### 4.6 Granularité du dommage = bloc T_b

Somme les D_p,bloc puis ajuste Kappa4 sur la **distribution des D_p** (cf. [1] Tableau C.2 p. 85). La projection longue durée utilise μ_K4·M et σ²_K4·M (TCL → log-normale).

### 4.7 Parallélisation `multiprocessing.shared_memory`

La boucle sur f₀ est embarrassingly parallel. Le signal d'excitation (typiquement >10 M points × 8 octets = 80 Mo) est mis en `shared_memory` pour éviter de le sérialiser à chaque tâche. Le cache JIT Numba est précompilé avant le fork pour qu'il soit partagé entre workers.

### 4.8 SRE & SRX analytiques DSP — pourquoi deux niveaux α

La branche DSP (`calculer_sre_analytique`) implémente les deux estimateurs analytiques narrow-band gaussiens de [4] PR NORMDEF 0101 §5.4.2-5.4.3 :

- **SRE** (§5.4.2) — `(2π·f₀)²·z_eff·√(2·ln(n₀⁺·T))` — pic moyen attendu sur T (limite α → 1/(n₀⁺·T) de [5.2]).
- **SRX(α)** (§5.4.3 eq. [5.2]) — `(2π·f₀)²·z_eff·√(-2·ln(1-(1-α)^(1/(n₀⁺·T))))` — pic à risque de dépassement α explicite.

Deux niveaux α sont calculés en parallèle pour couvrir les deux **usages métier distincts** de [4] §5.4.3 :

1. **α faible (~1%) — Dimensionnement** : enveloppe haute. La structure doit résister à un pic que la réponse a 1% de probabilité de dépasser pendant T.
2. **α élevé (~99%) — Comparaison vs choc** : si SRX(99%) du signal aléatoire dépasse le SRC d'un choc, on démontre avec ≥99% de probabilité que la vibration est plus sévère que le choc (essai de choc évitable, cf. [4] fig. 5.3).

Le choix de la **méthode DSP plutôt que temporelle** (`z_eff = std(z(t))` sur SDOF simulé, comme dans le prototype `SRX_0.1.py`) tient au fait que cette branche est *complémentaire* à la branche MBD-Kappa4 qui est elle déjà temporelle, par bloc. L'estimateur analytique stationnaire-gaussien est cohérent avec [4] §5.4.5 (méthode analytique) et plus rapide ; il converge vers le résultat temporel à hypothèse de stationnarité respectée.

Les modèles asymptotiques de Gumbel ([4] eq. [5.6]) et Poisson ([5.7]) ont été remplacés par le modèle non-asymptotique [5.2] : selon [5] Colin (MI0460), ils ne deviennent comparables au modèle non-asymptotique que pour `n₀⁺·T > 1000` et Gumbel est par ailleurs très conservatif pour les courtes durées.

**Convention opposée et alignement automatique** : NORMDEF [4] §5.4.3 utilise α comme **probabilité de dépassement**, alors que NF X50-144-3 [1] §C.9 (et `ALFA_PROJECTION` dans ce code) utilise la **probabilité de non-dépassement**. Pour garantir un comparatif homogène entre le **SRE MBD projeté** et le **SRX α_low projeté** sur la même durée `T_proj`, le code calcule automatiquement `ALPHA_SRX_LOW = 1 - ALFA_PROJECTION` (ex. ALFA_PROJECTION=0.9 ⇒ ALPHA_SRX_LOW=0.1, même risque de 10% des deux côtés). Seul `ALPHA_SRX_HIGH` reste à régler indépendamment.

### 4.9 Limitation BLAS à 1 thread par worker

Imposée **avant import numpy** par variables d'environnement `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc. — sinon chaque worker tente d'utiliser tous les cœurs (sur-souscription, ralentissement 2-5×).

---

## 5. Guide d'utilisation

### 5.1 Installation

```bash
# Python 3.10+
pip install numpy scipy scikit-learn pandas tqdm plotly numba psutil
```

`psutil` est optionnel (utilisé pour détecter les cœurs physiques quand `N_WORKERS=None`).

### 5.2 Format du fichier CSV d'entrée

Deux colonnes : **temps (s)** et **accélération (m/s²)** :
```
... lignes d'en-tête (nombre = CSV_SKIP_ROWS) ...
0.000000000;0.0345
0.000078125;0.0382
0.000156250;0.0411
...
```
- Délimiteur : `;` ou `,` (paramètre `CSV_DELIMITER`)
- Décimal : auto-détection (`,` ou `.`)
- Encodage : auto-détection (utf-8, latin-1, windows-1252, iso-8859-1, cp1252)
- Nombre de lignes d'en-tête à sauter : `CSV_SKIP_ROWS`
- La fréquence d'échantillonnage `fs` est calculée comme `1/mean(diff(t))`.

### 5.3 Configuration

Modifier les constantes en tête du fichier `mbd_simple-multi-process_v3_4.py` (SECTION 1), puis :

```bash
python mbd_simple-multi-process_v3_4.py
```

Le programme produit un dossier de sortie (`OUTPUT_FOLDER`, par défaut `mbd_simple_output/`) contenant les CSV, le sidecar JSON des paramètres, et les rapports HTML.

### 5.4 Modes d'exécution

- **Multiprocess** (défaut, recommandé) : `USE_MULTIPROCESS = True`
- **Séquentiel** (debug, profiling) : `USE_MULTIPROCESS = False`

### 5.4bis Loi d'ajustement statistique — `LOI_AJUSTEMENT`

Sélectionne la loi *a priori* inférée par L-moments pour le quantile SRE/SDF :

- `'kappa4'` (défaut) — loi de Hosking à 4 paramètres.
- `'rayleigh_gen'` — **loi de Rayleigh généralisée** (Kundu & Raqab,
  `F(x;α,λ)=(1−e^(−(λx)²))^α`), modèle exact de l'excitation gaussienne.
  Ajustement par **L-moments modifiés (MLME, Kundu & Raqab §6 éq. 18-20)** :
  la transformée `Y=X²` suit une exponentielle généralisée `GE(α,λ²)` à
  L-moments en forme close (digamma `ψ`) ; la forme `α` est la racine de
  `[ψ(2α+1)−ψ(α+1)]/[ψ(α+1)−ψ(1)] = l₂/l₁` (par `brentq`) et l'échelle
  `λ = √([ψ(α+1)−ψ(1)]/l₁)`. **PPF analytique exacte**. D'après l'article CFM 2025 (Clou/Lelan), plus
  stable que Kappa4 entre degrés de liberté pour signaux gaussiens et
  non-gaussiens. Famille distincte de Kappa4 (les deux sont opposées dans
  l'article). La loi retenue apparaît dans le cadre paramètres HTML et une
  colonne `loi` du CSV de fit.

### 5.4ter Mode démo — `mbd_demo_v1.py`

Programme court réutilisant `main()` du module principal. Génère un signal
gaussien ou non-gaussien (méthode ZMNL Hermite de l'article CFM 2025 §4.3,
pilotée par `KURTOSIS`/`SKEWNESS`) puis lance le calcul MBD complet avec les
mêmes visuels. `python mbd_demo_v1.py` (config en tête de fichier).

### 5.4quater Tests unitaires — `tests_unitaires/`

Le dossier `tests_unitaires/` contient un test par fonction critique. Chaque
test construit une donnée dont la réponse exacte est connue d'avance (théorie
fermée ou construction maîtrisée), appelle la fonction et compare le résultat
à la référence, puis produit un PNG avec bandeau PASS/FAIL et une explication
en clair (rôle de la fonction, ce qui est prouvé, critère de réussite).

Fonctions couvertes (un fichier `test_*.py` chacune) :

| Test | Fonction vérifiée |
|------|-------------------|
| `test_extraire_caracteristiques.py` | extraction des features par bloc (SECTION 4) |
| `test_reponse_sdof.py` | réponse SDOF FOH Smallwood (SECTION 5) |
| `test_kappa4.py` | ajustement Kappa4 L-moments analytiques (SECTION 6) |
| `test_lmoments.py` | calcul des L-moments empiriques (SECTION 6) |
| `test_rayleigh_gen.py` | loi de Rayleigh généralisée MLME (SECTION 6) |
| `test_sre_analytique.py` | SRE / SRX / SDF depuis la DSP (SECTION 7) |
| `test_rainflow.py` | comptage rainflow ASTM E1049 (SECTION 8) |
| `test_projection.py` | projection longue durée TVE / TCL (SECTION 9) |
| `test_quality_gate_iid.py` | Quality Gate IID (SECTION 9bis) |

Le module `_th.py` regroupe les utilitaires de tracé partagés.

```bash
python tests_unitaires/run_all.py
```

`run_all.py` exécute tous les `test_*.py`, agrège les résultats dans
`tests_unitaires/_resultats/index.html` (un PNG par test) et renvoie le code
de sortie `0` si tous les tests passent, `1` sinon (exploitable en
intégration continue).

### 5.4quinquies Échelles des graphiques HTML

Les rapports HTML offrent, par sous-graphe, des boutons interactifs
**X-lin / X-log / Y-lin / Y-log**. Le cadre des paramètres est désormais un
bloc statique **repliable** en tête de page (n'occulte plus le graphique).

### 5.5 Diagnostic Kappa4

Activer `KAPPA4_DEBUG_ANALYTIC = True` pour exporter un CSV `Kappa4_Debug_*.csv` contenant pour chaque (f₀, classe) : L-moments empiriques, warm-start fsolve, résidus de l'optimiseur, g₁/g₂, exit_reason.

### 5.6 Performance typique

Sur un signal de 36 Ms (≈ 5 min à 12,8 kHz), 500 fréquences, 10 workers : **2–5 min** par run avec K=1, **5–15 min** avec K=8.

---

## 6. Paramètres utilisateur — description complète

### 6.1 Fichier d'entrée

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `CSV_FILEPATH` | str | (chemin local) | Chemin absolu vers le CSV d'entrée |
| `CSV_SKIP_ROWS` | int | 10 | Lignes d'en-tête à sauter — ≥ 0 |
| `CSV_DELIMITER` | str | `";"` | Délimiteur — typiquement `";"` (FR) ou `","` (US) |

### 6.2 Paramètres SDOF — Réf. [1] §C.3, [2] eq. 3

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `Q` | float | 10 | **Coefficient de surtension** Q = 1/(2ξ). Plage usuelle **5–50** ; 10 = standard pour étalon mécanique d'équipement embarqué (≈ ξ=5%). |
| `TB` | float | 1.00 | **Durée de bloc T_b** (s). Pas de bloc MBD. Doit être ≫ 1/f_min mais ≪ durée totale. Plage **0.05–10 s**. |

### 6.3 Spectre de fréquences

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `F0_MIN` | float | 1 | Fréquence min (Hz) — ≥ 1/T_b recommandé |
| `F0_MAX` | float | 500 | Fréquence max (Hz) — < fs/4 (anti-repliement) |
| `DELTA_F0` | float | 1 | Pas en fréquence (Hz). Détermine `num_f0 = (F0_MAX-F0_MIN)·DELTA_F0 + 1` |

### 6.4 Classification K-Means

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `N_CLUSTERS` | int | 1 | Nombre de classes a priori. **1 = mono-classe** (pas de partition). Plage 1–10 ; au-delà, statistiques par classe trop pauvres. |
| `AUTO_SELECT_K` | bool | False | Si True, choisit K dans `K_RANGE` par maximisation du score Silhouette |
| `K_RANGE` | range | `range(3,9)` | Plage testée pour K optimal |
| `MIN_SAMPLES_PER_CLUSTER` | int | 40 | Si une classe < ce seuil, K est décrémenté automatiquement. Doit rester ≥ `MIN_POINTS_KAPPA4` (40) sinon les fits Kappa4 échouent |
| `FEATURE_FLAGS` | dict | (voir code) | Booléens pour activer/désactiver chaque feature : `mean`, `variance`, `skewness`, `kurtosis`, `rms`, `mav`, `crest_factor`, `autocorr_lag1`, `zcr`, `dominant_freq`, `spectral_centroid`, `spectral_spread` |

### 6.5 Quantile cible (PPF) — Réf. [1] §C.7, [2] Tab. 1

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `PROBABILITE_CIBLE` | float | 0.51 | Probabilité de référence si Cunnane désactivé. Plage **(0,1)** stricte. La norme préconise α=0.1 (10%, criticité faible), 0.01 (criticité moyenne), 0.001 (criticité forte) ⇒ `PROBABILITE_CIBLE = 1 - α` |
| `OPTION_CUNNANE` | bool | True | Si True, p_eff = (N - a) / (N + 1 - 2a) au lieu de PROBABILITE_CIBLE. Recommandé par [1] §C.7 |
| `CUNNANE_A` | float | 0.4 | Constante Cunnane. **0.4** = standard hydrologique (ν=0.4 dans la norme). Plage 0.0–0.5 |

### 6.6 SRE / SRX depuis DSP — Réf. [4] §5.4.2 et §5.4.3

Branche analytique de comparaison au MBD-Kappa4, calculée depuis la DSP Welch
du signal et intégration `z_eff² = ∫ Pxx(f)·|Fd(f,f₀,Q)|² df`. Hypothèse
narrow-band gaussienne : `n₀⁺ ≈ f₀` ([4] §5.4.3, juste après [5.2]).

- **SRE** (Spectre de Réponse Extrême) = `(2π·f₀)²·z_eff·√(2·ln(n₀⁺·T))` — cas particulier de [5.2] avec α → 1/(n₀⁺·T) ; pic moyen attendu sur T.
- **SRX(α)** = `(2π·f₀)²·z_eff·√(-2·ln(1-(1-α)^(1/(n₀⁺·T))))` — équation [5.2] de [4] §5.4.3.

Deux niveaux α sont calculés simultanément, qui correspondent aux deux usages métier documentés en [4] §5.4.3 :

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `ALPHA_SRX_LOW`  | float | **dérivé** | **DÉRIVÉ automatiquement** : `ALPHA_SRX_LOW = 1 - ALFA_PROJECTION`. Garantit l'homogénéité du risque de dépassement avec le SRE MBD projeté (qui utilise la convention opposée *probabilité de non-dépassement*, [1] §C.9). Régler via `ALFA_PROJECTION`. Cas d'usage : **dimensionnement** enveloppe haute. |
| `ALPHA_SRX_HIGH` | float | 0.99 | Risque **élevé** (≈ 99%) → **comparaison vs SRC d'un choc** : si SRX(99%) > SRC, la vibration aléatoire est plus sévère que le choc avec ≥99% de probabilité ([4] fig. 5.3). Plage usuelle 0.9–0.999. Réglable indépendamment. |

### 6.7 SDF — Loi de Basquin / Miner

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `SDF_ENABLED` | bool | True | Active le calcul SDF (rainflow + Bendat). Désactiver pour gain ~30% sur très long signal |
| `SDF_B` | float | 8.0 | **Pente de Basquin b** (N·σ_a^b = C). Plage usuelle métaux **3–14** : alu 3-5, acier 5-8, soudures 3-4, composites 8-14. b=8 = défaut "matériau dur" cf. [2] |
| `SDF_C` | float | 1.0 | **Constante de Basquin C** — fixée à 1 par défaut (analyse relative ; la valeur absolue de SDF dépend de l'application) |

### 6.8 Projection longue durée — Réf. [1] §C.8-C.9

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `ENABLE_PROJECTION` | bool | True | Active la projection T_v |
| `DUREE_PROJECTION` | float | 36_000_000 | T_v (s). 36×10⁶ s ≈ 1 an d'usage (ou 10 000 km à v moyenne typique). Plage **T_mesure × 10² à T_mesure × 10⁸** |
| `ALFA_PROJECTION` | float | 0.9 | Risque de dépassement projeté (1−α dans la norme). 0.9 = 90% non-dépassé. Plage **0.5–0.999** |

### 6.8bis Quality Gate IID — Réf. cahier des charges §3-5

Brique de validation post-traitement (couplage lâche, vectorisée
numpy/scipy). Vérifie l'hypothèse d'indépendance des blocs temporels —
requise par l'inférence Kappa-4 via L-moments — avant de faire confiance au
SRE/SDF. Pour chaque f₀ : autocorrélation lag-1 de **Spearman sur les rangs**
(robuste aux extrêmes Kappa-4) + **test des suites de Wald-Wolfowitz**
(binarisation vs médiane). Calcul global par f₀ **et** par classe K-Means.
Verdict 🟢 GO / 🟡 WARNING (bande isolée, souvent basses fréquences) /
🔴 NO-GO (dépendance généralisée → message correctif imposant d'augmenter
`TB`). Le run n'est jamais interrompu : les f₀ fautives sont taguées en
confiance réduite (visible au survol du rapport principal).

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `IID_GATE_ENABLED` | bool | True | Active la vérification. `False` ⇒ comportement strictement identique à l'historique (aucune colonne / section ajoutée) |
| `IID_RHO_MAX` | float | 0.2 | Seuil \|ρ\| Spearman lag-1 acceptable. Plage usuelle 0.1–0.3 |
| `IID_PVALUE_MIN` | float | 0.05 | Seuil de p-value du test des suites. Sous ce seuil ⇒ rejet de l'hypothèse d'indépendance |
| `IID_FAIL_FRAC_MAX` | float | 0.05 | Fraction max de f₀ hors-tolérance pour rester 🟢 GO (5% du spectre) |
| `IID_NOGO_FRAC` | float | 0.30 | Au-delà de cette fraction de f₀ en échec ⇒ 🔴 NO-GO. Entre `IID_FAIL_FRAC_MAX` et cette valeur ⇒ 🟡 WARNING |
| `IID_MIN_N` | int | 20 | Taille d'échantillon minimale pour qu'un test soit jugé fiable (sinon métriques NaN, colonne ignorée du verdict) |

### 6.9 Numérique / runtime

| Paramètre | Type | Défaut | Plage / Notes |
|-----------|------|--------|--------------|
| `RANDOM_SEED` | int | 53 | Graine RNG (KMeans, etc.) — reproductibilité |
| `MIN_POINTS_KAPPA4` | int | 40 | Seuil min pour tenter un fit Kappa4. < 40 → fit refusé (instabilité L-moments). [1] §C.9 demande au moins n=50 pour que LAR converge |
| `OUTPUT_FOLDER` | str | `"mbd_simple_output"` | Dossier de sortie (créé si absent) |
| `USE_MULTIPROCESS` | bool | True | Active le pool multiprocess sur la boucle f₀ |
| `N_WORKERS` | int / None | 10 | Nombre de workers. `None` = `psutil.cpu_count(logical=False)` ou `os.cpu_count()-1` |
| `KAPPA4_METHOD` | str | `'direct_pwm_analytic'` | Seule méthode disponible en V3 |
| `KAPPA4_ANALYTIC_XTOL` | float | 1.49e-8 | Tolérance fsolve. Plage 1e-12 à 1e-4. Plus serré = plus précis mais plus lent |
| `KAPPA4_DEBUG_ANALYTIC` | bool | False | Si True, exporte le CSV diagnostic Kappa4_Debug_*.csv |

---

## 7. Fichiers de sortie

Tous dans `OUTPUT_FOLDER`, suffixés par un **tag descriptif** (ex. `v3p2_f1-500_Q10_Tb1p28_b8_T36Ms_a0p9_20260503_155038`) qui encode tous les paramètres physiquement significatifs.

| Fichier | Contenu |
|---------|---------|
| `params_*.json` | Sidecar JSON : tous les paramètres du run (build_run_meta) |
| `SRE_*.csv` | SRC, SRE Kappa4, SRE DSP (NORMDEF §5.4.2), SRX α_low, SRX α_high (NORMDEF §5.4.3 eq. [5.2]), + colonnes Quality Gate IID (ρ lag-1, p-value runs, statut) par fréquence |
| `SDF_*.csv` | SDF rainflow global, SDF MBD-AnnexeC (Σ D_bloc), SDF Bendat spectral, RMSE/MSDI moyen Kappa4-Dmg, + colonnes Quality Gate IID branche dommage |
| `SDF_Kappa4_Fit_*.csv` | Diagnostic fit Kappa4 sur D_bloc : ξ, α, k, h, t3, t4, RMSE, MSDI par (f₀, classe) |
| `IID_QualityGate_*.csv` | (si `IID_GATE_ENABLED`) Diagnostic IID par (f₀, classe) — `Classe=-1` = global toutes classes : ρ Spearman lag-1, p-value test des suites, n, statut, branches SRE et SDF |
| `SRE_Projection_*.csv` | SRE projeté, M, SDF projetés (TCL empirique / K4-LogN / DSP) |
| `Kappa4_Debug_*.csv` | (optionnel) Trace fsolve pour chaque fit |
| `Rapport_*.html` | Rapport principal Plotly : 4 graphiques (SRC+SRE / Comparatif 5 courbes / SDF / Projection) ; statut IID dans le cadre paramètres, confiance IID au survol du SRE |
| `Rapport_Details_*.html` | Rapport diagnostic : τ₃/τ₄, CDF par classe (dropdown f₀), comparatif SRE+MSDI, comparatif SDF projeté, + spectres ρ/p-value IID vs f₀ |

**Convention de suffixage des colonnes CSV** : chaque grandeur porte les paramètres dont elle dépend physiquement.
- SRC dépend de Q, T_b → `SRC_Q10_Tb1p28s`
- SRE Kappa4 dépend de Q, T_b, P_cible, T_mes → `SRE_Kappa4_Q10_Tb1p28s_P0p51_Tmes281p25s`
- SRE DSP (NORMDEF §5.4.2) dépend de Q, T_b, T_mes → `SRE_DSP_NormDef_Q10_Tb1p28s_Tmes281p25s`
- SRX α_low / α_high (NORMDEF §5.4.3) ajoutent le risque α → `SRX_alpha_low_Q10_Tb1p28s_aL0p01_Tmes281p25s` / `SRX_alpha_high_Q10_Tb1p28s_aH0p99_Tmes281p25s`
- SDF dépend de b, Q, T_b, T_mes → `SDF_Temporel_Rainflow_b8_Q10_Tb1p28s_Tmes281p25s`
- SRE projeté ajoute T_proj, α → `SRE_Projection_Q10_Tb1p28s_P0p51_T36Ms_a0p9`
- Quality Gate IID dépend de Q, T_b (réponse SDOF) → `IID_rho_lag1_Q10_Tb1p28s`

Cela permet d'**empiler des CSV de runs avec paramètres différents** sans collision de noms.

---

## 8. Architecture du code (sections internes)

Le fichier `mbd_simple-multi-process_v3_4.py` est mono-fichier autoporteur, organisé en 13 sections numérotées :

| Section | Lignes (approx) | Rôle |
|---------|-----------------|------|
| 1 | CONFIGURATION | Constantes utilisateur (modifier ici avant lancement) |
| 2 | LOGGING | `logging.basicConfig` |
| 3 | IMPORT CSV | `importer_signal_csv()` — auto-encodage |
| 4 | FEATURES & MAXIMA | `extraire_caracteristiques()`, `_trouver_diviseurs()` |
| 5 | RÉPONSE SDOF FOH | `reponse_sdof()` — Smallwood récursif (norme §C.3) |
| 6 | KAPPA4 ANALYTIQUE | `_calculer_pwm`, `calculer_lmoments`, `_g_functions`, `_fit_loc_scale`, `_tau3_tau4_from_kh_analytic`, `ajuster_kappa4_pwm_analytic`, `kappa4_rmse_msdi`, `kappa4_ppf` |
| 7 | SRE / SDF DSP | `calculer_sre_analytique()` — Welch + Rayleigh + Gumbel + Bendat |
| 8 | SDF RAINFLOW | `_rainflow_extrema_numba`, `_rainflow_stack_damage_numba`, `_rainflow_damage`, `calculer_sdf_rainflow`, `calculer_sdf_per_bloc` |
| 9 | PROJECTION L-MOMENTS | `calculer_projection_lmoments`, `calculer_projection_sdf_tcl`, `calculer_projection_dmg_kappa4`, `_kappa4_mean_var` |
| 9bis | QUALITY GATE IID | `quality_gate_iid`, `_iid_spearman_lag1`, `_iid_runs_pvalue`, `_iid_verdict`, `agreger_quality_gate` — couplage lâche, vectorisé |
| 10 | TRAITEMENT f₀ | `traiter_f0()` — pipeline complet pour une fréquence ; workers multiprocessing |
| 11 | EXPORTS CSV | `exporter_csv_sre/sdf/projection/sdf_kappa4_fit/iid/debug_analytic`, helpers de suffixage |
| 12 | RAPPORTS HTML | `generer_html()`, `generer_html_details()` |
| 13 | MAIN | `main()`, `main_mp()`, point d'entrée |

