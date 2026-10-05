# MyMBD — Outil d'analyse vibratoire Méthode des Blocs Disjoints « MBD » / Kappa4 & Rayleigh généralisée / SRE-SRX / SDF

> **Auteur** : Guillaume LE ROUSSEAU
> **Programme principal documenté** : `mbd_simple-multi-process_v3_6.py`
> **Version courante** : V3.6
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


---

## 1. Présentation et finalité

Le programme calcule, à partir d'un signal d'accélération mesuré (essai sur véhicule, banc, enregistrement de roulage), **trois grandeurs spectrales** fonction d'une fréquence propre f₀ :

- **SRC(f₀)** — Spectre de Réponse au Choc : maximum de la réponse d'un oscillateur à 1 degré de liberté (1-DDL) ;
- **SRE(f₀)** — Spectre de Réponse Extrême : quantile haut des maxima par bloc, méthode MBD ;
- **SDF(f₀)** — Spectre de Dommage par Fatigue : loi de Basquin, cumul de Miner, comptage rainflow.

Il fournit aussi leur **projection à la durée de vie visée** (T_proj, par défaut 36×10⁶ s = 10 000 h).

Le calcul applique la **Méthode des Blocs Disjoints (MBD)** non corrélée de la norme **NF X50-144-3 (2021), Annexe C**, avec l'apport de **B. Colin (COFREND 2023)** [2] : une **loi Kappa4 à 4 paramètres (Hosking)** sert de loi unique, à la place de l'arbre de choix entre 7 lois de la norme. Une seconde loi est disponible : la **Rayleigh généralisée à 2 paramètres** (A. Clou & P. Lelan, DGA TT / CFM 2025) [3], de domaine plus restreint mais d'estimation plus stable d'une fréquence à l'autre.

Autour de ce cœur, le programme apporte :

- une **projection longue durée** : théorie des valeurs extrêmes pour le SRE (F^M), théorème central limite pour le SDF ;
- une **classification K-Means** des blocs d'excitation, pour traiter un signal non stationnaire en classes localement stationnaires, et la **synthèse** de ces classes ;
- deux **contrôles de validité** dont le résultat est expliqué en clair : la stationnarité du signal, et l'indépendance des blocs (contrat IID) qu'exige la méthode ;
- un **comptage rainflow ASTM E1049-85 strict**, accéléré par Numba ;
- une **branche analytique de comparaison** depuis la densité spectrale : SRE et SRX de la PR NORMDEF 0101, SDF de Bendat-Lalanne.

---

## 2. Cadre normatif et références documentaires

### Références principales

| # | Document | Fichier local | Rôle |
|---|----------|---------------|------|
| **[1]** | **NF X50-144-3 (2021)** — *Démonstration de la tenue aux environnements mécaniques, Partie 3 : Personnalisation* | non présent dans le dépôt (document AFNOR) | Cadre normatif, Annexe C (méthode MBD) |
| **[2]** | **B. Colin (Nexter Systems / COFREND 2023)** — *Maintenance prévisionnelle des équipements critiques, embarqués sur systèmes d'armes terrestres*, e-Journal of NDT, doi:10.58286/28496 | `_docs_MBD-Kappa4_MBD&KAPPA4_ME3E2_B_Colin.pdf` | Loi Kappa4 par L-moments, projection, synthèse stochastique |
| **[3]** | **A. Clou & P. Lelan (DGA TT / CFM 2025)** — *Development of statistical methods for vibration analysis* | `_docs_MBD-Rayleigh_ASTE_2025-12-08_Article_CFM_2025_CLOU_LELAN.pdf` | Limites de la Kappa4, loi de Rayleigh généralisée |
| **[4]** | **PR NORMDEF 0101 (DGA 2009)** — *Personnalisation des essais en environnement mécanique* | `_docs-SRX_prnormdef0101pcemv12versionaste.pdf` (extrait p. 31-39 : `_docs-SRX_extract_31-39_…pdf`) | SRE (§5.4.2) et SRX (§5.4.3, éq. [5.2]) |
| **[5]** | **B. Colin (MI0460, 2008)** — *Définition d'un Spectre de Réponse à risque de dépassement (SRX)* | `_docs-SRX_Colin_mi0460-2008.pdf` | Origine du modèle SRX non asymptotique |
| **[6]** | **Kundu & Raqab** — *Generalized Rayleigh Distribution: Different Methods of Estimations* | `_docs_MBD-Rayleigh_KUNDUetRAQABpaper96.pdf` | Estimation de la Rayleigh généralisée par L-moments modifiés |
| **[7]** | **W. Asquith** — *Distributional Analysis with L-moment Statistics* | `_docs-Distributional_Analysis_with_L-moment_Statistics_…pdf` | Ouvrage de référence sur les L-moments |

### Pages clés

**[1] NF X50-144-3 — Annexe C :**
- §C.2 (Figure C.1) : σ(t) = K · z(t), K = 1 forfaitaire ;
- §C.3 : réponse 1-DDL par formules récursives de Smallwood ;
- §C.5–C.7 : n-échantillons par bloc, tests d'ajustement, position de tracé de Cunnane (ν = 0,4), MSDI ;
- §C.8–C.9 : coefficient d'extrapolation M, critères M > 100 (valeurs extrêmes) et M > 50 (théorème central limite) ;
- §C.10 : synthèse stochastique des classes.

**[2] Colin 2023 :**
- éq. 1 : loi de Basquin N·σᵇ = C ; Tableau 1 : risque α selon la criticité (10 % / 1 % / 0,1 %) ;
- éq. 8–15 et 24–27 : L-moments, moments pondérés, estimateurs non biaisés ;
- éq. 16–20 : L-moments de la Kappa4 par les fonctions g_r ; Figure 11 : domaine de validité de (k, h) ;
- éq. 28–34 : procédure d'ajustement (h*, k*, α*, ξ*) ; éq. 30–31 : répartition et quantile ;
- éq. 35–36 : passage au modèle global (F^M, somme des dommages) ;
- §4.2, éq. 37–38 : classification K-Means et synthèse stochastique des classes.

**[4] PR NORMDEF 0101 :** §5.4.2 (SRE, pic moyen sur T), §5.4.3 (SRX, équations [5.2] et [5.3], figures 5.2 et 5.3).

### Références secondaires (algorithmes)

- **ASTM E1049-85 (2017)** et **AFNOR A03-406** — comptage rainflow.
- **Hosking, J.R.M. (1994)** — *The four-parameter kappa distribution*, IBM J. Res. Dev. 38(3):251–258.
- **Hosking & Wallis (1997)** — *Regional frequency analysis: an approach based on L-moments*, Cambridge University Press.
- **Smallwood, D.O. (1981)** — *An improved recursive formula for calculating shock response spectra*, Shock & Vibration Bulletin 51.
- **Cunnane, C. (1978)** — *Unbiased plotting positions — A review*, J. Hydrology 37.
- **Lalanne, C.** — *Mechanical Vibration and Shock*, vol. 3 (*Random Vibration*) et vol. 4 (*Fatigue Damage*).

### Bibliothèques utilisées

- **NumPy / SciPy** — calcul numérique (`lfilter`, `welch`, `fsolve`, `brentq`, `scipy.stats.kappa4`, `lognorm`) ;
- **scikit-learn** — `KMeans`, `StandardScaler`, `silhouette_score` ;
- **Numba** — compilation du rainflow ;
- **pandas** — lecture et écriture des CSV ;
- **plotly** — rapports HTML interactifs ;
- **tqdm** — barres de progression ; **psutil** (optionnel) — nombre de cœurs physiques ;
- **matplotlib** — uniquement pour les tests unitaires.

---

## 3. Workflow détaillé du programme

```
┌───────────────────────────────────────────────────────────────────┐
│  ENTRÉE : signal CSV  (t, ẍ)                                       │
└───────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 1. Import CSV                      │  importer_signal_csv()
│    encodage et décimale détectés   │  TRIM_DEBUT_S : retrait du début
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 1bis. Contrôle de stationnarité    │  detecter_non_stationnarite()
│    sondes f₀ + CV du RMS par bloc  │  → STATIONNAIRE / NON_STATIONNAIRE
│    + caractère gaussien            │    / DOUTEUX, recommandation
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 2. Découpage en blocs T_b          │  extraire_caracteristiques()
│    bloc = round(T_b·fs) éch.       │  _taille_bloc()
│    + features par bloc             │  fin de signal incomplète ignorée
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 3. Classification K-Means          │  K = 1 si signal stationnaire
│    des blocs d'excitation          │  sinon score Silhouette
│                                    │  → clusters[n_blocs]
└────────────────────────────────────┘
              │
              ▼  pour chaque f₀ ∈ [F0_MIN .. F0_MAX]  (parallèle)   traiter_f0()
┌────────────────────────────────────┐
│ 4A. Réponse 1-DDL z(t)             │  reponse_sdof()
│     Smallwood récursif, amorcée    │  filtre IIR d'ordre 2
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4B. Pseudo-accélération (2πf₀)²·z  │  SRC = max sur le signal
│     Maxima par bloc → {Z_max}      │  branche SRE
│     Rainflow par bloc → {D_p}      │  branche dommage (sur z)
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4C. Contrat IID                    │  quality_gate_iid()
│     ρ de Spearman au décalage 1    │  global et par classe
│     + test des suites              │
└────────────────────────────────────┘
              │
              ▼  pour chaque classe
┌────────────────────────────────────┐
│ 4D. Ajustement de la loi           │  ajuster_loi()
│     L-moments → Kappa4 (2 étapes)  │  kappa4_from_lmoments()
│     ou Rayleigh généralisée        │  ajuster_rayleigh_gen()
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4E. Quantile par classe            │  loi_ppf() à p de Cunnane
│     + qualité : RMSE, MSDI         │  loi_rmse_msdi()
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4F. Synthèse des classes           │  SRE = quantile de Π F_j^{N_j}
│     (durée du signal)              │  SDF = Σ dommages par bloc
│                                    │  classes exclues consignées
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 4bis. Verdict du contrat IID       │  agreger_quality_gate()
│                                    │  → GO / WARNING / NO-GO
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 5. SRE / SRX / SDF analytiques     │  calculer_sre_analytique()
│    depuis la densité spectrale     │  Welch + NORMDEF + Bendat
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 6. Projection longue durée         │  M_j = Occ(j)·T_proj / T_b
│    SRE : quantile de Π F_j^{M_j}   │  quantile_produit_classes()
│    SDF : somme de M blocs (TCL)    │  calculer_projection_sdf_tcl()
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 7. Exports CSV + JSON              │  SECTION 11
│ 8. Rapports HTML                   │  SECTION 12, avec le bloc
│                                    │  « Contrat IID » commenté
└────────────────────────────────────┘
```

---

## 4. Choix techniques et justifications

### 4.1 Pourquoi la Kappa4 plutôt que l'arbre de lois de la norme ?

La norme [1] choisit entre 7 lois selon la taille d'échantillon. L'article [2] propose la Kappa4 comme **loi unique** : elle contient les lois classiques comme cas particuliers (h = −1 logistique généralisée, h = 0 valeurs extrêmes généralisée GEV, h = 1 Pareto généralisée) et couvre une large part du diagramme des L-moments (τ3, τ4). Le même traitement sert pour les maxima (SRE) et pour les dommages par bloc (SDF).

**Limite connue** ([3], Figure 10) : sur un signal gaussien, le SRE Kappa4 fluctue fortement d'une fréquence à l'autre. Le programme fournit donc en parallèle la loi de Rayleigh généralisée et la branche analytique (SRX, Bendat) pour recoupement — voir aussi §9.

### 4.2 Identification de la Kappa4 : deux étapes et contrôle du domaine

Fonction `kappa4_from_lmoments`. L'entrée est le quadruplet (L1, L2, τ3, τ4) calculé sur l'échantillon par les moments pondérés non biaisés ([2] éq. 24-27).

**Étape 1 — la forme (k, h)** ne dépend que des rapports τ3, τ4. On résout le système

```
τ3(k, h) = τ3 mesuré        τ4(k, h) = τ4 mesuré
```

où τ3(k, h) et τ4(k, h) sont les expressions analytiques de [2] (éq. 18-19) par les fonctions g_r de Hosking (éq. 20.1 à 20.3), résolues par `scipy.optimize.fsolve`. C'est l'équivalent continu de la procédure de l'article (éq. 28 : minimum de distance sur une famille de polynômes à h discret) : à l'intérieur du domaine, la distance minimale est nulle et les deux procédures coïncident. Trois garde-fous :

1. convergence de `fsolve` **et** résidu recalculé sous `KAPPA4_RESIDUAL_TOL`, sinon nouvel essai sur d'autres amorces (`KAPPA4_RETRY_WARM_STARTS`) ;
2. **domaine de h** : une racine h < `KAPPA4_H_MIN` (−1) est rejetée — sous cette borne, plusieurs couples (k, h) reproduisent le même (τ3, τ4) avec des queues de loi différentes ;
3. **point hors domaine** : si (τ3, τ4) est au-dessus de la courbe h = −1, d'équation τ4 = (1 + 5τ3²)/6, aucune Kappa4 admissible ne le reproduit. La loi retenue est celle du **bord** : h* = −1, au point de la courbe **le plus proche en distance euclidienne** ([2] éq. 28-29), obtenu en forme close (racine d'une cubique). Le cas est repéré par `fail_reason = 'ok_h_borne'`.

**Étape 2 — la position et l'échelle** se déduisent en forme close ([2] éq. 33-34) :

```
α = k·L2 / (g1 − g2)          ξ = L1 − (α/k)·(1 − g1)
```

avec passage à la limite k → 0 (développement des g_r) pour les lois de la ligne k = 0 (Gumbel, logistique, exponentielle).

Les fonctions quantile et de répartition sont celles de [2] (éq. 30-31). `scipy.stats.kappa4` est utilisé partout sauf près de k = 0 avec h ≤ 0, où il renvoie NaN : le programme bascule alors sur ses propres formules exactes.

### 4.3 Réponse 1-DDL : Smallwood et amorçage

Conformément à [1] §C.3 et [2] §4.1, la réponse z(t) est calculée par le filtre récursif de **Smallwood (1981)**, exact pour une excitation linéaire entre deux échantillons.

**Amorçage** (`SDOF_AMORCAGE`, actif par défaut). L'état initial naturel d'un filtre récursif est celui d'une excitation constante égale à la première valeur du signal : l'oscillateur part alors d'un déplacement statique et libère une oscillation libre parasite, d'amplitude ≈ |x[0]| en pseudo-accélération, qui s'amortit avec la constante de temps τ = Q/(π·f₀) (0,64 s à 5 Hz pour Q = 10). Aux basses fréquences ce parasite fixe le maximum du premier bloc, donc le SRC. Avec l'amorçage, le filtre est d'abord lancé sur le début du signal renversé dans le temps pendant `SDOF_AMORCAGE_N_TAU` constantes de temps ; la sortie garde la longueur de l'entrée.

### 4.4 Rainflow : ASTM E1049 strict

Comptage par pile, équivalent à l'algorithme 4 points de Downing-Socie. Les résidus non fermés comptent pour des demi-cycles. Convention **amplitude** : σ_a = étendue/2, cohérente avec Basquin sous la forme N·σ_aᵇ = C.

### 4.5 Le dommage est calculé sur z(t)

[1] §C.2 : σ(t) = K·z(t), K = 1. Le rainflow s'applique au **déplacement relatif** z(t), pas à la pseudo-accélération (réservée au SRC et au SRE).

### 4.6 Granularité du dommage : le bloc T_b

Le rainflow est mené à l'intérieur de chaque bloc : c'est la variable D_p de la méthode ([2] §4.1). Le SDF MBD est la **somme** des D_p ([2] éq. 36). Il est légèrement inférieur au rainflow sur tout le signal, qui voit en plus les grands cycles à cheval sur plusieurs blocs.

### 4.7 Parallélisation

La boucle sur f₀ est répartie sur `N_WORKERS` processus. Le signal est placé en mémoire partagée (`multiprocessing.shared_memory`) pour ne pas être recopié à chaque tâche.

### 4.8 SRE et SRX analytiques : deux niveaux de risque

La branche `calculer_sre_analytique` intègre la densité spectrale (Welch) contre la fonction de transfert de l'oscillateur, puis applique [4] :

- **SRE** (§5.4.2) : `(2π·f₀)²·z_eff·√(2·ln(n₀⁺·T))` ;
- **SRX(α)** (§5.4.3, éq. [5.2]) : `(2π·f₀)²·z_eff·√(−2·ln(1 − (1 − α)^(1/(n₀⁺·T))))`, avec n₀⁺ ≈ f₀.

Deux niveaux α sont calculés : α faible pour le **dimensionnement** (enveloppe haute), α élevé (0,99) pour la **comparaison à un choc**.

**Conventions opposées** : [4] utilise α comme probabilité de *dépassement*, [1] et `ALFA_PROJECTION` comme probabilité de *non-dépassement*. Le code aligne les deux : `ALPHA_SRX_LOW = 1 − ALFA_PROJECTION`.

Cette branche suppose un signal **stationnaire gaussien**. Elle fait référence dans ce cas, et sous-estime la queue sinon.

### 4.9 Un seul fil BLAS par processus

Les variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc. sont fixées à 1 **avant** l'import de numpy, sinon chaque processus tenterait d'utiliser tous les cœurs.

### 4.10 Projection longue durée du SRE

Le maximum sur la durée de vie est le plus grand de M maxima de blocs supposés indépendants ([2] éq. 35) : F_Zsup = F_Zmax^M, avec M = Occ(j)·T_proj/T_b.

1. **`'puissance'`** (défaut) : SRE projeté = quantile de la loi ajustée à la probabilité α^(1/M). C'est l'application directe de l'éq. 35.
2. **`'gev_domaines'`** : la loi est ré-ajustée dans la famille GEV (h figé à 0) et projetée par max-stabilité, en forme close. Le signe de k désigne le domaine — Gumbel (k ≈ 0), Fréchet (k < 0), Weibull négative (k > 0) — exporté dans la colonne `GEV_domaine`. Voir la réserve du §9.

### 4.11 Découpage en blocs et coefficient M

Un bloc compte **toujours** `round(T_b·fs)` échantillons (`_taille_bloc`). Si la longueur du signal n'en est pas un multiple, la fin incomplète — moins d'un bloc — est ignorée. La durée de bloc est une donnée de la méthode : elle fixe M = T_proj / T_b et ne doit pas dépendre de la longueur du signal. La durée effective (taille entière / fs) est celle qui entre dans M ; elle est écrite au journal, dans le cadre des rapports et dans `params_*.json` (`Tb_effectif_s`, `n_blocs`).

### 4.12 Contrôle de stationnarité

Calculé avant la boucle des fréquences (`detecter_non_stationnarite`). La méthode suppose des maxima de blocs indépendants et de même loi ; deux causes de violation appellent des remèdes **opposés** :

- T_b trop court devant la mémoire de l'oscillateur → il faut **allonger T_b** ;
- enveloppe lente du signal (changement de régime, dérive) → il faut **classer les blocs**.

Trois indicateurs :

| Indicateur | Mesure | Seuil |
|---|---|---|
| (a) sondes | fraction de fréquences sondes dont les maxima par bloc échouent au contrat IID ; les sondes sont placées à f₀ ≥ 3·Q/(π·T_b), là où la mémoire de l'oscillateur ne peut pas à elle seule expliquer un échec | `STATIONNARITE_FRAC_ECHEC` |
| (b) énergie | coefficient de variation du RMS de l'excitation d'un bloc à l'autre | `STATIONNARITE_CV_RMS_MAX` |
| (c) forme | kurtosis et asymétrie du signal (caractère gaussien) | `STATIONNARITE_KURT_TOL`, `STATIONNARITE_SKEW_TOL` |

Verdict : **NON_STATIONNAIRE** si (a) et (b) dépassent leur seuil ; **STATIONNAIRE** si aucun ; **DOUTEUX** sinon. Le verdict informe l'utilisateur ; la seule décision automatique est K = 1 quand `AUTO_SELECT_K` et `AUTO_K_SELON_STATIONNARITE` sont actifs et que le verdict est STATIONNAIRE (le score Silhouette, non défini pour K = 1, découperait sinon un signal homogène).

### 4.13 Synthèse des classes

Les classes sont des situations réparties en série ([2] Figure 15). Le maximum sur la durée est le plus grand des maxima de toutes les classes :

```
P(Z_sup ≤ z) = Π_j F_j(z)^{M_j}                    ([1] §C.10, [2] éq. 37)
```

Le quantile α de ce produit est obtenu en résolvant Σ_j M_j·ln F_j(z) = ln α (`quantile_produit_classes`, `_quantile_produit`), avec un ln F précis près de 1 (`_loi_logcdf`).

- **En projection** : M_j = Occ(j)·T_proj/T_b, α = `ALFA_PROJECTION`.
- **À la durée du signal** : M_j = nombre de blocs mesurés de la classe, et le niveau visé est celui du quantile mono-classe appliqué à l'ensemble des N blocs, Π F_j^{N_j} = p^N — ce qui redonne exactement F(z) = p s'il n'y a qu'une classe.

Le mode `SYNTHESE_CLASSES = 'max'` (maximum des quantiles par classe) reste disponible ; il est toujours inférieur ou égal au produit, donc non conservatif.

**Classes exclues.** Une classe qui contient des blocs mais n'a pas de loi exploitable (moins de `MIN_POINTS_KAPPA4` blocs, ajustement en échec) ne peut pas entrer dans la synthèse. Le spectre est alors **sous-estimé**. Le cas est signalé au journal, dans le cadre des rapports HTML (ligne « ⚠ Classes exclues ») et dans les colonnes `Classes_exclues` / `Occurrence_exclue` des CSV.

Pour le dommage, la synthèse est la **somme** des contributions de classes (convolution des densités, [2] éq. 38) : moyennes et variances s'additionnent.

### 4.14 Contrat IID et son commentaire

La méthode suppose que les maxima par bloc (et les dommages par bloc) sont **indépendants et identiquement distribués** : c'est ce qui autorise l'ajustement par L-moments et la projection F^M ([2] éq. 35-36). Deux tests par fréquence, sur la série des blocs dans l'ordre du temps (`quality_gate_iid`) :

1. corrélation de **Spearman** entre un bloc et le suivant, calculée sur les rangs — échec si |ρ| > `IID_RHO_MAX` ;
2. **test des suites** de Wald-Wolfowitz par rapport à la médiane — échec si p-value < `IID_PVALUE_MIN`.

Verdict global (`agreger_quality_gate`) selon la fraction de fréquences en échec : GO, WARNING, NO-GO. Le calcul n'est jamais interrompu.

Le verdict seul ne dit ni quel test échoue, ni où, ni pourquoi. Les deux rapports HTML contiennent donc un bloc **« Contrat IID — explication des résultats »** (`commenter_quality_gate`) :

| Rubrique | Contenu |
|---|---|
| Verdict | statut, nombre de fréquences en échec, règle de décision |
| Données testées | nombre de blocs, durée de bloc effective |
| Résultat test par test | part de la corrélation et du test des suites, valeurs extrêmes, signe de ρ |
| Localisation des échecs | bandes de fréquences contiguës |
| Part attribuable au hasard | fraction de fréquences en échec que produisent des blocs **parfaitement indépendants** (les deux tests ont un taux de fausse alarme), comparée à la fraction observée |
| Cause probable | mémoire de l'oscillateur (τ = Q/(π·f₀) > T_b/3), enveloppe lente du signal, ou fluctuation statistique |
| Conséquence, action | ce que cela change pour le SRE ; allonger T_b, classer, ou ne rien faire |
| Par classe | si K > 1 : l'indépendance à l'intérieur de chaque classe, celle qui conditionne l'ajustement |
| Sondes de stationnarité | résultat des sondes du §4.12 |

---

## 5. Guide d'utilisation

### 5.1 Installation

```bash
# Python 3.10+
pip install numpy scipy scikit-learn pandas tqdm plotly numba psutil
pip install matplotlib        # uniquement pour les tests unitaires
```

### 5.2 Format du fichier CSV d'entrée

Deux colonnes : **temps (s)** et **accélération (m/s²)**.

```
... lignes d'en-tête (nombre = CSV_SKIP_ROWS) ...
0.000000000;0.0345
0.000078125;0.0382
```

- Délimiteur : paramètre `CSV_DELIMITER` (`;` ou `,`).
- Séparateur décimal (`,` ou `.`) et encodage : détectés automatiquement.
- Les lignes illisibles sont retirées.
- La fréquence d'échantillonnage est calculée comme `1/mean(diff(t))`.

### 5.3 Lancer un calcul

Modifier les constantes en tête de `mbd_simple-multi-process_v3_6.py` (SECTION 1), puis :

```bash
python mbd_simple-multi-process_v3_6.py
```

Chaque calcul crée son **sous-dossier** dans `OUTPUT_FOLDER` :

```
<AAAAMMJJ_HHMMSS>_<nom du fichier ≤35 car.>_f<min>-<max>_Q<Q>_Tb<Tb>_b<b>_T<Tproj>_a<alfa>
ex. : 20261002_224557_signal_asymlaplace_f5-400_Q10_Tb1p28_b8_T36Ms_a0p9/
```

### 5.4 Lire les résultats : par où commencer

1. Ouvrir `Rapport_*.html`. Le cadre **« Paramètres du calcul »** rappelle les réglages, le découpage obtenu, le verdict de stationnarité et sa recommandation.
2. Lire le bloc **« Contrat IID »** juste en dessous : il dit si les résultats appellent une réserve, et laquelle.
3. Graphe 2 : comparer le SRE MBD (trait plein = durée du signal, tirets = projeté) au SRE analytique et au SRX.
4. En cas de doute sur une fréquence, ouvrir `Rapport_Details_*.html` : diagramme des L-moments, répartitions empirique et ajustée par classe (sélecteur de fréquence), indicateur MSDI.

### 5.5 Quelle configuration pour quel signal ?

| Signal | Réglage conseillé |
|---|---|
| Stationnaire | `N_CLUSTERS = 1`, ou `AUTO_SELECT_K = True` (K = 1 sera retenu) |
| Non stationnaire (régimes, dérive) | `AUTO_SELECT_K = True` ; sans feature activée, rms + kurtosis + crest_factor sont utilisées d'office ; `SYNTHESE_CLASSES = 'produit'` |
| Verdict DOUTEUX | calculer les deux voies et comparer les SRE projetés |
| Basses fréquences en échec IID | allonger T_b : viser T_b ≥ 3·Q/(π·F0_MIN) |
| Signal de synthèse avec transitoire de début | `TRIM_DEBUT_S` > 0 |

### 5.6 Modes d'exécution

- **Multiprocess** (défaut) : `USE_MULTIPROCESS = True`, `N_WORKERS` au plus égal au nombre de cœurs physiques.
- **Séquentiel** (mise au point) : `USE_MULTIPROCESS = False`.

### 5.7 Loi d'ajustement — `LOI_AJUSTEMENT`

- `'kappa4'` (défaut) — loi de Hosking à 4 paramètres, §4.2.
- `'rayleigh_gen'` — **Rayleigh généralisée** F(x ; α, λ) = (1 − e^(−(λx)²))^α. Ajustement par L-moments modifiés ([6] éq. 18-20) : Y = X² suit une exponentielle généralisée dont les L-moments s'expriment par la fonction digamma ; la forme α est racine de `[ψ(2α+1) − ψ(α+1)] / [ψ(α+1) − ψ(1)] = l₂/l₁`. Quantile analytique exact.

### 5.8 Mode démo — `mbd_demo_v1.py`

Programme court qui génère un signal de nature **connue**, puis lance le calcul complet du module principal : mêmes rapports, mêmes CSV. `python mbd_demo_v1.py`, configuration en tête de fichier.

| Section | Contenu |
|---|---|
| A — signal | `MODE_SIGNAL` : `'stationnaire'` (DSP plate + transformation de Hermite, pilotée par kurtosis et asymétrie), `'phases'` (segments stationnaires mis bout à bout), `'enveloppe'` (signal modulé par une enveloppe RMS lente) |
| B — classification | `N_CLUSTERS`, `AUTO_SELECT_K`, `AUTO_K_SELON_STATIONNARITE`, `K_RANGE`, `MIN_SAMPLES_PER_CLUSTER`, `SYNTHESE_CLASSES`, `FEATURE_FLAGS` |
| B bis — stationnarité | `STATIONNARITE_*` |
| C — calcul | loi, Q, T_b, spectre f₀, Cunnane, SRX, SDF, projection, contrat IID, amorçage, `KAPPA4_H_MIN` |
| D — exécution | `MODULE_PATH`, dossier de sortie, multiprocess |

La démo sert à voir réagir les contrôles : un signal `'stationnaire'` doit donner le verdict STATIONNAIRE et K = 1 ; `'phases'` et `'enveloppe'` doivent donner NON_STATIONNAIRE. Elle range dans le dossier du calcul le signal généré et un `RESUME_demo.txt` (configuration, statistiques atteintes, contrôles constatés), et affiche une synthèse en console.

La démo applique ses réglages au niveau module et enregistre le module principal dans `sys.modules` : `USE_MULTIPROCESS = True` fonctionne aussi sous Windows.

### 5.9 Tests unitaires — `tests_unitaires/`

```bash
python tests_unitaires/run_all.py
```

Chaque test fabrique une donnée dont la **bonne réponse est connue d'avance** (théorie exacte, construction maîtrisée, ou calcul par un code indépendant), appelle la fonction et compare. Il produit une figure : explication en clair à gauche, donnée de test au centre, vérification à droite, bandeau PASS/FAIL. `run_all.py` assemble `tests_unitaires/_resultats/index.html` (sommaire puis fiches, dans l'ordre du calcul) et renvoie le code de sortie 0 si tout passe, 1 sinon.

| Étape | Test | Fonction(s) vérifiée(s) | Étalon |
|---|---|---|---|
| Import, découpage | `test_importer_signal_csv` | `importer_signal_csv` | signal écrit sous 3 conventions de fichier |
| | `test_extraire_caracteristiques` | `extraire_caracteristiques`, `_taille_bloc` | moyennes et maxima imposés ; durée de bloc pour 6 longueurs de signal |
| | `test_features_blocs` | les 12 features | expressions exactes pour un sinus |
| Réponse 1-DDL | `test_reponse_sdof` | `reponse_sdof` | amplification Q, fonction de transfert, solveur `scipy.signal.lsim` |
| | `test_reponse_sdof_amorcage` | option `SDOF_AMORCAGE` | régime établi exact |
| Lois | `test_lmoments` | `calculer_lmoments` | L-moments exacts de 3 lois |
| | `test_kappa4_lmoments` | `kappa4_from_lmoments`, `_tau3_tau4_from_kh_analytic`, `_fit_loc_scale` | intégration numérique de la quantile, 42 lois |
| | `test_kappa4` | `ajuster_kappa4`, `kappa4_ppf` | échantillons de lois connues, cas k = 0 |
| | `test_kappa4_bord_domaine` | domaine de h, `_kappa4_sur_borne_h` | point le plus proche par force brute |
| | `test_kappa4_cdf_ppf` | `_kappa4_cdf_exact`, `_kappa4_ppf_exact`, `_loi_logcdf` | formules de [2] éq. 30-31 |
| | `test_msdi` | `_msdi_queue_droite`, `kappa4_rmse_msdi` | MSDI = 100·ε² |
| | `test_rayleigh_gen` | `ajuster_rayleigh_gen`, `rayleigh_gen_ppf` | inversion exacte de la loi |
| Analytique, dommage | `test_sre_analytique` | `calculer_sre_analytique` | linéarité ; formule de Miles |
| | `test_rainflow` | `calculer_sdf_rainflow` | sinus compté à la main |
| | `test_sdf_par_bloc` | `calculer_sdf_per_bloc` | dommage exact de chaque bloc |
| Projection, synthèse | `test_projection` | `calculer_projection_lmoments` | identité α^(1/M) |
| | `test_projection_gev` | `ajuster_gev_lmoments`, `calculer_projection_gev_domaines` | `scipy.stats.genextreme` |
| | `test_projection_dommage` | `calculer_projection_sdf_tcl`, `calculer_projection_dmg_kappa4` | somme exacte de lois Gamma |
| | `test_synthese_classes` | `quantile_produit_classes`, `_quantile_produit` | simulation de 20 000 durées de vie |
| Contrôles | `test_quality_gate_iid` | `quality_gate_iid`, `_iid_verdict` | séries indépendantes et corrélées |
| | `test_contrat_iid` | `commenter_quality_gate`, `agreger_quality_gate`, `_iid_taux_hasard`, `_iid_bandes` | 4 situations construites |
| | `test_stationnarite` | `detecter_non_stationnarite` | 4 signaux de nature connue |
| Chaîne complète | `test_traiter_f0` | `traiter_f0` | classes imposées, équation du produit |
| | `test_bout_en_bout` | `main` | calcul complet sur un CSV temporaire, 9 contrôles |

Le module `_th.py` porte l'outillage commun (chargement du programme, mise en page, étalons indépendants de la Kappa4).

### 5.10 Diagnostic de l'ajustement

`KAPPA4_DEBUG_ANALYTIC = True` exporte `Kappa4_Debug_*.csv` : pour chaque (f₀, classe), L-moments, amorce, résidus du solveur, g₁ et g₂, raison de sortie.

### 5.11 Temps de calcul

Mesuré sur un signal de 600 s à 12,8 kHz (7,68 millions de points), 396 fréquences, une classe, SDF désactivé, machine à 10 cœurs physiques : **28 s en multiprocess (10 processus)**, 56 s en séquentiel, lecture du CSV comprise. Les deux modes donnent des résultats strictement identiques. Le SDF (rainflow) allonge le calcul.

---

## 6. Paramètres utilisateur — description complète

Les valeurs « défaut » sont celles du fichier livré.

### 6.1 Fichier d'entrée

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `CSV_FILEPATH` | (chemin local) | Chemin du CSV d'entrée |
| `CSV_SKIP_ROWS` | 10 | Lignes d'en-tête à sauter |
| `CSV_DELIMITER` | `";"` | Délimiteur de colonnes |
| `TRIM_DEBUT_S` | 0.0 | Durée (s) retirée en début de signal avant tout calcul |

### 6.2 Oscillateur étalon et bloc

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `Q` | 10 | Surtension Q = 1/(2ξ). Plage usuelle 5–50 |
| `TB` | 1.28 | Durée de bloc T_b (s). Plage usuelle 0,05–10 s. Viser T_b ≥ 3·Q/(π·F0_MIN) et assez de blocs (≥ 40 par classe) |

### 6.3 Spectre de fréquences

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `F0_MIN` | 5 | Fréquence propre minimale (Hz) |
| `F0_MAX` | 400 | Fréquence propre maximale (Hz). Rester sous fs/18 pour une erreur d'amplitude < 1 % (§9) |
| `DELTA_F0` | 1 | Pas en fréquence (Hz) |

### 6.4 Classification et synthèse des classes

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `N_CLUSTERS` | 1 | Nombre de classes imposé quand `AUTO_SELECT_K = False`. Demande au moins une feature activée |
| `AUTO_SELECT_K` | False | True : K = 1 si le signal est jugé stationnaire, sinon K maximisant le score Silhouette |
| `AUTO_K_SELON_STATIONNARITE` | True | Lie `AUTO_SELECT_K` au verdict de stationnarité |
| `K_RANGE` | `range(2, 7)` | Plage testée pour K |
| `MIN_SAMPLES_PER_CLUSTER` | 40 | Si une classe a moins de blocs, K est décrémenté. Garder ≥ `MIN_POINTS_KAPPA4` |
| `SYNTHESE_CLASSES` | `'produit'` | `'produit'` ou `'max'` — §4.13 |
| `FEATURE_FLAGS` | tous False | `mean`, `variance`, `skewness`, `kurtosis`, `rms`, `mav`, `crest_factor`, `autocorr_lag1`, `zcr`, `dominant_freq`, `spectral_centroid`, `spectral_spread` |

### 6.5 Contrôle de stationnarité

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `STATIONNARITE_ENABLED` | True | False : aucun diagnostic |
| `STATIONNARITE_N_SONDES` | 8 | Nombre de fréquences sondes. Plage 4–20 |
| `STATIONNARITE_FRAC_ECHEC` | 0.3 | Seuil de l'indicateur (a). Plage 0,1–0,5 |
| `STATIONNARITE_CV_RMS_MAX` | 0.15 | Seuil de l'indicateur (b). Plage 0,05–0,40 ; à relever vers 0,20–0,25 si T_b < 0,2 s |
| `STATIONNARITE_KURT_TOL` | 0.5 | Tolérance sur \|kurtosis − 3\| pour « gaussien » |
| `STATIONNARITE_SKEW_TOL` | 0.3 | Tolérance sur \|asymétrie\| pour « gaussien » |

### 6.6 Quantile à la durée du signal

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `PROBABILITE_CIBLE` | 0.9 | Probabilité de non-dépassement du maximum d'**un bloc**, utilisée si `OPTION_CUNNANE = False`. Risque α de [2] Tableau 1 : 0,9 / 0,99 / 0,999 |
| `OPTION_CUNNANE` | True | p = (N − a)/(N + 1 − 2a) : plus grand quantile atteignable avec N blocs mesurés |
| `CUNNANE_A` | 0.4 | Constante de Cunnane (ν de la norme) |

### 6.7 SDF

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `SDF_ENABLED` | False | Active le calcul du dommage (rainflow et Bendat) |
| `SDF_B` | 8.0 | Pente de Basquin b. [2] : 4, 5 ou 8 pour les équipements électroniques, optroniques, mécaniques |
| `SDF_C` | 1.0 | Constante de Basquin (analyse relative) |

### 6.8 Projection longue durée

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `ENABLE_PROJECTION` | True | Active la projection |
| `DUREE_PROJECTION` | 36 000 000 | T_proj (s) = 10 000 h |
| `ALFA_PROJECTION` | 0.90 | Probabilité de **non-dépassement** projetée |
| `METHODE_PROJECTION` | `'puissance'` | `'puissance'` ou `'gev_domaines'` — §4.10 |
| `ALPHA_SRX_HIGH` | 0.99 | Risque élevé du SRX (comparaison à un choc) |
| `ALPHA_SRX_LOW` | dérivé | = 1 − `ALFA_PROJECTION` |
| `LOI_AJUSTEMENT` | `'kappa4'` | `'kappa4'` ou `'rayleigh_gen'` |

### 6.9 Contrat IID

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `IID_GATE_ENABLED` | True | Active le contrôle et son commentaire |
| `IID_RHO_MAX` | 0.2 | Seuil de \|ρ\| de Spearman |
| `IID_PVALUE_MIN` | 0.05 | Seuil de p-value du test des suites |
| `IID_FAIL_FRAC_MAX` | 0.05 | Fraction de fréquences en échec jusqu'à laquelle le verdict reste GO |
| `IID_NOGO_FRAC` | 0.30 | Fraction à partir de laquelle le verdict est NO-GO |
| `IID_MIN_N` | 20 | Nombre de blocs minimal pour tester |

### 6.10 Divers et exécution

| Paramètre | Défaut | Notes |
|-----------|--------|-------|
| `RANDOM_SEED` | 53 | Graine (K-Means, taux de hasard du contrat IID) |
| `MIN_POINTS_KAPPA4` | 40 | Nombre de blocs minimal pour ajuster une loi |
| `OUTPUT_FOLDER` | `"mbd_simple_output"` | Dossier racine des sorties |
| `USE_MULTIPROCESS` | True | Boucle des fréquences en parallèle |
| `N_WORKERS` | 10 | Nombre de processus ; `None` = cœurs physiques |

### 6.11 Paramètres experts

Regroupés dans la carte « PARAMÈTRES EXPERT » qui suit la SECTION 1.

| Paramètre | Défaut | Rôle |
|-----------|--------|------|
| `KAPPA4_ANALYTIC_XTOL` | 1.49e-8 | Tolérance de `fsolve` |
| `KAPPA4_L2_MIN_FOR_FIT` | 1e-10 | Sous ce L2, données jugées constantes : ajustement refusé |
| `KAPPA4_RESIDUAL_TOL` | 1e-6 | Résidu² maximal accepté sur (τ3, τ4) |
| `KAPPA4_WARM_START_INITIAL` | (0.1, 0.1) | Amorce (k, h) de la première tentative |
| `KAPPA4_RETRY_ENABLED`, `KAPPA4_RETRY_WARM_STARTS`, `KAPPA4_RETRY_MAXFEV` | True, 4 amorces, 2000 | Nouveaux essais si le solveur n'aboutit pas |
| `KAPPA4_H_MIN` | −1.0 | Borne basse de h ; `None` = aucun contrôle |
| `KAPPA4_DOMAIN_RETRY_STARTS` | 4 amorces | Amorces dans h ∈ [−1, 0) après rejet d'une racine hors domaine |
| `KAPPA4_DEBUG_ANALYTIC` | False | Exporte `Kappa4_Debug_*.csv` |
| `SDOF_AMORCAGE`, `SDOF_AMORCAGE_N_TAU` | True, 8.0 | Amorçage de l'oscillateur — §4.3 |
| `GEV_GUMBEL_K_TOL` | 0.01 | Seuil \|k\| de l'étiquette « gumbel » (diagnostic) |
| `KAPPA4_MEAN_VAR_GRID` | 4096 | Grille d'intégration des moments de la loi (dommage) |
| `PROBA_CLIP_EPS` | 1e-7 | Bornes de la probabilité de Cunnane |
| `MIN_ECH_PAR_BLOC` | 10 | Nombre minimal d'échantillons par bloc |

---

## 7. Fichiers de sortie

Tous dans le sous-dossier du calcul, suffixés par un tag (`v3p5pub_f5-400_Q10_Tb1p28_b8_T36Ms_a0p9_<date>`).

| Fichier | Contenu |
|---------|---------|
| `params_*.json` | Tous les paramètres du calcul, `Tb_effectif_s`, `n_blocs`, diagnostic de stationnarité |
| `Stationnarite_*.csv` | Une ligne de synthèse (verdict, indicateurs), puis une ligne par sonde |
| `SRE_*.csv` | SRC, SRE MBD, SRE analytique, SRX α faible et élevé, colonnes du contrat IID. Si K > 1 : `Synthese_classes`, `SRE_Kappa4_max_classe_*`, `Classes_exclues`, `Occurrence_exclue` |
| `SRE_Projection_*.csv` | SRE projeté, SRE et SRX analytiques projetés, M, SDF projetés (si SDF actif). `GEV_domaine` en méthode GEV. Si K > 1 : mêmes colonnes de synthèse |
| `IID_QualityGate_*.csv` | Par (f₀, classe) : ρ, p-value, nombre de blocs, statut. `Classe = −1` désigne la série de tous les blocs |
| `SDF_*.csv` | (si SDF actif) rainflow sur tout le signal, somme des dommages par bloc, Bendat, qualité de l'ajustement sur D_p |
| `SDF_Kappa4_Fit_*.csv` | (si SDF actif) paramètres de la loi ajustée sur les D_p, par (f₀, classe) |
| `Kappa4_Debug_*.csv` | (si `KAPPA4_DEBUG_ANALYTIC`) trace du solveur |
| `Rapport_*.html` | Synthèse spectrale : (1) SRC et SRE ; (2) comparatif SRE / SRX, mesure et projection ; (3) SDF ; (4) SRE mesuré et projeté |
| `Rapport_Details_*.html` | Diagnostic : (1) diagramme (τ3, τ4) ; (2) répartitions par classe, sélecteur de fréquence ; (3) comparatif SRE / SRX et MSDI ; (4) SDF projeté ; (5) contrat IID vs f₀ |

En tête des deux rapports, trois blocs repliables : **paramètres du calcul**, **contrat IID**, **guide de lecture des courbes**.

**Suffixes des colonnes CSV** : chaque grandeur porte les paramètres dont elle dépend, ce qui permet d'empiler des CSV de calculs différents.

- `SRC_Q10_Tb1p28s`
- `SRE_Kappa4_Q10_Tb1p28s_P0p9_Tmes599p999s`
- `SRX_alpha_low_Q10_Tb1p28s_aL0p1_Tmes599p999s`
- `SRE_Projection_Q10_Tb1p28s_P0p9_T36Ms_a0p9`
- `IID_rho_lag1_Q10_Tb1p28s`

**Statuts d'ajustement** (infobulle des courbes SRE, CSV de debug) : `ok`, `ok_retry` (obtenu sur une amorce de secours), `ok_h_borne` (loi du bord du domaine), `n_lt_min` (trop peu de blocs), `fit:<raison>` (échec).

---

## 8. Architecture du code (sections internes)

`mbd_simple-multi-process_v3_6.py` est un fichier unique autoporteur.

| Section | Rôle | Fonctions principales |
|---------|------|-----------------------|
| 1 | Configuration, puis carte des paramètres experts | — |
| 2 | Journal | `logging.basicConfig` |
| 3 | Import CSV | `importer_signal_csv` |
| 4 | Découpage et features | `_taille_bloc`, `extraire_caracteristiques` |
| 5 | Réponse 1-DDL | `reponse_sdof` |
| 6 | Kappa4 | `_calculer_pwm`, `calculer_lmoments`, `_g_functions`, `_g_deriv_k0`, `_fit_loc_scale`, `_tau3_tau4_from_kh_analytic`, `_kappa4_h_hors_domaine`, `_kappa4_au_dessus_glo`, `_kappa4_sur_borne_h`, `kappa4_from_lmoments`, `ajuster_kappa4`, `_msdi_queue_droite`, `kappa4_rmse_msdi`, `_kappa4_ppf_exact`, `_kappa4_cdf_exact`, `kappa4_ppf` |
| 6bis | Rayleigh généralisée et aiguillage | `ajuster_rayleigh_gen`, `rayleigh_gen_ppf`, `rayleigh_gen_rmse_msdi` ; `ajuster_loi`, `loi_ppf`, `loi_rmse_msdi` |
| 6ter | Projection GEV | `_gev_domaine`, `ajuster_gev_lmoments`, `calculer_projection_gev_domaines` |
| 7 | Branche analytique | `calculer_sre_analytique` |
| 8 | Rainflow | `_rainflow_damage`, `calculer_sdf_rainflow`, `calculer_sdf_per_bloc` |
| 9 | Projection | `calculer_projection_lmoments`, `calculer_projection_sdf_tcl`, `calculer_projection_dmg_kappa4`, `_kappa4_mean_var` |
| 9bis | Contrat IID | `quality_gate_iid`, `_iid_verdict`, `agreger_quality_gate`, `_iid_taux_hasard`, `_iid_bandes`, `commenter_quality_gate` |
| 9ter | Stationnarité | `detecter_non_stationnarite`, `exporter_csv_stationnarite` |
| 9quater | Synthèse des classes | `_loi_logcdf`, `quantile_produit_classes`, `_quantile_produit` |
| 10 | Traitement d'une fréquence | `traiter_f0`, processus de travail |
| 11 | Exports CSV | `build_run_meta`, `exporter_csv_*`, `_cols_synthese_classes`, `_bilan_classes_exclues` |
| 12 | Rapports HTML | `generer_html`, `generer_html_details`, `_ecrire_html_avec_cadre` |
| 13 | Programme principal | `main`, `main_mp` |

---

