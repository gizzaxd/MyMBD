# -*- coding: utf-8 -*-
"""
MBD Kappa4 — Script mono-fichier autoporteur (V3.5)
====================================================

Outil de calcul des spectres SRC / SRE & SRX / SDF d'un signal d'accélération mesuré,
avec projection longue durée. Implémente la méthode "MBD non corrélée" de la
norme NF X50-144-3 (2021) Annexe C, avec deux modes :
- un mode utilisant une loi paramétrique unique Kappa4 (Hosking 1994) ajustée par L-moments analytiques (Colin 2023, COFREND) ;
- un mode utilisant une loi paramétrique "Rayleigh généralisée" couvrant un domaine restreint (A. Clou & P. Lelan , DGA TT / CFM 2025) ;

Révision V3.5 (2026-06) :
  - Nouvelle option METHODE_PROJECTION='gev_domaines' : projection SRE par
    max-stabilité GEV avec h figé à 0 et discrimination des 3 domaines
    (Gumbel k≈0 / Fréchet k<0 / Weibull négative k>0), cf. [2] §4.1.
  - Diagramme τ3/τ4 : limite basse de validité corrigée, τ4=(5τ3²−1)/4
    ([2] eq. 15) — l'ancien tracé dupliquait la courbe h=−1 (GLO).
  - Un sous-dossier de résultats est créé par run (date_heure + fichier +
    paramètres) ; le nom du fichier analysé (≤35 car.) figure dans les HTML.
  - Convention de signe documentée pour reponse_sdof (sortie = −z, sans
    effet sur SRC/SRE/SDF).
  - MSDI corrigé : Mean Square Deviation Index de [1] §C.7 / CFM 2025 §3.1,
    MSDI(%) = (100/m)·Σ[(X_exp − X_loi)/X_exp]² sur les m points > moyenne
    (queue droite, positions Cunnane a=0.4). L'ancien calcul (écart SIGNÉ
    moyen ECDF−CDF, sans carré) ne correspondait pas à la norme.

Méthode d'identification Kappa4 : `direct_pwm_analytic`
    L-moments empiriques → fsolve(τ3_théorique(k,h)=τ3_empirique,
                                  τ4_théorique(k,h)=τ4_empirique)
    avec warm-start (k=0.1, h=0.1).

Calcul du dommage MBD selon NF X50-144-3 §C.10-C.11 :
  - SDF empirique = somme des dommages-bloc (rainflow par bloc),
    et non plus Σ max_bloc^b. La granularité élémentaire du dommage
    est le bloc Tb ; le dommage total d'une classe est la somme des
    D_bloc (Miner), pas le maximum.
  - Ajustement Kappa4 sur la distribution des D_bloc par classe
    (en parallèle de la branche SRE qui reste sur les maxima).
  - Projection longue durée du dommage par TCL/lognormale alimentée
    par les moments K4 ajustés sur les D_bloc (somme de variables
    i.i.d. → log-normale, pas F^M).
  - SDF activé par défaut.

Pipeline complet :
  1. Import CSV (auto-encodage)                         → SECTION 3
  2. Extraction features par bloc Tb                    → SECTION 4
  3. Classification K-Means des blocs d'excitation      → main()
  4. Réponse SDOF (FOH + lfilter, Smallwood récursif)   → SECTION 5
  5. Maxima par bloc + rainflow par bloc                → SECTIONS 4, 8
  6. Ajustement Kappa4 (L-moments analytiques)          → SECTION 6
  7. SRE/SDF analytique depuis DSP Welch                → SECTION 7
  8. Projection CDF longue durée                        → SECTION 9
  9. Boucle f0 multiprocess (shared_memory)             → SECTION 10
 10. Exports CSV + rapports HTML interactifs            → SECTIONS 11, 12

Usage:
    python mbd_simple-multi-process_v3_5.py

Dépendances : numpy, scipy, scikit-learn, pandas, tqdm, plotly, numba
              (psutil optionnel pour détection cœurs physiques)

----------------------------------------------------------------------
Références documentaires (cf. README.md pour pages détaillées)
----------------------------------------------------------------------
[1] NF X50-144-3 (2021) — Démonstration de la tenue aux environnements
    mécaniques, Partie 3 : Personnalisation. Annexe C (pp. 70-89 du PDF) :
    méthode MBD non corrélée. Fichier local : "[NF X50 144-3] 2022.pdf".

[2] B. Colin (KNDS / COFREND 2023) — Maintenance prévisionnelle des
    équipements critiques embarqués sur systèmes d'armes terrestres.
    e-Journal of NDT, doi:10.58286/28496. Pages 10-13 : formules Kappa4
    par L-moments analytiques (eq. 16-19, 20.1-20.3, 28-34).
    Fichier local : "MBD&KAPPA4_ME3E2_B_Colin.pdf".

[3] A. Clou & P. Lelan (DGA TT / CFM 2025) — Development of statistical
    methods for vibration analysis. 26ème Congrès Français de Mécanique,
    Metz. Critères SSI, limites Kappa4 sur signaux gaussiens.
    Fichier local : "2025-12-08_Article_CFM_2025_CLOU_LELAN.pdf".

[4] ASTM E1049-85 (2017) — Standard Practices for Cycle Counting in
    Fatigue Analysis. Algorithme rainflow strict (4-point Downing-Socie).

[5] AFNOR A03-406 — Méthodes de comptage des cycles pour l'analyse en
    fatigue. Équivalent rainflow français de [4].

[6] Hosking, J.R.M. (1994) — The four-parameter kappa distribution.
    IBM Journal of Research and Development, 38(3):251-258.

[7] Lalanne, C. (2009) — Mechanical Vibration and Shock Analysis,
    Vol. 4 Fatigue Damage, Wiley/ISTE. Approximation narrow-band
    gaussienne du SDF spectral (Bendat).

[8] Smallwood, D.O. (1981) — An improved recursive formula for
    calculating shock response spectra. Shock & Vibration Bulletin 51.
    Coefficients FOH récursifs utilisés en NF X50-144-3 §C.3 p. 76.

[9] Cunnane, C. (1978) — Unbiased plotting positions, A review.
    Journal of Hydrology 37, 205-222. Plotting position avec a=0.4
    préconisée par [1] §C.7.

[10] LALANNE C. (2002) — Mechanical Vibration and Shock, Volume 3: Random Vibration, Hermes Penton, 2002

----------------------------------------------------------------------
"""

import os
# --- Limitation BLAS (DOIT précéder l'import de numpy/scipy/sklearn) ----------
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('BLIS_NUM_THREADS',     '1')

import sys
import math
import time
import logging
import multiprocessing as mp
from datetime import datetime

import numpy as np
from scipy.signal import lfilter, lfilter_zi, welch
from scipy.integrate import simpson
from scipy.special import gamma as sp_gamma, beta as sp_beta, digamma as sp_digamma
from scipy.stats import kappa4 as scipy_kappa4, skew, kurtosis as sp_kurtosis
from scipy.stats import lognorm as scipy_lognorm
from scipy.stats import rankdata as sp_rankdata, norm as sp_norm
from tqdm import tqdm

# Imports lourds (pandas, sklearn, plotly) restent en lazy/local pour éviter
# de les recharger dans chaque worker multiprocess.

from numba import njit
# HAS_RAINFLOW conservé pour ne pas casser les call-sites historiques.
# Désormais : comptage rainflow par pile (stack) Numba — ASTM E1049 / AFNOR A03-406,
# convention AMPLITUDE (σ_a = range/2). Plus de dépendance au paquet 'rainflow'.
HAS_RAINFLOW = True

import plotly  # noqa: F401
HAS_PLOTLY = True


# =============================================================================
# SECTION 1 — CONFIGURATION (modifier ici avant de lancer)
# =============================================================================
#
# Tous les paramètres ci-dessous sont modifiables sans toucher au reste du
# code. Pour chaque paramètre :
#   - une PLAGE recommandée est indiquée quand elle existe ;
#   - une RÉFÉRENCE normative est citée quand le paramètre découle de [1]/[2]/[3].
#
# Voir README.md §6 pour un tableau récapitulatif et les conventions de
# suffixage des CSV de sortie.
# -----------------------------------------------------------------------------

# --- Fichier CSV d'entrée ----------------------------------------------------
# Format attendu : 2 colonnes (temps en s, accélération en m/s²).
# Encodage et séparateur décimal auto-détectés (utf-8/latin-1/cp1252, ',' ou '.').
CSV_FILEPATH    = r"C:\Users\aaaaaa.csv"

CSV_SKIP_ROWS   = 10        # Lignes d'en-tête à sauter (≥ 0)
CSV_DELIMITER   = ";"       # Délimiteur — typiquement ";" (FR) ou "," (US)

# --- Paramètres SDOF ---------------------------------------------------------
# Référence : [1] NF X50-144-3 §C.3 (p. 76) — coefficients FOH Smallwood [8].
#
# Q  = coefficient de surtension de l'oscillateur 1-DDL étalon. Q = 1/(2ξ).
#      Plage usuelle 5–50. Q=10 ⇔ ξ=5% — standard pour étalon mécanique
#      d'équipement embarqué.
# TB = durée de bloc T_b (s) pour la méthode MBD ([1] §C.5, [2] §4.1).
#      Plage usuelle 0.05–10 s.
#      Doit vérifier T_b ≫ 1/f₀_min (capter le mode bas) ET T_b ≪ T_mesure
#      (avoir au moins MIN_SAMPLES_PER_CLUSTER blocs par classe).
Q   = 10
TB  = 1.0

# --- Spectre de fréquences ---------------------------------------------------
# Bornes du SRE/SDF en fréquence propre f₀.
#   F0_MIN  : ≥ 1/T_b recommandé pour que le mode soit captable sur un bloc.
#   F0_MAX  : < fs/4 (anti-repliement). Idéalement < fs/10 pour précision FOH < 1%.
#   DELTA_F0: pas en fréquence (Hz). num_f0 = round((F0_MAX - F0_MIN)/DELTA_F0) + 1.
#             Ex. 1→pas de 1 Hz ; 0.5→pas de 0.5 Hz (2× plus de points).
F0_MIN   = 5
F0_MAX   = 400
DELTA_F0 = 1

# --- Classification K-Means des blocs d'excitation ---------------------------
# Référence : [1] §C.4 — Run-Test classifie le signal en N_c classes localement
# stationnaires. Ici implémenté par K-Means sur features par bloc (alternative
# pratique au Run-Test ; cf. [3] §4.1 SSI pour une variante plus rigoureuse).
#
# N_CLUSTERS              : K a priori. 1 = mono-classe (pas de partition).
#                            Plage 1–10 ; au-delà, statistiques par classe trop
#                            pauvres pour Kappa4 (n < 40).
# AUTO_SELECT_K           : si True, choisit K dans K_RANGE par maximisation
#                            du score Silhouette (sklearn).
# K_RANGE                 : plage testée pour K optimal (utilisée si
#                            AUTO_SELECT_K=True).
# MIN_SAMPLES_PER_CLUSTER : si une classe a moins de blocs que ce seuil, K
#                            est décrémenté automatiquement (jusqu'à K=1).
#                            Doit rester ≥ MIN_POINTS_KAPPA4 (40) sinon les
#                            fits Kappa4 par classe échouent.
N_CLUSTERS              = 1
AUTO_SELECT_K           = False
K_RANGE                 = range(3, 9)
MIN_SAMPLES_PER_CLUSTER = 40

# Features calculées par bloc et utilisées comme vecteurs d'entrée pour
# K-Means. Activer/désactiver via True/False. Chaque feature est centrée et
# réduite (StandardScaler) avant K-Means.
#   - mean, variance, skewness, kurtosis : moments statistiques classiques
#   - rms       : racine de la moyenne des carrés
#   - mav       : mean absolute value
#   - crest_factor : peak / rms
#   - autocorr_lag1 : autocorrélation à lag 1
#   - zcr       : zero-crossing rate
#   - dominant_freq, spectral_centroid, spectral_spread : indicateurs FFT
FEATURE_FLAGS = {
    'mean': False,   'variance': False,   'skewness': False,   'kurtosis': False,
    'rms': False,   'mav': False,        'crest_factor': False,
    'autocorr_lag1': False, 'zcr': False,
    'dominant_freq': False, 'spectral_centroid': False, 'spectral_spread': False,
}
# Ordre canonique pour la cohérence des colonnes — ne pas modifier.
ORDERED_FEATURE_KEYS = [
    'mean', 'variance', 'skewness', 'kurtosis', 'rms', 'mav',
    'crest_factor', 'autocorr_lag1', 'zcr', 'dominant_freq',
    'spectral_centroid', 'spectral_spread',
]

# --- Probabilité cible pour le PPF (quantile SRE sur base population signal d'entrée) ----------------------------
# Référence : [1] §C.7 (p. 80) — Calcul du quantile avec la probabilité cible déterminée via la formule de Cunnane avec ν=0.4 [9].
# [2] Tableau 1 : risque α = 0.1 / 0.01 / 0.001 selon criticité (faible/moy/forte).
# la p_eff via Cunnane sert à obtenir le quantile maximal calculable sur la population empirique N (durée signal) — il n'a pas de rôle quand on extrapole à T_proj via M.
# PROBABILITE_CIBLE : probabilité de NON-dépassement du maximum d'un bloc
#                     (quantile cible de la loi locale ; risque α = 1 − P).
#                     Plage (0,1) stricte. Si OPTION_CUNNANE=True, remplacée
#                     par p_eff = (N - a) / (N + 1 - 2a).
#                     Attention : quantile du maximum d'UN bloc (différent de
#                     la probabilité associée aux "M" maxima projetés par TVE).
# OPTION_CUNNANE    : True (recommandé par [1] pour obtenir le quantile maximal calculable avec la population empirique N) → p_eff dépend de N.
# CUNNANE_A         : constante 'a' (= ν dans la norme). 0.4 = standard.
PROBABILITE_CIBLE = 0.9      # pris si OPTION_CUNNANE=False ; probabilité de NON-dépassement du maximum d'un bloc => 0.9 <=> risque alpha=10% ; 0.99 <=> 1% ; 0.999 <=> 0.1%
OPTION_CUNNANE    = True    # utile pour obtenir le quantile maximal calculable avec la population empirique N  => pour projection DUREE_PROJECTION = durée du signal ou ENABLE_PROJECTION = False
CUNNANE_A         = 0.4

# --- SDF (Spectre de Dommage par Fatigue) ------------------------------------
# Référence : [1] §C.10-C.11 (Σ D_bloc), [2] eq. 1 (Basquin), [4]/[5] rainflow.
#
# Modèle : N · σ_a^b = C  (Basquin), Miner additif sur les cycles rainflow.
# Convention amplitude : σ_a = range/2 (cohérent ASTM E1049 / AFNOR A03-406).
#
# SDF_ENABLED : active le calcul SDF (rainflow + Bendat). Désactiver pour
#                gain ~30% sur très long signal si seul le SRE intéresse.
# SDF_B       : pente de Basquin b. Plage usuelle métaux 3–14 :
#                 - alu      : b ≈ 3-5
#                 - acier    : b ≈ 5-8
#                 - soudures : b ≈ 3-4
#                 - composites: b ≈ 8-14
#                b=8 : valeur par défaut "matériau dur" cf. [2].
# SDF_C       : constante de Basquin C. Fixée à 1 par défaut → analyse
#                relative (la valeur absolue de SDF dépend du matériau réel).
SDF_ENABLED = False
SDF_B       = 8.0
SDF_C       = 1.0

# --- SRE MBD projeté KAPPA 4 - Paramètres de projection CDF longue durée ---------------------------------------------
# Référence : [1] §C.8-C.9 (M = (T_v/T_b)·Occ(j), critère M > 100 pour TVE,
# M > 50 pour TCL), §C.10 (synthèse stochastique).
#
# ENABLE_PROJECTION : active la projection à T_v.
# DUREE_PROJECTION  : T_v en secondes. 36×10⁶ s = 10000h
#                      Plage typique : T_mesure × 10² à T_mesure × 10⁸.
# ALFA_PROJECTION   : probabilité de NON-dépassement projetée (la norme [1]
#                      raisonne en risque de dépassement α_risque = 1 − cette
#                      valeur). 0.9 = 90% de chances de ne pas être dépassé.
#                      Plage 0.5–0.999.
ENABLE_PROJECTION  = True
DUREE_PROJECTION   = 36_000_000 #  36_000_000
ALFA_PROJECTION    = 0.90  # probabilité de NON-dépassement projetée (= 1 − α_risque de [1]).

# --- Méthode de projection du SRE -------------------------------------------
# Référence : [2] Colin §4.1 — la variable globale Z_sup = max des M Z_max,i
# (eq. 35 : F_Zsup = F_Zmax^M) tend asymptotiquement (M grand), d'après le
# théorème de Fisher-Tippett / Gnedenko, vers l'une des 3 lois des valeurs
# extrêmes (EVD) suivant le domaine d'attraction de la loi locale :
#   - Gumbel  (EV1) si k* = 0 (queue fine),
#   - Fréchet (EV2) si k* < 0 (queue épaisse),
#   - Weibull négative (EV3) si k* > 0 (queue bornée).
#
# METHODE_PROJECTION :
#   - 'puissance'     : méthode historique. Élévation directe à la puissance M
#                        de la CDF de la loi sélectionnée (LOI_AJUSTEMENT) :
#                        SRE_proj = PPF(α^(1/M))  (cf. calculer_projection_lmoments).
#   - 'gev_domaines'  : alternative [2] §4.1. La loi locale est ré-ajustée en
#                        fixant h = 0 (famille GEV, sous-famille de Kappa4) sur
#                        les Z_max par classe ; k* est déduit de τ3 seul (1 éq.
#                        à 1 inconnue, plus stable que le système 2×2 en (k,h)).
#                        La projection utilise alors la max-stabilité EXACTE de
#                        la GEV, discriminée selon les 3 domaines ci-dessus :
#                          k≈0 : ξ_M = ξ + α·ln(M),          α_M = α      (Gumbel)
#                          k≠0 : ξ_M = ξ + (α/k)(1 − M^−k), α_M = α·M^−k (Fréchet k<0 /
#                                                                         Weibull nég. k>0)
#                        puis SRE_proj = quantile GEV(ξ_M, α_M, k) à α.
#                        Forme close — évite le calcul de α^(1/M) → 1 (clip
#                        numérique) pour M très grand. Le domaine retenu par
#                        (f₀, classe) est exporté (CSV projection + HTML).
METHODE_PROJECTION = 'puissance'   # 'puissance' (défaut) | 'gev_domaines'

# --- SRE / SRX analytiques depuis DSP ----------------------------------------
# Branche de comparaison alternative au MBD-Kappa4, calculée depuis la DSP Welch
# du signal d'entrée et l'intégration spectrale ∫ Pxx(f)·|Fd(f,f₀,Q)|² df.
#
# Référence : [4] PR NORMDEF 0101 (DGA 2009) et [10] LALANNE C. (2002) — Mechanical Vibration and Shock, Volume 3: Random Vibration
#               §5.4.2 — définition du SRE : pic moyen sur T de la réponse SDOF
#               §5.4.3 — SRX (Spectre de Réponse à risque de Dépassement)
#                       formule [5.2] : R_X = (2π·f₀)²·z_eff·√(-2·ln(1-(1-α)^(1/(n₀⁺·T))))
#                       formule [5.3] : SRX/SRE en fonction de α et n₀⁺·T
#                       fig. 5.2/5.3 : enveloppes typiques SRX(α) vs SRE vs SRC
#             [5] B. Colin, MI0460 (2008) — origine du modèle SRX non-asymptotique
#                                            (vs approches asymptotiques Gumbel/Poisson).
#
# Deux niveaux α sont conservés simultanément, qui correspondent aux deux usages
# documentés en NORMDEF §5.4.3 :
#   - ALPHA_SRX_LOW   : risque faible (typ. 1-10%) → DIMENSIONNEMENT enveloppe
#                       haute, pic que la réponse a α de chances de dépasser sur T.
#                       DÉRIVÉ : ALPHA_SRX_LOW = 1 - ALFA_PROJECTION (cf. plus bas).
#                       Convention opposée à ALFA_PROJECTION (probabilité de
#                       non-dépassement côté NF X50-144-3 §C.9) — l'alignement
#                       automatique garantit que le SRE MBD projeté et le SRX α_low
#                       projeté représentent le MÊME risque de dépassement.
#   - ALPHA_SRX_HIGH  : risque élevé (typ. 99%) → comparaison vs SRC d'un CHOC.
#                       Si SRX(99%) > SRC, la vibration aléatoire est plus sévère
#                       que le choc avec ≥99% de probabilité (NORMDEF fig. 5.3).
#                       Réglable indépendamment (pas lié à la projection).
#
# Plage stricte (0,1). Hypothèse narrow-band gaussienne (n₀⁺ ≈ f₀).
ALPHA_SRX_HIGH = 0.99 # valeur par défaut 99% => 0.99

# ALPHA_SRX_LOW est DÉRIVÉ de ALFA_PROJECTION ("SRE MBD projeté KAPPA 4 - Paramètres de projection CDF longue durée"). Garantit l'homogénéité du risque entre SRE MBD projeté
# et SRX α_low projeté sur la même durée T_proj.
ALPHA_SRX_LOW      = 1.0 - ALFA_PROJECTION    # "ALFA_PROJECTION" -> paramètre pour projection long terme "SRE MBD projeté KAPPA 4" 
# ---
# ---
# --- -------------------------------------------------------------------------
# --- Loi d'ajustement statistique --------------------------------------------
# Référence : [2] Hosking-Wallis (Kappa4) ; CFM 2025 Clou/Lelan §4.2 (Kundu &
# Raqab) pour la loi de Rayleigh généralisée F(x;α,λ)=(1−e^(−(λx)²))^α.
#
# LOI_AJUSTEMENT : loi a priori utilisée pour inférer le quantile SRE/SDF :
#   - 'kappa4'       : loi de Hosking à 4 paramètres.
#   - 'rayleigh_gen' : loi de Rayleigh généralisée à 2 paramètres (α forme,
#                       λ échelle). PPF analytique exacte ; l'article CFM 2025
#                       montre une meilleure stabilité inter-DDL que Kappa4
#                       pour les signaux gaussiens/non-gaussiens.
LOI_AJUSTEMENT      = 'kappa4'

# --- Indépendance statistique des blocs par f₀ ("Quality Gate IID") ------------
# Brique de validation "post-traitement" (Quality Gate). Vérifie l'hypothèse
# IID des blocs temporels — requise par l'inférence Kappa-4 via L-moments
# (NF X50-144-3 Annexe C ; [2] Colin 2023) — AVANT de faire confiance au
# SRE/SDF. Pour chaque colonne f₀,
#   1) autocorrélation lag-1 de Spearman calculée sur les RANGS (robuste aux
#      extrêmes de la Kappa-4) ;
#   2) test des suites de Wald-Wolfowitz (binarisation vs médiane, p-value
#      par approximation normale).
# Si TB est trop court pour englober la traîne de la réponse SDOF (typiquement
# basses fréquences / Q élevé), les blocs successifs sont corrélés et les
# L-moments biaisés : ce module l'objective et déclenche le feedback métier.
#
# IID_GATE_ENABLED   : active la vérification. False ⇒ comportement identique
#                       à l'historique (aucune colonne / section ajoutée).
# IID_RHO_MAX        : seuil |ρ| Spearman lag-1 acceptable (cahier : 0.2).
# IID_PVALUE_MIN     : seuil p-value test des suites (cahier : 0.05).
# IID_FAIL_FRAC_MAX  : fraction max de f₀ hors-tolérance pour rester 🟢 GO
#                       (cahier : 5% du spectre).
# IID_NOGO_FRAC      : au-delà de cette fraction de f₀ en échec ⇒ 🔴 NO-GO
#                       (dépendance généralisée). Entre les deux ⇒ 🟡 WARNING
#                       (bande isolée, souvent les basses fréquences).
# IID_MIN_N          : taille d'échantillon minimale pour qu'un test soit
#                       jugé fiable (sinon NaN, colonne ignorée du verdict).
IID_GATE_ENABLED  = True
IID_RHO_MAX       = 0.2
IID_PVALUE_MIN    = 0.05
IID_FAIL_FRAC_MAX = 0.05
IID_NOGO_FRAC     = 0.30
IID_MIN_N         = 20

# --- Divers ------------------------------------------------------------------
# RANDOM_SEED       : graine RNG (KMeans, etc.). Reproductibilité.
# MIN_POINTS_KAPPA4 : seuil minimum d'échantillons pour tenter un fit Kappa4.
#                      < 40 → fit refusé (instabilité L-moments, cf. [3] Fig. 10).
#                      [1] §C.9 demande au moins n=50 pour LAR convergent.
# OUTPUT_FOLDER     : dossier racine de sortie (créé si absent). À chaque
#                      calcul, un SOUS-DOSSIER dédié y est créé :
#                      <AAAAMMJJ_HHMMSS>_<fichier ≤35 car.>_<fmin-fmax_Q_Tb_b_Tproj_alfa>
RANDOM_SEED         = 53
MIN_POINTS_KAPPA4   = 40
OUTPUT_FOLDER       = "mbd_simple_output"

# --- Exécution parallèle -----------------------------------------------------
# USE_MULTIPROCESS : True = boucle f₀ parallélisée via mp.Pool + shared_memory.
# N_WORKERS        : nombre de workers. None → auto (psutil cœurs physiques).
#                     Sur Windows, ne pas dépasser le nombre de cœurs physiques
#                     (l'hyperthreading dégrade le rainflow Numba).
USE_MULTIPROCESS = True
N_WORKERS        = 10

# =============================================================================
# !!!! - PARAMÈTRES EXPERT — MODIFIER AVEC PRÉCAUTION - !!!!
# =============================================================================
# Ces paramètres règlent finement le comportement numérique du programme. Les
# valeurs par défaut couvrent la quasi-totalité des cas pratiques. Chaque
# constante est documentée : rôle, plage de confiance, et fonction(s) où elle
# intervient. Les epsilons anti-division-par-zéro (1e-9/1e-10 inline dans le
# code) restent non exposés : leur modification casserait la stabilité.
# -----------------------------------------------------------------------------

# --- Fit Kappa4 (L-moments + fsolve) -----------------------------------------
# KAPPA4_ANALYTIC_XTOL       : tolérance fsolve, plage 1e-12 à 1e-4.
#                               1.49e-8 = défaut SciPy (~ √eps machine).
#                               Plus serré = plus précis mais plus lent.
#                               Intervient : ajuster_kappa4_pwm_analytic (fsolve xtol).
# KAPPA4_L2_MIN_FOR_FIT      : seuil L2 dégénéré, plage 1e-14 à 1e-6.
#                               Si |L2| < seuil → fit refusé (données quasi-constantes).
#                               Intervient : ajuster_kappa4_pwm_analytic (garde-fou L2).
# KAPPA4_RESIDUAL_TOL        : seuil acceptation résidu² fsolve, plage 1e-9 à 1e-3.
#                               Solution rejetée si τ3/τ4 calculé s'éloigne trop des empiriques.
#                               Intervient : ajuster_kappa4_pwm_analytic (post-fsolve).
# KAPPA4_WARM_START_INITIAL  : warm-start 1ʳᵉ tentative, k0, h0 ∈ [-1, 1].
#                               (0.1, 0.1) = compromis historique ; cas pathologiques
#                               relevés par le retry si KAPPA4_RETRY_ENABLED=True.
#                               Intervient : ajuster_kappa4_pwm_analytic (x0 initial).
# KAPPA4_DEBUG_ANALYTIC      : True/False. Si True, exporte CSV Kappa4_Debug_* avec
#                               L-moments, warm-start, résidus fsolve, exit_reason
#                               par (f₀, classe). Utile pour investiguer les échecs.
#                               Coût négligeable hors I/O finale.
KAPPA4_ANALYTIC_XTOL        = 1.49e-8
KAPPA4_L2_MIN_FOR_FIT       = 1e-10
KAPPA4_RESIDUAL_TOL         = 1e-6
KAPPA4_WARM_START_INITIAL   = (0.1, 0.1)
KAPPA4_DEBUG_ANALYTIC       = False

# --- Retry fsolve (récupération des cas ier≠1 résolubles) --------------------
# Cas typique récupéré : f₀ où la 1ʳᵉ tentative renvoie ier=5 (maxfev atteint)
# alors que (τ3, τ4) est parfaitement dans le domaine de Kappa4. On retente sur
# une grille de warm-starts couvrant les 4 quadrants (k, h) avec maxfev étendu.
# KAPPA4_RETRY_ENABLED       : True/False. False = les ier≠1
#                               redeviennent des trous dans le spectre SRE
#                               Intervient : ajuster_kappa4_pwm_analytic (boucle attempts).
# KAPPA4_RETRY_WARM_STARTS   : liste de (k, h). 4 max recommandé (coût fsolve cumulé).
#                               Choix par défaut couvre les 4 quadrants.
#                               Intervient : idem.
# KAPPA4_RETRY_MAXFEV        : budget itérations fsolve par retry, plage 500 à 20000.
#                               Défaut fsolve = 200·(N+1) ≈ 600 pour N=2 → souvent insuffisant.
#                               Intervient : idem (paramètre maxfev).
KAPPA4_RETRY_ENABLED        = True
KAPPA4_RETRY_WARM_STARTS    = [(-0.2, -0.2), (0.5, 0.0), (0.0, 0.5), (-0.5, 0.5)]
KAPPA4_RETRY_MAXFEV         = 2000

# --- Projection GEV 3 domaines (METHODE_PROJECTION='gev_domaines') -----------
# GEV_GUMBEL_K_TOL           : tolérance |k*| sous laquelle le domaine est
#                               ÉTIQUETÉ 'gumbel' (EV1), plage 1e-6 à 0.1.
#                               Un k* estimé n'est jamais exactement nul : ce
#                               seuil sert au diagnostic (CSV / HTML). La
#                               formule de projection reste continue en k
#                               (limite Gumbel prise pour |k| < 1e-6) ; le
#                               seuil n'altère donc pas le résultat numérique.
#                               Intervient : _gev_domaine.
GEV_GUMBEL_K_TOL            = 0.01

# --- Projection longue durée (intégration moments Kappa4) --------------------
# KAPPA4_MEAN_VAR_GRID       : résolution numérique de la PPF pour estimer μ, σ²
#                               d'une Kappa4 ajustée, plage 256 à 65536.
#                               Sert dans la branche projection lognormale du dommage.
#                               Plus grand = plus précis mais O(N) sur scipy.stats.kappa4.
#                               Intervient : _kappa4_mean_var (TCL via moments K4).
KAPPA4_MEAN_VAR_GRID        = 4096

# --- Clip probabilités (Cunnane / projection TVE) ----------------------------
# Bornes pour éviter les PPF infinis aux extrémités (0 et 1).
# PROBA_CLIP_EPS             : plage 1e-12 à 1e-3. Bornes (eps, 1-eps) du quantile
#                               Cunnane à la durée du signal.
#                               Intervient : traiter_classes_kappa4 (clip prob_eff).
# PROBA_CLIP_EPS_PROJ        : plage 1e-15 à 1e-6. Bornes du quantile α projeté TVE.
#                               Séparé car M très grand → α^(1/M) très proche de 1,
#                               nécessite une borne plus fine.
#                               Intervient : calculer_projection_lmoments.
PROBA_CLIP_EPS              = 1e-7
PROBA_CLIP_EPS_PROJ         = 1e-12

# --- Extraction des blocs ----------------------------------------------------
# MIN_ECH_PAR_BLOC           : nombre minimum d'échantillons par bloc Tb,
#                               plage 5 à 1000. Sécurité contre les blocs trop courts
#                               qui produiraient des statistiques aberrantes (max,
#                               rainflow). 10 = défaut historique.
#                               Intervient : extraire_caracteristiques (argument min_ech).
MIN_ECH_PAR_BLOC            = 10

# Compteur cumulé de temps d'exécution Kappa4 (instrumentation, non réglable).
KAPPA4_TIMINGS = {
    'analytic': 0.0,
    'analytic_n': 0,
}
# =============================================================================


def _auto_n_workers():
    """Sélectionne N_WORKERS par défaut quand N_WORKERS=None.
    Préfère les cœurs PHYSIQUES (psutil), fallback os.cpu_count()-1."""
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        if phys and phys >= 1:
            return phys
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) - 1)


# --- Méthode d'ajustement Kappa4 ---------------------------------------------
# Référence : [2] §3-4, équations 16-19, 20.1-20.3, 28-34.
#
# Seule méthode supportée en V3 : `direct_pwm_analytic`.
#   Calcule (τ3, τ4) théoriques par les fonctions g_r de Hosking [6],
#   puis résout le système 2×2 :
#       τ3_théorique(k, h) = τ3_empirique
#       τ4_théorique(k, h) = τ4_empirique
#   par scipy.optimize.fsolve, démarré sur warm-start KAPPA4_WARM_START_INITIAL.
# Avantages vs polynômes Mielke (Tableaux 2.1-2.2 de [2]) :
#   - pas de table polynomiale à maintenir,
#   - tolérance contrôlable (KAPPA4_ANALYTIC_XTOL — cf. carte EXPERT),
#   - vérification a posteriori du résidu (rejet si > KAPPA4_RESIDUAL_TOL).
# Tous les seuils numériques sont définis dans la carte EXPERT ci-dessus.
KAPPA4_METHOD = 'direct_pwm_analytic'

# =============================================================================
# SECTION 2 — LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mbd_simple_v3")

# =============================================================================
# SECTION 3 — IMPORT DU SIGNAL CSV
# =============================================================================

def importer_signal_csv(filepath, skip_rows=10, delimiter=';'):
    """Lit un fichier CSV à 2 colonnes (temps en s, signal d'accélération en m/s²).

    Auto-détection :
        - encodage parmi utf-8, latin-1, windows-1252, iso-8859-1, cp1252
        - séparateur décimal ',' ou '.'

    Robustesse : les lignes contenant des NaN après conversion numérique sont
    retirées silencieusement (mask = ~np.isnan(t) & ~np.isnan(signal)).

    Paramètres
    ----------
    filepath  : str   chemin absolu vers le CSV.
    skip_rows : int   nombre de lignes d'en-tête à sauter (≥ 0).
    delimiter : str   typiquement ';' (FR) ou ',' (US).

    Retour
    ------
    (t, signal) : (np.ndarray, np.ndarray)  — temps et accélération nettoyés.

    Lève RuntimeError si le CSV ne peut être lu, ValueError si aucune donnée
    valide après nettoyage.
    """
    import pandas as pd
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
    df = None
    for enc in encodings:
        for dec in [',', '.']:
            try:
                df = pd.read_csv(filepath, sep=delimiter, skiprows=skip_rows,
                                 encoding=enc, header=None, decimal=dec)
                if df.shape[1] >= 2:
                    break
            except Exception:
                continue
        if df is not None and df.shape[1] >= 2:
            break

    if df is None or df.shape[1] < 2:
        raise RuntimeError(f"Impossible de lire le fichier CSV : {filepath}")

    t      = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    signal = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    mask   = ~(np.isnan(t) | np.isnan(signal))
    t, signal = t[mask], signal[mask]

    if len(t) == 0:
        raise ValueError("Aucune donnée valide dans le fichier CSV.")

    logger.info("Signal importé : %d points, durée=%.2f s", len(t), t[-1] - t[0])
    return t, signal

# =============================================================================
# SECTION 4 — EXTRACTION DES FEATURES ET MAXIMA PAR BLOCS
# =============================================================================

def _trouver_diviseurs(n):
    """Diviseurs de n (utilisé pour ajuster la taille de bloc à un diviseur exact
    de la longueur du signal — évite les blocs résiduels partiels)."""
    divs = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def extraire_caracteristiques(signal, fs, Tb_initial=0.05, feature_flags=None, min_ech=None):
    """Découpe le signal en blocs de durée ≈ T_b et calcule des features par bloc.

    Référence : [1] NF X50-144-3 §C.5 — n-échantillons par bloc T_b.
    La taille de bloc effective est ajustée au diviseur de N_total le plus
    proche de round(T_b·fs), pour éviter les blocs résiduels partiels.

    Features disponibles (activables via `feature_flags` ou FEATURE_FLAGS global) :
        Statistiques temporelles : mean, variance, skewness, kurtosis, rms, mav
        Forme :                    crest_factor (peak/rms)
        Corrélation :              autocorr_lag1, zcr (zero-crossing rate)
        FFT :                      dominant_freq, spectral_centroid, spectral_spread

    Paramètres
    ----------
    signal        : array — signal d'excitation (m/s²).
    fs            : float — fréquence d'échantillonnage (Hz).
    Tb_initial    : float — durée de bloc cible T_b (s). Plage typique 0.05–10.
    feature_flags : dict  — {nom_feature: bool}. Si None, utilise FEATURE_FLAGS.
    min_ech       : int   — nombre minimum d'échantillons par bloc (sécurité,
                            évite les blocs trop courts pour des stats robustes).

    Retour
    ------
    (features_array, maxima_array, noms_features, n_blocs, taille_bloc)
        features_array : (n_blocs, n_features) — vecteurs pour KMeans.
        maxima_array   : (n_blocs,)            — Z_ext = max(|bloc|) par bloc.
        noms_features  : list[str]             — noms des features actives.
        n_blocs        : int
        taille_bloc    : int — nombre de points par bloc (effectif).
    """
    if feature_flags is None:
        feature_flags = FEATURE_FLAGS
    if min_ech is None:
        # Carte EXPERT — défaut programmatique unique.
        min_ech = MIN_ECH_PAR_BLOC

    signal = np.asarray(signal, dtype=float)
    if not np.all(np.isfinite(signal)):
        mv = np.nanmean(signal)
        signal = np.nan_to_num(signal, nan=mv if np.isfinite(mv) else 0.0)

    n = len(signal)
    cible = max(round(Tb_initial * fs), min_ech)
    diviseurs_valides = [d for d in _trouver_diviseurs(n) if d >= min_ech]
    taille = (min(diviseurs_valides, key=lambda d: abs(d - cible))
              if diviseurs_valides else n)
    n_blocs = n // taille

    used = [k for k in ORDERED_FEATURE_KEYS if feature_flags.get(k, False)]

    if not used:
        blocks = signal[:n_blocs * taille].reshape(n_blocs, taille)
        return np.array([]), np.max(np.abs(blocks), axis=1), [], n_blocs, taille

    features_all, maxima = [], []
    for i in range(n_blocs):
        bloc = signal[i * taille: (i + 1) * taille]
        N    = len(bloc)
        maxima.append(float(np.max(np.abs(bloc))))
        bf = {}

        mean_b = np.mean(bloc)
        var_b  = np.var(bloc)
        rms_b  = math.sqrt(float(np.mean(bloc ** 2)))
        peak_b = float(np.max(np.abs(bloc)))

        if 'mean'         in used: bf['mean']         = mean_b
        if 'variance'     in used: bf['variance']      = var_b
        if 'rms'          in used: bf['rms']           = rms_b
        if 'mav'          in used: bf['mav']           = float(np.mean(np.abs(bloc)))
        if 'crest_factor' in used: bf['crest_factor']  = peak_b / rms_b if rms_b > 1e-10 else 0.0

        if var_b > 1e-12:
            if 'skewness' in used:
                bf['skewness'] = float(skew(bloc, nan_policy='omit'))
            if 'kurtosis' in used:
                kv = float(sp_kurtosis(bloc, fisher=False, nan_policy='omit'))
                bf['kurtosis'] = kv if np.isfinite(kv) and kv >= 0 else 3.0
        else:
            if 'skewness' in used: bf['skewness'] = 0.0
            if 'kurtosis' in used: bf['kurtosis'] = 3.0

        if N > 1:
            if 'autocorr_lag1' in used:
                c = bloc - mean_b
                vb = np.sum(c ** 2)
                bf['autocorr_lag1'] = float(np.sum(c[:-1] * c[1:]) / vb) if vb > 1e-9 else 0.0
            if 'zcr' in used:
                bf['zcr'] = len(np.where(np.diff(np.signbit(bloc)))[0]) / (N - 1)
        else:
            if 'autocorr_lag1' in used: bf['autocorr_lag1'] = 0.0
            if 'zcr'           in used: bf['zcr']           = 0.0

        fft_needed = any(f in used for f in ('dominant_freq', 'spectral_centroid', 'spectral_spread'))
        if N > 1 and fft_needed:
            fft_v = np.fft.rfft(bloc)
            fft_f = np.fft.rfftfreq(N, d=1.0 / fs)
            pwr   = np.abs(fft_v) ** 2
            tp    = float(np.sum(pwr))
            if tp > 1e-10:
                if 'dominant_freq'    in used:
                    bf['dominant_freq'] = float(fft_f[np.argmax(pwr[1:]) + 1])
                sc = float(np.sum(fft_f * pwr) / tp)
                if 'spectral_centroid' in used: bf['spectral_centroid'] = sc
                if 'spectral_spread'   in used:
                    bf['spectral_spread'] = float(np.sqrt(np.sum(((fft_f - sc) ** 2) * pwr) / tp))
            else:
                for k in ('dominant_freq', 'spectral_centroid', 'spectral_spread'):
                    if k in used: bf[k] = 0.0

        features_all.append([bf.get(k, np.nan) for k in used])

    return np.array(features_all), np.array(maxima), used, n_blocs, taille

# =============================================================================
# SECTION 5 — RÉPONSE SDOF PAR FOH (First-Order Hold + lfilter)
# =============================================================================

def reponse_sdof(excitation, f0, Q, fs):
    """Réponse en déplacement relatif z(t) d'un oscillateur 1-DDL par FOH.

    Référence : [1] NF X50-144-3 §C.3 (p. 76) — coefficients FOH récursifs
                 Smallwood [8].

    Le système 1-DDL (m, k, c) sous excitation de base ẍ(t) vérifie :
        z̈ + 2ξω₀ż + ω₀²z = -ẍ(t)
    avec ω₀ = 2πf₀ et ξ = 1/(2Q).

    L'hypothèse FOH (First-Order Hold = interpolation linéaire de l'excitation
    entre échantillons) conduit à un filtre IIR ordre 2 EXACT à l'échantillonnage,
    contrairement au ZOH qui sous-estime la réponse au-dessus de fs/10.

    Coefficients (a1, a2, b0, b1, b2) issus de Smallwood 1981, transcrits ici en
    formules directement utilisables par scipy.signal.lfilter. L'état initial
    `lfilter_zi * y[0]` annule le transitoire au démarrage du filtre.

    Paramètres
    ----------
    excitation : array — accélération d'excitation ẍ(t) (m/s²).
    f0  : float — fréquence propre (Hz). Doit vérifier f0 < fs/4 (anti-repliement)
                  et idéalement f0 < fs/10 (précision FOH < 1%).
    Q   : float — coefficient de surtension. Plage usuelle 5–50 ; Q ≥ 0.5 requis
                  (sinon ξ ≥ 1, système sur-amorti, retour zéros).
    fs  : float — fréquence d'échantillonnage (Hz).

    Retour
    ------
    z : np.ndarray — déplacement relatif (m), même taille que excitation.
        ⚠ z(t) est la grandeur à utiliser pour le rainflow SDF (cf. [1] §C.2),
        PAS la pseudo-accélération (2πf₀)²·z (qui sert au calcul du SRE final).

    Note convention de signe (vérifiée numériquement) : le filtre renvoie la
    réponse à +ẍ(t), soit l'OPPOSÉ du z de l'équation ci-dessus (réponse
    statique +a/ω₀² au lieu de −a/ω₀²). Sans incidence sur les résultats :
    SRC/SRE (max de |·|) et rainflow (comptage de cycles symétrique) sont
    pairs en z. Convention identique à celle des SRS usuels (Smallwood).
    """
    dt     = 1.0 / fs
    omega0 = 2.0 * math.pi * f0
    xi     = 1.0 / (2.0 * max(Q, 0.50001))

    if xi >= 1.0:
        return np.zeros_like(excitation)

    omega_d = omega0 * math.sqrt(1.0 - xi ** 2)
    if omega_d < 1e-9:
        return np.zeros_like(excitation)

    e   = math.exp(-xi * omega0 * dt)
    c   = math.cos(omega_d * dt)
    s   = math.sin(omega_d * dt)
    e2  = math.exp(-2.0 * xi * omega0 * dt)
    od  = omega0 / omega_d
    t2  = 2.0 * xi ** 2 - 1.0
    inv = 1.0 / (omega0 ** 3 * dt) if abs(omega0 ** 3 * dt) > 1e-12 else 0.0

    a1 = -2.0 * e * c
    a2 =  e2

    b0 = inv * (2.0 * xi * (e * c - 1.0) + e * od * t2 * s + omega0 * dt)
    b1 = inv * (-2.0 * omega0 * dt * e * c - 2.0 * od * t2 * e * s + 2.0 * xi * (1.0 - e2))
    b2 = inv * ((2.0 * xi + omega0 * dt) * e2 + e * (od * t2 * s - 2.0 * xi * c))

    b_c = [b0, b1, b2]
    a_c = [1.0, a1, a2]
    zi  = lfilter_zi(b_c, a_c) * excitation[0]
    y, _ = lfilter(b_c, a_c, excitation, zi=zi)
    return y

# =============================================================================
# SECTION 6 — KAPPA4 : direct_pwm_analytic (L-moments + fsolve)
# =============================================================================

def _calculer_pwm(data, nmom=4):
    """Probability Weighted Moments (PWM) sur données triées.

    Référence : [2] eq. 25 (formule de Greenwood pour PWM non-biaisés à partir
    d'un échantillon trié). Utilisés ensuite pour calculer les L-moments
    (eq. 17-19 de [2]).

    nmom = 4 suffit pour identifier les 4 paramètres de Kappa4.
    """
    xs = np.sort(data)
    n  = len(xs)
    b  = np.zeros(nmom)
    iv = np.arange(1, n + 1, dtype=np.float64)
    b[0] = np.mean(xs)
    for r in range(1, nmom):
        num, den = np.ones(n, dtype=np.float64), 1.0
        for j in range(r):
            num *= (iv - j - 1)
            den *= (n  - j - 1)
        b[r] = np.sum(num * xs) / (n * den)
    return b


def calculer_lmoments(data):
    """L-moments et ratios (l1, l2, τ3, τ4) depuis un échantillon.

    Référence : [2] eq. 17-19 — relation linéaire entre PWM et L-moments :
        l1 = b0
        l2 = 2·b1 - b0
        l3 = 6·b2 - 6·b1 + b0
        l4 = 20·b3 - 30·b2 + 12·b1 - b0
    avec τ3 = l3/l2 (L-skewness), τ4 = l4/l2 (L-kurtosis).
    """
    b  = _calculer_pwm(data)
    l1 = b[0]
    l2 = 2*b[1] - b[0]
    l3 = 6*b[2] - 6*b[1] + b[0]
    l4 = 20*b[3] - 30*b[2] + 12*b[1] - b[0]
    return l1, l2, l3/l2, l4/l2


def _g_functions(k, h, rmax=4):
    """Fonctions g_r de Hosking pour la loi Kappa4.

    Référence : [2] eq. 20.1–20.3, [6] Hosking 1994.

    Trois branches selon le signe de h :
        h > 0  : g_r = r·h^{-(k+1)} · B(r/h, k+1)        (eq 20.1)
        h = 0  : g_r = r^{-k} · Γ(1+k)                    (eq 20.2, limite)
        h < 0  : g_r = r·(-h)^{-(k+1)} · B(-r/h - k, k+1) (eq 20.3)

    Retourne None si une fonction Beta ou Gamma diverge.
    """
    try:
        g = []
        for r in range(1, rmax + 1):
            if h > 0:
                gr = r * h**(-(k+1)) * sp_beta(r/h, k+1)
            elif abs(h) < 1e-12:
                gr = r**(-k) * sp_gamma(1+k)
            else:
                gr = r * (-h)**(-(k+1)) * sp_beta(-r/h - k, k+1)
            g.append(gr)
        return g
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


# Largeur de la bande |k| < ε où l'on bascule sur le développement limité en
# k→0 : pour |k| < ε, g_r ≈ 1 ⇒ (g₁−g₂) → 0 (annulation catastrophique en
# double précision) et les ratios τ₃/τ₄ ainsi que _fit_loc_scale deviennent
# une forme 0/0. La logistique (k=0, h=−1) et toute la ligne k=0 (Gumbel,
# exponentielle…) tombent dans ce cas. ε=1e-6 : au-delà, la différence de g_r
# garde >10 chiffres significatifs ; en-deçà, on utilise la limite analytique
# (exacte à O(k), donc continue pour fsolve).
_KAPPA4_K_EPS = 1e-6


def _g_deriv_k0(h, rmax=4):
    """Coefficients γ_r ≡ ∂g_r/∂k évalués en k=0 (où g_r(0)=1 ∀r).

    Référence : développement limité des g_r de Hosking (cf. _g_functions),
    g_r(k) = 1 + k·γ_r + O(k²). Aux ratios (τ₃, τ₄) les constantes en r
    s'éliminent ; pour (ξ, α) elles comptent ⇒ on renvoie γ_r COMPLET.

        h > 0 : γ_r = −ln(h)   + ψ(1) − ψ(r/h + 1)
        h = 0 : γ_r = −ln(r)   + ψ(1)
        h < 0 : γ_r = −ln(−h)  + ψ(1) − ψ(−r/h)

    (ψ = digamma ; ψ(1) = −γ_Euler). Permet le passage à la limite k→0 :
        λ₂ = α·(g₁−g₂)/k → α·(γ₁−γ₂)
        λ₁ = ξ + α·(1−g₁)/k → ξ − α·γ₁
    Renvoie None si une valeur n'est pas finie (pôle de ψ).
    """
    try:
        psi1 = sp_digamma(1.0)
        gam = []
        for r in range(1, rmax + 1):
            rr = float(r)
            if h > 0:
                gr = -math.log(h) + psi1 - sp_digamma(rr / h + 1.0)
            elif abs(h) < 1e-12:
                gr = -math.log(rr) + psi1
            else:
                gr = -math.log(-h) + psi1 - sp_digamma(-rr / h)
            if not np.isfinite(gr):
                return None
            gam.append(float(gr))
        return gam
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def _fit_loc_scale(l1, l2, k, h, return_diag=False):
    """Calcule (ξ, α) depuis (L1, L2, k, h).

    Référence : [2] eq. 33–34 :
        α = k·L2 / (g1 - g2)
        ξ = L1 - (α/k)·(1 - g1)        si |k| ≥ ε

    Limite k→0 (|k| < _KAPPA4_K_EPS) — sinon α = 0/0 (logistique h=−1,
    Gumbel h=0, exponentielle h=1, toute la ligne k=0). Avec γ_r = ∂g_r/∂k|₀
    (cf. _g_deriv_k0) : (g₁−g₂)/k → γ₁−γ₂ et (1−g₁)/k → −γ₁, donc
        α = L2 / (γ₁ − γ₂) ;   ξ = L1 + α·γ₁

    Rejette si α ≤ 0 ou non-fini, ou ξ non-fini.
    Si return_diag=True, retourne aussi un dict de diagnostics (g1, g2, etc.).
    """
    if abs(k) < _KAPPA4_K_EPS:
        gam = _g_deriv_k0(h, rmax=2)
        if gam is None:
            return (None, {'g_failure': True}) if return_diag else None
        g1d, g2d = gam[0], gam[1]
        denom = g1d - g2d
        diag = {'g1': 1.0, 'g2': 1.0, 'g1_minus_g2': float(denom),
                'k0_limit': True}
        alpha = l2 / denom if abs(denom) > 1e-10 else l2
        if not np.isfinite(alpha) or alpha <= 1e-9:
            return (None, diag) if return_diag else None
        xi = l1 + alpha * g1d
        if not np.isfinite(xi):
            return (None, diag) if return_diag else None
        return ((xi, alpha), diag) if return_diag else (xi, alpha)

    g = _g_functions(k, h, rmax=3)
    if g is None:
        return (None, {'g_failure': True}) if return_diag else None
    g1, g2, _ = g
    denom = g1 - g2
    diag = {'g1': float(g1), 'g2': float(g2), 'g1_minus_g2': float(denom)}
    alpha = (k * l2) / denom if abs(denom) > 1e-10 else l2
    if not np.isfinite(alpha) or alpha <= 1e-9:
        return (None, diag) if return_diag else None
    xi = l1 - (alpha / k) * (1.0 - g1) if abs(k) > 1e-10 else l1
    if not np.isfinite(xi):
        return (None, diag) if return_diag else None
    return ((xi, alpha), diag) if return_diag else (xi, alpha)


def _tau3_tau4_from_kh_analytic(k, h):
    """Calcule (τ3, τ4) théoriques à partir de (k, h) via les g_r de Hosking.

    Référence : [2] eq. 18-19 (pour les expressions de l3, l4 en fonction des g_r) :
        l2 ∝ g1 - g2
        l3 ∝ -(g1 - 3·g2 + 2·g3)
        l4 ∝  g1 - 6·g2 + 10·g3 - 5·g4
        τ3 = l3/l2,  τ4 = l4/l2

    Remplace l'usage des polynômes Mielke d'ordre 6 ([2] Tableaux 2.1–2.2) qui
    dépendent de h discret. Calcul direct continu en (k, h).

    Pour |k| < _KAPPA4_K_EPS, (g₁−g₂)→0 (forme 0/0) : on substitue les g_r
    par leurs dérivées γ_r = ∂g_r/∂k|₀ (cf. _g_deriv_k0). Les constantes en r
    s'éliminent dans les ratios ⇒ limite exacte (logistique : τ₃=0, τ₄=1/6 ;
    Gumbel : τ₃≈0.1699, τ₄≈0.1504 ; exponentielle : τ₃=1/3, τ₄=1/6).
    """
    if abs(k) < _KAPPA4_K_EPS:
        gam = _g_deriv_k0(h, rmax=4)
        if gam is None:
            return np.nan, np.nan
        g1, g2, g3, g4 = gam[0], gam[1], gam[2], gam[3]
    else:
        g = _g_functions(k, h, rmax=4)
        if g is None or len(g) < 4:
            return np.nan, np.nan
        g1, g2, g3, g4 = g[0], g[1], g[2], g[3]
    l2_n = g1 - g2
    if abs(l2_n) < 1e-12 or not np.isfinite(l2_n):
        return np.nan, np.nan
    l3_n = -(g1 - 3.0 * g2 + 2.0 * g3)
    l4_n = g1 - 6.0 * g2 + 10.0 * g3 - 5.0 * g4
    return l3_n / l2_n, l4_n / l2_n


def ajuster_kappa4_pwm_analytic(data, warm_start=None):
    """Ajustement Kappa4 par L-moments analytiques + fsolve.

    Référence : [2] §4 (procédure complète d'inférence), eq. 28-34.
                Loi de Hosking [6].

    Algorithme :
        1. Calcule (l1, l2, τ3, τ4) empiriques sur `data`.
        2. Résout par fsolve le système 2×2 :
              τ3_théorique(k, h) = τ3_empirique
              τ4_théorique(k, h) = τ4_empirique
           démarré sur warm_start (défaut KAPPA4_WARM_START_INITIAL, carte EXPERT).
           Si la 1ʳᵉ tentative échoue (ier≠1 ou résidu² > seuil) et que
           KAPPA4_RETRY_ENABLED=True, on rejoue sur KAPPA4_RETRY_WARM_STARTS.
        3. Rejette si :
              - fsolve n'a pas convergé (ier ≠ 1) sur toutes les tentatives,
              - résidu² > KAPPA4_RESIDUAL_TOL,
              - α ≤ 0 ou ξ non-fini.
        4. Calcule (ξ, α) par _fit_loc_scale (eq. 33–34).

    Retour
    ------
    dict avec clés :
        'xi', 'alpha', 'k', 'h'   : paramètres Kappa4 (xi=loc, alpha=scale)
        't3', 't4'                : τ3, τ4 empiriques (input)
        'tau3', 'tau4'            : τ3, τ4 recalculés depuis (k*, h*) (vérif)
        'success'                 : bool — True si fit accepté
        '_t_ms'                   : durée du fit en ms (instrumentation)
        '_debug' (opt)            : si KAPPA4_DEBUG_ANALYTIC=True, trace fsolve

    Notes
    -----
    - warm_start peut être réutilisé entre appels successifs (continuité en f₀)
      pour accélérer la convergence — pas exploité actuellement (chaque fit est
      indépendant), pourrait être une optimisation V4.
    """
    from scipy.optimize import fsolve

    res = {'xi': float(np.median(data)), 'alpha': float(np.std(data)),
           'k': 0.1, 'h': 0.1, 't3': np.nan, 't4': np.nan,
           'tau3': None, 'tau4': None, 'success': False,
           'fail_reason': 'not_run'}

    debug_on = bool(KAPPA4_DEBUG_ANALYTIC)
    dbg = None
    if debug_on:
        dbg = {'n_data': int(len(data)),
               'warm_start': None, 'l1': np.nan, 'l2': np.nan,
               't3': np.nan, 't4': np.nan,
               'fsolve_ier': None, 'fsolve_sol_k': np.nan,
               'fsolve_sol_h': np.nan,
               'fsolve_res1': np.nan, 'fsolve_res2': np.nan,
               'fsolve_res_norm2': np.nan,
               'tau3_calc': np.nan, 'tau4_calc': np.nan,
               'xi': np.nan, 'alpha': np.nan,
               'g1': np.nan, 'g2': np.nan, 'g1_minus_g2': np.nan,
               'exit_reason': 'unknown'}

    try:
        l1, l2, t3, t4 = calculer_lmoments(data)
        res['t3'], res['t4'] = t3, t4
        if debug_on:
            dbg['l1'] = float(l1); dbg['l2'] = float(l2)
            dbg['t3'] = float(t3) if np.isfinite(t3) else np.nan
            dbg['t4'] = float(t4) if np.isfinite(t4) else np.nan
        if abs(l2) < KAPPA4_L2_MIN_FOR_FIT or not (np.isfinite(t3) and np.isfinite(t4)):
            res['fail_reason'] = 'l2_or_tau_invalid'
            if debug_on:
                dbg['exit_reason'] = 'l2_or_tau_invalid'
                res['_debug'] = dbg
            return res

        def equations(params):
            k, h = params
            tau3_c, tau4_c = _tau3_tau4_from_kh_analytic(k, h)
            if not (np.isfinite(tau3_c) and np.isfinite(tau4_c)):
                return [1e6, 1e6]
            return [tau3_c - t3, tau4_c - t4]

        x0_initial = (warm_start if (warm_start is not None
                      and all(np.isfinite(warm_start)))
                      else tuple(KAPPA4_WARM_START_INITIAL))
        if debug_on:
            dbg['warm_start'] = (float(x0_initial[0]), float(x0_initial[1]))

        attempts = [(x0_initial, int(200 * (2 + 1)))]  # défaut fsolve = 200·(N+1)
        if KAPPA4_RETRY_ENABLED:
            attempts += [(tuple(x0), int(KAPPA4_RETRY_MAXFEV))
                         for x0 in KAPPA4_RETRY_WARM_STARTS]

        sol = None
        ier = None
        res_norm2 = np.inf
        last_exc_name = None
        retry_idx = -1
        for idx, (x0, maxfev) in enumerate(attempts):
            try:
                sol_t, info_t, ier_t, _ = fsolve(
                    equations, x0, full_output=True,
                    xtol=float(KAPPA4_ANALYTIC_XTOL), maxfev=int(maxfev))
            except Exception as exc:
                last_exc_name = type(exc).__name__
                continue
            if ier_t != 1:
                if ier is None:  # garde la 1ʳᵉ trace d'échec pour le debug
                    ier = ier_t
                continue
            r_chk_t = equations((float(sol_t[0]), float(sol_t[1])))
            rn2 = (r_chk_t[0] ** 2 + r_chk_t[1] ** 2) if all(np.isfinite(r_chk_t)) else np.inf
            if not np.isfinite(rn2) or rn2 > KAPPA4_RESIDUAL_TOL:
                ier = ier_t
                res_norm2 = rn2
                continue
            # Convergé + résidu OK
            sol, ier, res_norm2 = sol_t, ier_t, rn2
            retry_idx = idx
            break

        if sol is None:
            # Tous les essais ont échoué : remonte la 1ʳᵉ raison rencontrée.
            if last_exc_name is not None and ier is None:
                res['fail_reason'] = f'fsolve_exception:{last_exc_name}'
                if debug_on:
                    dbg['exit_reason'] = f'fsolve_exception:{last_exc_name}'
                    res['_debug'] = dbg
                return res
            if ier is None:
                res['fail_reason'] = 'fsolve_no_attempt'
                if debug_on:
                    dbg['exit_reason'] = 'fsolve_no_attempt'
                    res['_debug'] = dbg
                return res
            if not np.isfinite(res_norm2) or res_norm2 > KAPPA4_RESIDUAL_TOL:
                # Au moins un essai a convergé (ier=1) mais résidu trop grand.
                res['fail_reason'] = 'residual_too_large'
                if debug_on:
                    dbg['fsolve_res_norm2'] = float(res_norm2) if np.isfinite(res_norm2) else np.inf
                    dbg['exit_reason'] = 'residual_too_large'
                    res['_debug'] = dbg
                return res
            res['fail_reason'] = f'fsolve_not_converged(ier={ier})'
            if debug_on:
                dbg['fsolve_ier'] = int(ier)
                dbg['exit_reason'] = f'fsolve_not_converged(ier={ier})'
                res['_debug'] = dbg
            return res

        k_star, h_star = float(sol[0]), float(sol[1])
        if debug_on:
            dbg['fsolve_ier']       = int(ier)
            dbg['fsolve_sol_k']     = k_star
            dbg['fsolve_sol_h']     = h_star
            r_chk = equations((k_star, h_star))
            dbg['fsolve_res1']      = float(r_chk[0]) if np.isfinite(r_chk[0]) else np.nan
            dbg['fsolve_res2']      = float(r_chk[1]) if np.isfinite(r_chk[1]) else np.nan
            dbg['fsolve_res_norm2'] = float(res_norm2) if np.isfinite(res_norm2) else np.inf
        # Trace télémétrique du retry pour QA / réglage futur du warm_start.
        res['solve_attempt']  = retry_idx
        res['solve_recovered'] = (retry_idx > 0)

        if debug_on:
            ls_out = _fit_loc_scale(l1, l2, k_star, h_star, return_diag=True)
            loc_scale, ls_diag = ls_out
            for key in ('g1', 'g2', 'g1_minus_g2'):
                if key in ls_diag:
                    dbg[key] = ls_diag[key]
        else:
            loc_scale = _fit_loc_scale(l1, l2, k_star, h_star)
        if loc_scale is None:
            res['fail_reason'] = 'loc_scale_invalid'
            if debug_on:
                dbg['exit_reason'] = 'loc_scale_invalid'
                res['_debug'] = dbg
            return res
        xi_star, alpha_star = loc_scale

        tau3_calc, tau4_calc = _tau3_tau4_from_kh_analytic(k_star, h_star)
        res.update({'xi': xi_star, 'alpha': alpha_star,
                    'k': k_star, 'h': h_star,
                    'tau3': tau3_calc, 'tau4': tau4_calc, 'success': True,
                    'fail_reason': 'ok_retry' if res.get('solve_recovered') else 'ok'})
        if debug_on:
            dbg.update({'xi': float(xi_star), 'alpha': float(alpha_star),
                        'tau3_calc': float(tau3_calc),
                        'tau4_calc': float(tau4_calc),
                        'exit_reason': 'ok'})
            res['_debug'] = dbg
    except Exception as exc:
        res['fail_reason'] = f'outer_exception:{type(exc).__name__}'
        if debug_on:
            dbg['exit_reason'] = f'outer_exception:{type(exc).__name__}'
            res['_debug'] = dbg
    return res


def _fit_analytic(data, warm_start=None):
    """Wrapper instrumenté autour de ajuster_kappa4_pwm_analytic."""
    t0 = time.perf_counter()
    out = ajuster_kappa4_pwm_analytic(data, warm_start=warm_start)
    dt = time.perf_counter() - t0
    KAPPA4_TIMINGS['analytic']   += dt
    KAPPA4_TIMINGS['analytic_n'] += 1
    out['_t_ms'] = dt * 1000.0
    return out


def ajuster_kappa4(data):
    """Point d'entrée unique — V3 : direct_pwm_analytic uniquement."""
    return _fit_analytic(data)


def _msdi_queue_droite(sorted_d, ppf_vec):
    """MSDI — Mean Square Deviation Index (NF X50-144-3 §C.7 ; CFM 2025 §3.1).

        MSDI(%) = (100/m) · Σ_{i=n-m+1..n} [ (X_i:n^exp − X_i:n^loi) / X_i:n^exp ]²

    Écart quadratique RELATIF entre les quantiles empiriques (queue droite)
    et les quantiles de la loi ajustée, évalués aux mêmes positions de tracé
    (Cunnane a=0.4, préconisée par [1] §C.7). La comparaison ne porte que
    sur les m points expérimentaux situés AU-DELÀ de la moyenne (priorité à
    l'ajustement de la queue de distribution). La loi présentant le plus
    petit MSDI est la mieux ajustée. Toujours ≥ 0, exprimé en %.

    Paramètres
    ----------
    sorted_d : np.ndarray — échantillon TRIÉ croissant.
    ppf_vec  : callable   — fonction quantile vectorisée de la loi ajustée,
                            p ∈ (0,1)^m → x^loi (np.ndarray).

    Retour : float MSDI (%) ou NaN si dégénéré (m=0, X_exp≈0, PPF non finie).
    """
    n = sorted_d.size
    if n == 0:
        return np.nan
    mean_v = float(np.mean(sorted_d))
    idx = np.where(sorted_d > mean_v)[0]
    m = idx.size
    if m == 0:
        return np.nan
    # Positions de tracé Cunnane (a = 0.4) des rangs i = idx+1 (1..n).
    p_i = ((idx + 1) - 0.4) / (n + 0.2)
    x_fit = np.asarray(ppf_vec(p_i), dtype=float)
    x_exp = sorted_d[idx]
    valid = np.isfinite(x_fit) & (np.abs(x_exp) > 1e-300)
    if not np.any(valid):
        return np.nan
    rel = (x_exp[valid] - x_fit[valid]) / x_exp[valid]
    return float(100.0 / valid.sum() * np.sum(rel ** 2))


def kappa4_rmse_msdi(data, params):
    """Métriques de qualité de l'ajustement Kappa4 vs échantillon empirique.

    Référence : [1] §C.7 (p. 80) — tests KS-M et MSDI sur la queue droite.

    Métriques calculées :
        RMSE = √(mean((ECDF - CDF_K4)²)) — écart RMS en espace probabilité,
               sur tout l'échantillon (diagnostic global, hors norme).
        MSDI = Mean Square Deviation Index ([1] §C.7, en %) — écart
               quadratique relatif des QUANTILES sur la queue droite
               (m points > moyenne), positions Cunnane a=0.4 :
               MSDI(%) = (100/m)·Σ[(X_exp − X_K4)/X_exp]². Toujours ≥ 0 ;
               plus petit = meilleur ajustement de la queue.
        KS   = max|ECDF - CDF_K4|       — distance Kolmogorov-Smirnov
        pearson_r                        — corrélation ECDF / CDF_K4

    Mute `params` : ajoute 'ks' et 'pearson_r' (setdefault, n'écrase pas).
    """
    if not params.get('success'):
        return np.nan, np.nan
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    sorted_d = np.sort(data)
    ecdf     = np.arange(1, n + 1) / n
    try:
        if _kappa4_use_exact(params['k'], params['h']):
            cdf_v = _kappa4_cdf_exact(sorted_d, params['xi'],
                                      params['alpha'], params['k'],
                                      params['h'])
            ppf_vec = lambda p: _kappa4_ppf_exact(p, params['xi'],
                                                  params['alpha'],
                                                  params['k'], params['h'])
        else:
            loi = scipy_kappa4(h=params['h'], k=params['k'],
                               loc=params['xi'], scale=params['alpha'])
            cdf_v   = loi.cdf(sorted_d)
            ppf_vec = loi.ppf
        diff  = ecdf - cdf_v
        rmse  = float(np.sqrt(np.mean(diff ** 2)))
        msdi  = _msdi_queue_droite(sorted_d, ppf_vec)
        try:
            params.setdefault('ks', float(np.max(np.abs(diff))))
            if np.std(cdf_v) > 1e-12 and np.std(ecdf) > 1e-12:
                params.setdefault('pearson_r',
                                  float(np.corrcoef(ecdf, cdf_v)[0, 1]))
            else:
                params.setdefault('pearson_r', np.nan)
        except Exception:
            pass
        return rmse, msdi
    except Exception:
        return np.nan, np.nan


def _kappa4_use_exact(k, h):
    """Vrai si l'on doit éviter scipy.stats.kappa4 (branche k==0 buggée :
    `np.log(h)` ⇒ NaN silencieux pour h ≤ 0). On bascule sur les formules
    de Hosking analytiques dans la zone dégénérée uniquement ; scipy reste
    utilisé partout ailleurs (zéro régression)."""
    return (abs(k) < _KAPPA4_K_EPS) or (h <= 0.0 and abs(k) < 1e-3)


def _kappa4_ppf_exact(prob, xi, alpha, k, h):
    """Quantile Kappa4 EXACTE (Hosking), avec limites h→0 et k→0.

        x = ξ + (α/k)·[1 − ((1−F^h)/h)^k]
      h→0 : (1−F^h)/h → −ln F      k→0 : x = ξ − α·ln((1−F^h)/h)

    Convention IDENTIQUE à scipy.stats.kappa4, mais sans le `np.log(h)`
    fautif pour la branche k=0, h≤0 (logistique, Gumbel). Vectorisé numpy ;
    valeurs non finies → NaN (le site appelant filtre comme avec scipy).
    """
    F = np.asarray(prob, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        if abs(h) < 1e-12:
            inner = -np.log(F)                       # limite h→0
        else:
            inner = (1.0 - np.power(F, h)) / h
        inner = np.where(inner > 0.0, inner, np.nan)
        if abs(k) < _KAPPA4_K_EPS:
            x = xi - alpha * np.log(inner)           # limite k→0
        else:
            x = xi + (alpha / k) * (1.0 - np.power(inner, k))
    return x


def _kappa4_cdf_exact(x, xi, alpha, k, h):
    """CDF Kappa4 EXACTE (Hosking), avec limites k→0 et h→0.

        F = [1 − h·A]^(1/h),  A = {1 − k·(x−ξ)/α}^(1/k)
      k→0 : A = exp(−(x−ξ)/α)      h→0 : F = exp(−A)

    Même motivation que `_kappa4_ppf_exact`. Résultat clampé dans [0, 1].
    """
    xx = np.asarray(x, dtype=np.float64)
    y  = (xx - xi) / alpha
    with np.errstate(divide='ignore', invalid='ignore'):
        if abs(k) < _KAPPA4_K_EPS:
            A = np.exp(-y)                           # limite k→0
        else:
            base = 1.0 - k * y
            base = np.where(base > 0.0, base, 0.0)
            A = np.power(base, 1.0 / k)
        if abs(h) < 1e-12:
            F = np.exp(-A)                           # limite h→0
        else:
            base2 = 1.0 - h * A
            base2 = np.where(base2 > 0.0, base2, 0.0)
            F = np.power(base2, 1.0 / h)
    return np.clip(F, 0.0, 1.0)


def kappa4_ppf(params, prob):
    """Quantile (PPF) de la loi Kappa4 pour une probabilité prob ∈ (0, 1).

    Délègue à scipy.stats.kappa4(h, k, loc=ξ, scale=α).ppf(prob), SAUF en
    zone dégénérée (cf. _kappa4_use_exact) où scipy renvoie NaN : on utilise
    alors la quantile analytique exacte. Retourne None si fit invalide,
    prob hors (0, 1), ou résultat non fini.
    """
    if not params.get('success') or not (0 < prob < 1):
        return None
    try:
        k, h = params['k'], params['h']
        if _kappa4_use_exact(k, h):
            v = float(_kappa4_ppf_exact(prob, params['xi'],
                                        params['alpha'], k, h))
        else:
            v = float(scipy_kappa4(h=h, k=k, loc=params['xi'],
                                   scale=params['alpha']).ppf(prob))
        return v if np.isfinite(v) else None
    except Exception:
        return None

# =============================================================================
# SECTION 6bis — LOI DE RAYLEIGH GÉNÉRALISÉE (Kundu & Raqab)
# =============================================================================
#
# Référence : CFM 2025 (Clou/Lelan) §4.2 — loi de Rayleigh généralisée, modèle
#             exact de l'excitation gaussienne en environnement vibratoire :
#                 F(x;α,λ) = (1 − e^(−(λx)²))^α     (x > 0, α > 0, λ > 0)
#             α = paramètre de forme, λ = paramètre d'échelle (taux).
#
# C'est une famille DISTINCTE de Kappa4 (l'article les oppose). Choisie via la
# constante LOI_AJUSTEMENT='rayleigh_gen'. Ajustement par L-moments (réutilise
# calculer_lmoments) ; PPF analytique exacte ⇒ pas de solveur pour le quantile.

# Nœuds Gauss-Legendre (transposés sur p ∈ (0,1)) pour intégrer les L-moments
# et les moments via la fonction quantile. 512 nœuds → erreur < 1e-6 sur le
# domaine α ∈ [1e-2, 3e2] malgré la singularité log intégrable en p→1.
from numpy.polynomial.legendre import leggauss as _leggauss
_RG_GL_X, _RG_GL_W = _leggauss(512)
_RG_P  = 0.5 * (_RG_GL_X + 1.0)          # p ∈ (0,1)
_RG_PW = 0.5 * _RG_GL_W                   # poids associés
_RG_P  = np.clip(_RG_P, 1e-12, 1.0 - 1e-12)


def _rg_quantile_unit(p, alpha):
    """Fonction quantile de la Rayleigh généralisée à λ=1 :
        Q(p) = √( −ln(1 − p^(1/α)) )
    (la PPF complète vaut Q(p)/λ)."""
    val = 1.0 - np.power(p, 1.0 / alpha)
    val = np.clip(val, 1e-300, 1.0)
    return np.sqrt(-np.log(val))


def _rg_lmoments_unit(alpha):
    """(L1, L2, L3, L4) de la Rayleigh généralisée à λ=1, par quadrature
    Gauss-Legendre de la quantile (les ratios τ3=L3/L2, τ4=L4/L2 ne dépendent
    que de la forme α ; L1, L2 sont en 1/λ)."""
    p, w = _RG_P, _RG_PW
    q  = _rg_quantile_unit(p, alpha)
    L1 = float(np.sum(w * q))
    L2 = float(np.sum(w * q * (2.0 * p - 1.0)))
    L3 = float(np.sum(w * q * (6.0 * p * p - 6.0 * p + 1.0)))
    L4 = float(np.sum(w * q * (20.0 * p**3 - 30.0 * p * p + 12.0 * p - 1.0)))
    return L1, L2, L3, L4


def _rg_tau3(alpha):
    """τ3 théorique en fonction de la seule forme α."""
    _, L2, L3, _ = _rg_lmoments_unit(alpha)
    return L3 / L2 if abs(L2) > 1e-15 else np.nan


def _rg_tau4(alpha):
    """τ4 théorique en fonction de la seule forme α (diagnostic)."""
    _, L2, _, L4 = _rg_lmoments_unit(alpha)
    return L4 / L2 if abs(L2) > 1e-15 else np.nan


def ajuster_rayleigh_gen(data):
    """Ajustement Rayleigh généralisée par L-moments modifiés (MLME).

    Référence : Kundu & Raqab, « Generalized Rayleigh Distribution: Different
    Methods of Estimations », §6 éq. (18)-(20). Les moments de la GR sur X
    n'ont pas de forme close ; en revanche la transformée Y = X² suit une loi
    exponentielle généralisée GE(α, θ) avec θ = λ², dont les deux premiers
    L-moments s'expriment en fonctions digamma (ψ) :

      1. y = x² triés ; L-moments échantillon (éq. 18) :
            l1 = mean(y)
            l2 = (2/(n(n−1)))·Σ_{i=1..n} (i−1)·y_(i) − l1
      2. α : racine de l'éq. (20) — rapport sans échelle, strictement
         décroissant de 1 (α→0⁺) vers 0 (α→∞), racine unique :
            [ψ(2α+1) − ψ(α+1)] / [ψ(α+1) − ψ(1)] = l2 / l1
      3. θ = [ψ(α̂+1) − ψ(1)] / l1 ;  λ = √θ  (échelle GR).

    Retour : dict aux mêmes clés que ajuster_kappa4 (compatibilité aval) —
        'success','fail_reason','xi','alpha','k','h','t3','t4','tau3','tau4',
        'loi'='rayleigh_gen' ; plus 'rg_alpha' (forme α), 'rg_lambda' (échelle
        λ). tau3/tau4 sont renseignés à titre diagnostique depuis α̂.
    """
    from scipy.optimize import brentq
    from scipy.special import digamma

    res = {'xi': 0.0, 'alpha': float(np.std(data) or 1.0),
           'k': np.nan, 'h': np.nan, 't3': np.nan, 't4': np.nan,
           'tau3': None, 'tau4': None, 'success': False,
           'fail_reason': 'not_run', 'loi': 'rayleigh_gen',
           'rg_alpha': np.nan, 'rg_lambda': np.nan}
    try:
        x = np.asarray(data, dtype=np.float64)
        x = x[np.isfinite(x) & (x > 0.0)]
        n = x.size
        if n < 3:
            res['fail_reason'] = 'rg_too_few_points'
            return res

        # Transformée Y = X² puis L-moments échantillon (Kundu & Raqab éq. 18).
        y  = np.sort(x * x)
        iv = np.arange(1, n + 1, dtype=np.float64)
        l1 = float(np.mean(y))
        l2 = float(2.0 / (n * (n - 1.0)) * np.sum((iv - 1.0) * y) - l1)
        if not (np.isfinite(l1) and np.isfinite(l2)) or l1 <= 0.0:
            res['fail_reason'] = 'rg_l1_invalid'
            return res
        ratio = l2 / l1                       # L-moments de GE : ∈ (0, 1)
        if not (0.0 < ratio < 1.0):
            res['fail_reason'] = 'rg_lmoment_ratio_out_of_range'
            return res

        # Éq. (20) : g(α) = R(α) − l2/l1, R strictement décroissante (1 → 0).
        psi1 = digamma(1.0)

        def g(a):
            return ((digamma(2.0 * a + 1.0) - digamma(a + 1.0))
                    / (digamma(a + 1.0) - psi1)) - ratio

        a_lo, a_hi = 1e-4, 1e4
        g_lo, g_hi = g(a_lo), g(a_hi)
        tries = 0
        while g_lo * g_hi > 0.0 and tries < 6:
            a_lo *= 0.1
            a_hi *= 10.0
            g_lo, g_hi = g(a_lo), g(a_hi)
            tries += 1
        if not (np.isfinite(g_lo) and np.isfinite(g_hi)) or g_lo * g_hi > 0.0:
            res['fail_reason'] = 'rg_alpha_no_bracket'
            return res

        alpha_hat = float(brentq(g, a_lo, a_hi, xtol=1e-10, maxiter=200))
        theta = (digamma(alpha_hat + 1.0) - psi1) / l1   # θ = λ²
        if not (np.isfinite(alpha_hat) and alpha_hat > 0.0
                and np.isfinite(theta) and theta > 0.0):
            res['fail_reason'] = 'rg_params_invalid'
            return res
        lambda_hat = float(np.sqrt(theta))
        if not (np.isfinite(lambda_hat) and lambda_hat > 0.0):
            res['fail_reason'] = 'rg_scale_invalid'
            return res

        res.update({
            'rg_alpha': alpha_hat, 'rg_lambda': lambda_hat,
            'xi': 0.0, 'alpha': 1.0 / lambda_hat,   # 'alpha' = échelle (1/λ)
            'k': alpha_hat, 'h': np.nan,             # mapping clés Kappa4
            'tau3': _rg_tau3(alpha_hat), 'tau4': _rg_tau4(alpha_hat),
            'success': True, 'fail_reason': 'ok'})
    except Exception as exc:
        res['fail_reason'] = f'rg_exception:{type(exc).__name__}'
    return res


def rayleigh_gen_ppf(params, prob):
    """Quantile (PPF) analytique : x(p) = (1/λ)·√( −ln(1 − p^(1/α)) ).
    Retourne None si fit invalide ou prob ∉ (0,1)."""
    if not params.get('success') or not (0.0 < prob < 1.0):
        return None
    try:
        a = float(params['rg_alpha']); lam = float(params['rg_lambda'])
        inner = 1.0 - prob ** (1.0 / a)
        if not (0.0 < inner <= 1.0):
            return None
        return float(np.sqrt(-np.log(inner)) / lam)
    except Exception:
        return None


def rayleigh_gen_rmse_msdi(data, params):
    """RMSE / MSDI de l'ajustement Rayleigh généralisée vs échantillon.
    Mêmes définitions que kappa4_rmse_msdi : RMSE en espace probabilité
    (tout l'échantillon) ; MSDI = Mean Square Deviation Index [1] §C.7
    (quantiles relatifs, queue droite, %). Mute params ('ks', 'pearson_r')."""
    if not params.get('success'):
        return np.nan, np.nan
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    try:
        a = float(params['rg_alpha']); lam = float(params['rg_lambda'])
        sorted_d = np.sort(data)
        ecdf = np.arange(1, n + 1) / n
        z = np.clip(lam * sorted_d, 0.0, None)
        cdf_v = np.power(1.0 - np.exp(-(z * z)), a)
        diff  = ecdf - cdf_v
        rmse  = float(np.sqrt(np.mean(diff ** 2)))
        msdi  = _msdi_queue_droite(
            sorted_d,
            lambda p: _rg_quantile_unit(np.asarray(p, dtype=float), a) / lam)
        try:
            params.setdefault('ks', float(np.max(np.abs(diff))))
            if np.std(cdf_v) > 1e-12 and np.std(ecdf) > 1e-12:
                params.setdefault('pearson_r',
                                  float(np.corrcoef(ecdf, cdf_v)[0, 1]))
            else:
                params.setdefault('pearson_r', np.nan)
        except Exception:
            pass
        return rmse, msdi
    except Exception:
        return np.nan, np.nan


def _rayleigh_gen_mean_var(params):
    """(μ, σ²) de la Rayleigh généralisée ajustée, par quadrature de la PPF
    (analogue _kappa4_mean_var, pour la projection lognormale du dommage)."""
    if not params.get('success'):
        return None, None
    try:
        a = float(params['rg_alpha']); lam = float(params['rg_lambda'])
        q  = _rg_quantile_unit(_RG_P, a) / lam
        w  = _RG_PW
        mu = float(np.sum(w * q))
        e2 = float(np.sum(w * q * q))
        var = max(0.0, e2 - mu * mu)
        if not (np.isfinite(mu) and np.isfinite(var)):
            return None, None
        return mu, var
    except Exception:
        return None, None


# --- Dispatchers neutres (routage Kappa4 / Rayleigh généralisée) -------------

def ajuster_loi(data):
    """Ajuste la loi sélectionnée par LOI_AJUSTEMENT. Tag 'loi' garanti."""
    if LOI_AJUSTEMENT == 'rayleigh_gen':
        return ajuster_rayleigh_gen(data)
    out = ajuster_kappa4(data)
    out.setdefault('loi', 'kappa4')
    return out


def loi_ppf(params, prob):
    """PPF routée selon params['loi'] (défaut : Kappa4)."""
    if params.get('loi') == 'rayleigh_gen':
        return rayleigh_gen_ppf(params, prob)
    return kappa4_ppf(params, prob)


def loi_rmse_msdi(data, params):
    """RMSE/MSDI routés selon params['loi'] (défaut : Kappa4)."""
    if params.get('loi') == 'rayleigh_gen':
        return rayleigh_gen_rmse_msdi(data, params)
    return kappa4_rmse_msdi(data, params)


def _loi_mean_var(params):
    """(μ, σ²) routés selon params['loi'] (défaut : Kappa4)."""
    if params.get('loi') == 'rayleigh_gen':
        return _rayleigh_gen_mean_var(params)
    return _kappa4_mean_var(params)


# =============================================================================
# SECTION 6ter — PROJECTION PAR LES 3 DOMAINES GEV (h = 0)
# =============================================================================
#
# Référence : [2] Colin §4.1 (note de bas de page 2) — discrimination du
# domaine d'attraction de la loi locale KAPPA(ξ,α,k,h) : Z_sup tend, pour
# M grand, vers Gumbel (k*=0), Fréchet (k*<0) ou Weibull négative (k*>0).
# Activée par METHODE_PROJECTION = 'gev_domaines' (cf. SECTION 1).
#
# La GEV est la sous-famille h = 0 de Kappa4 (g_r = r^{-k}·Γ(1+k), eq. 20.2
# de [2]). On fige h = 0 dans le processus d'inférence : k* est alors racine
# de l'unique équation τ3_GEV(k) = t3 (au lieu du système 2×2 en (k,h)), et
# (ξ*, α*) suivent par les eq. 33-34 de [2] avec h = 0.

def _gev_domaine(k):
    """Domaine d'attraction GEV selon le signe de k (convention Hosking) :
    'gumbel' (EV1, |k| < GEV_GUMBEL_K_TOL), 'frechet' (EV2, k < 0),
    'weibull_neg' (EV3, k > 0). Étiquette de DIAGNOSTIC uniquement — la
    formule de projection (calculer_projection_gev_domaines) est continue
    en k et n'utilise pas ce seuil."""
    if abs(k) < GEV_GUMBEL_K_TOL:
        return 'gumbel'
    return 'weibull_neg' if k > 0 else 'frechet'


def ajuster_gev_lmoments(data):
    """Ajustement GEV (= Kappa4 à h fixé 0) par L-moments.

    Algorithme :
      1. (l1, l2, t3, t4) empiriques (mêmes PWM non biaisés que Kappa4).
      2. k* racine de τ3_GEV(k) = t3 par `brentq` — τ3_GEV est strictement
         décroissante en k sur (−1, +∞) (de 1 vers −1), racine unique.
      3. (ξ*, α*) par _fit_loc_scale(l1, l2, k*, h=0) ([2] eq. 33-34).

    Retour : dict aux mêmes clés que ajuster_kappa4 ('h'=0.0, 'loi'='gev'),
    plus 'domaine' ∈ {'gumbel','frechet','weibull_neg'} ([2] §4.1).
    """
    from scipy.optimize import brentq

    res = {'xi': float(np.median(data)), 'alpha': float(np.std(data) or 1.0),
           'k': np.nan, 'h': 0.0, 't3': np.nan, 't4': np.nan,
           'tau3': None, 'tau4': None, 'success': False,
           'fail_reason': 'not_run', 'loi': 'gev', 'domaine': None}
    try:
        l1, l2, t3, t4 = calculer_lmoments(data)
        res['t3'], res['t4'] = t3, t4
        if abs(l2) < KAPPA4_L2_MIN_FOR_FIT or not np.isfinite(t3):
            res['fail_reason'] = 'l2_or_tau_invalid'
            return res
        if not (-1.0 < t3 < 1.0):
            res['fail_reason'] = 'gev_t3_out_of_range'
            return res

        def f(k):
            t3c, _ = _tau3_tau4_from_kh_analytic(float(k), 0.0)
            return (t3c - t3) if np.isfinite(t3c) else np.nan

        k_lo, k_hi = -0.99, 30.0
        f_lo, f_hi = f(k_lo), f(k_hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0.0:
            res['fail_reason'] = 'gev_k_no_bracket'
            return res
        k_star = float(brentq(f, k_lo, k_hi, xtol=1e-12, maxiter=200))

        loc_scale = _fit_loc_scale(l1, l2, k_star, 0.0)
        if loc_scale is None:
            res['fail_reason'] = 'loc_scale_invalid'
            return res
        xi_star, alpha_star = loc_scale
        tau3_c, tau4_c = _tau3_tau4_from_kh_analytic(k_star, 0.0)
        res.update({'xi': float(xi_star), 'alpha': float(alpha_star),
                    'k': k_star, 'h': 0.0,
                    'tau3': tau3_c, 'tau4': tau4_c,
                    'domaine': _gev_domaine(k_star),
                    'success': True, 'fail_reason': 'ok'})
    except Exception as exc:
        res['fail_reason'] = f'gev_exception:{type(exc).__name__}'
    return res


def calculer_projection_gev_domaines(params_gev, n_blocs_classe, total_blocs,
                                     Tb, T_proj, alfa):
    """Projection du SRE d'une classe par max-stabilité GEV — 3 domaines.

    Référence : [1] §C.8-C.9 (coefficient M), [2] §4.1 (lois asymptotiques
    de Z_sup), Fisher-Tippett / Gnedenko.

    La GEV est max-stable : si Z_max ~ GEV(ξ, α, k) alors le max de M
    tirages i.i.d. suit EXACTEMENT GEV(ξ_M, α_M, k), même paramètre de
    forme k (donc même domaine d'attraction), avec :

        k ≈ 0 (Gumbel, EV1)      : ξ_M = ξ + α·ln(M),           α_M = α
        k ≠ 0 (Fréchet  EV2 k<0,
               Weibull nég. EV3 k>0) :
                                   ξ_M = ξ + (α/k)·(1 − M^−k),  α_M = α·M^−k

    Le quantile projeté à probabilité de non-dépassement α est alors la
    forme close (y = −ln α) :
        SRE_α = ξ_M − α_M·ln(y)            si k ≈ 0
        SRE_α = ξ_M + (α_M/k)·(1 − y^k)    sinon

    Strictement équivalent à PPF_GEV(α^(1/M)) (méthode 'puissance' appliquée
    à la GEV) mais sans élévation de α à la puissance 1/M — donc sans perte
    de précision ni clip lorsque M est très grand (α^(1/M) → 1).

        M = (n_i / total_blocs) · T_proj / Tb     ([1] §C.8, Occ(j)·T_v/T_b)

    Retour
    ------
    (sre_proj, M, domaine) ou (None, None, None) si fit invalide ou M ≤ 0.
    """
    if (params_gev is None or not params_gev.get('success')
            or total_blocs <= 0 or Tb <= 0 or not (0.0 < alfa < 1.0)):
        return None, None, None
    M = (n_blocs_classe / total_blocs) * T_proj / Tb
    if M <= 0:
        return None, None, None
    try:
        xi, alpha, k = (float(params_gev['xi']), float(params_gev['alpha']),
                        float(params_gev['k']))
        y = -math.log(alfa)
        if abs(k) < _KAPPA4_K_EPS:
            xi_M, alpha_M = xi + alpha * math.log(M), alpha
            sre_proj = xi_M - alpha_M * math.log(y)
        else:
            Mk      = M ** (-k)
            alpha_M = alpha * Mk
            xi_M    = xi + (alpha / k) * (1.0 - Mk)
            sre_proj = xi_M + (alpha_M / k) * (1.0 - y ** k)
        if not np.isfinite(sre_proj):
            return None, None, None
        return float(sre_proj), M, params_gev.get('domaine')
    except Exception:
        return None, None, None


# =============================================================================
# SECTION 7 — SRE & SRX ANALYTIQUES DEPUIS LA DSP (PR NORMDEF 0101)
# =============================================================================

def calculer_sre_analytique(signal, fs, Q, f0_grid,
                             alpha_srx_low=0.01, alpha_srx_high=0.99,
                             sdf_b=None, sdf_C=1.0,
                             T_proj=None, alfa_proj=None):
    """SRE / SRX / SDF analytiques depuis la DSP Welch — branche de comparaison.

    Cette branche calcule, depuis la DSP du signal d'entrée et la fonction
    de transfert d'un SDOF (1 DDL) en relatif z(t), trois spectres :

      - SRE  : Spectre de Réponse Extrême        (NORMDEF §5.4.2)
      - SRX  : Spectre de Réponse à risque α     (NORMDEF §5.4.3 eq. [5.2])
               calculé à deux niveaux α (LOW/HIGH) pour couvrir les deux
               usages métier (dimensionnement enveloppe haute / comparaison
               vs SRC d'un choc — cf. fig. 5.3 de la norme).

    Références
    ----------
    [4] PR NORMDEF 0101 (DGA 2009) :
          §5.4.2 — SRE = pic moyen sur T de la réponse en accélération
                   pseudo (2π·f₀)²·z_sup d'un SDOF gaussien narrow-band.
          §5.4.3 — SRX, formule non-asymptotique [5.2] :
                   R_X = (2π·f₀)²·z_eff·√(-2·ln(1-(1-α)^(1/(n₀⁺·T))))
                   et expression du rapport SRX/SRE eq. [5.3].
                   n₀⁺ = fréquence moyenne des passages par 0 de la réponse
                   d'un SDOF ; sous hypothèse narrow-band : n₀⁺ ≈ f₀.
                   Le SRE correspond au cas particulier de [5.2] avec
                   α → 1/(n₀⁺·T) — i.e. SRE = (2π·f₀)²·z_eff·√(2·ln(n₀⁺·T)).
    [5] B. Colin, MI0460 (2008) — discussion modèle non-asymptotique vs
          asymptotiques (Gumbel eq. [5.6], Poisson eq. [5.7]).
    [7] Lalanne Vol. 4 — théorie spectrale narrow-band gaussienne.

    Hypothèse : signal stationnaire gaussien (à vérifier — sinon utiliser
    la branche temporelle MBD-Kappa4).

    Étapes
    ------
      1. Welch (Hann, 50% overlap) → DSP Pxx(f).
      2. Pour chaque f₀ :
            z_ef² = ∫ Pxx(f) · |F_d(f, f₀, Q)|² df     (variance déplacement)
         où :
            F_d   = 1/(4π²·f₀²·√((1-ρ²)² + (2ξρ)²)),     ρ = f/f₀
      3. SRE  : (2π·f₀)²·z_eff·√(2·ln(n₀⁺·T))           [NORMDEF §5.4.2]
      4. SRX(α) : (2π·f₀)²·z_eff·√(-2·ln(1-(1-α)^(1/(n₀⁺·T))))   [eq. 5.2]
      5. SDF Bendat : D ≈ f₀·T·(√2·z_ef)^b · Γ(1+b/2) / C
         (E[range^b] pour Rayleigh narrow-band, σ_a = range/2)

    Si T_proj est fourni, recalcule SRE et SRX projetés en remplaçant T par
    T_proj (z_ef inchangé : c'est un moment stationnaire). Pour la
    projection, le risque effectif est α_proj = 1 - ALFA_PROJECTION (norme :
    ALFA_PROJECTION = probabilité de non-dépassement).

    Retour
    ------
    (sre_dsp, srx_low, srx_high,
     sre_dsp_proj, srx_low_proj, srx_high_proj,
     sdf_spectral)
        Tous de taille len(f0_grid). Les *_proj sont None si T_proj=None.
        sdf_spectral est None si sdf_b=None.
    """
    nperseg = min(len(signal), max(int(8 * fs), 256))
    f_psd, Pxx = welch(signal, fs=fs, window='hann', nperseg=nperseg,
                       noverlap=int(0.5 * nperseg), detrend='constant',
                       scaling='density', average='mean')

    f0_grid = np.asarray(f0_grid)
    xi      = 1.0 / (2.0 * Q)
    duree   = len(signal) / fs

    # Intégration analytique z_eff² = ∫ Pxx · |F_d|² df.
    rho_2d   = f_psd[:, np.newaxis] / f0_grid[np.newaxis, :]
    FdT_depl = 1.0 / (4.0 * np.pi**2 * f0_grid[np.newaxis, :]**2 *
                      np.sqrt((1.0 - rho_2d**2)**2 + (2.0 * xi * rho_2d)**2))
    z_ef = np.sqrt(np.maximum(
        simpson(Pxx[:, np.newaxis] * FdT_depl**2, f_psd, axis=0), 0.0))

    omega02 = 4.0 * np.pi**2 * f0_grid**2  # (2π·f₀)² — facteur pseudo-accélération

    # Hypothèse narrow-band : n₀⁺ ≈ f₀ (NORMDEF §5.4.3, juste après [5.2]).
    # n₀⁺·T borné ≥ 1 pour stabilité du log lorsque f₀·T < 1.
    n0T = np.maximum(f0_grid * duree, 1.0)

    # SRE — formule narrow-band gaussienne, NORMDEF §5.4.2
    # (limite de [5.2] avec α → 1/(n₀⁺·T) ⇒ SRE = ω₀²·z_eff·√(2·ln(n₀⁺·T))).
    sre_dsp = omega02 * z_ef * np.sqrt(2.0 * np.log(n0T))

    def _srx_normdef(alpha, n0T_):
        """SRX formule non-asymptotique [5.2] de PR NORMDEF 0101 §5.4.3.

        R_X(α) = (2π·f₀)²·z_eff·√(-2·ln(1-(1-α)^(1/(n₀⁺·T))))
        Clip de α dans (1e-12, 1-1e-12) et de l'argument du log dans
        [1e-300, +∞) pour éviter les instabilités numériques aux bords.
        """
        a = float(np.clip(alpha, 1e-12, 1.0 - 1e-12))
        with np.errstate(divide='ignore', invalid='ignore'):
            inner = np.maximum(1.0 - (1.0 - a) ** (1.0 / n0T_), 1e-300)
        return omega02 * z_ef * np.sqrt(np.maximum(-2.0 * np.log(inner), 0.0))

    srx_low  = _srx_normdef(alpha_srx_low,  n0T)
    srx_high = _srx_normdef(alpha_srx_high, n0T)

    sre_dsp_proj  = None
    srx_low_proj  = None
    srx_high_proj = None
    if T_proj is not None and T_proj > 0:
        # z_ef inchangé (moment stationnaire). Seul n₀⁺·T change : T = T_proj.
        n0T_p = np.maximum(f0_grid * float(T_proj), 1.0)
        sre_dsp_proj = omega02 * z_ef * np.sqrt(2.0 * np.log(n0T_p))
        srx_low_proj  = _srx_normdef(alpha_srx_low,  n0T_p)
        srx_high_proj = _srx_normdef(alpha_srx_high, n0T_p)

    if sdf_b is not None:
        # SDF spectral (Bendat-Lalanne narrow-band), convention AMPLITUDE
        # (cohérent avec rainflow ASTM E1049 / AFNOR A03-406, σ_a = range/2).
        # Pour z(t) gaussien narrow-band, l'amplitude S_a suit une Rayleigh
        # de paramètre σ_z = z_ef ; E[S_a^b] = (√2·σ_z)^b · Γ(1+b/2).
        # D ≈ n+·T·E[S_a^b]/C avec n+ ≈ f0.
        gamma_t      = sp_gamma(1.0 + sdf_b / 2.0)
        sdf_spectral = (f0_grid * duree * (math.sqrt(2.0) * z_ef) ** sdf_b
                        * gamma_t / sdf_C)
        return (sre_dsp, srx_low, srx_high,
                sre_dsp_proj, srx_low_proj, srx_high_proj,
                sdf_spectral)

    return (sre_dsp, srx_low, srx_high,
            sre_dsp_proj, srx_low_proj, srx_high_proj,
            None)

# =============================================================================
# SECTION 8 — SDF TEMPOREL PAR COMPTAGE RAINFLOW (ASTM E1049 / AFNOR A03-406)
# =============================================================================
#
# Algorithme par pile (stack-based) accéléré Numba.
# Convention : AMPLITUDE σ_a = range / 2  (Basquin : N · σ_a^b = C).
# D = Σ n_i · σ_a,i^b / C    (cycle complet : n_i = 1 ; résidu : n_i = 0.5)
#
# La compilation JIT est mise en cache disque (cache=True) pour qu'elle soit
# partagée entre les workers du ProcessPoolExecutor.
# -----------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _rainflow_extrema_numba(signal_data):
    """Extraction des points de retournement (extrema) en O(N), un seul passage."""
    n = len(signal_data)
    extrema = np.zeros(n)
    extrema[0] = signal_data[0]
    idx = 1
    last_slope = 0.0
    for i in range(1, n):
        delta = signal_data[i] - signal_data[i-1]
        if delta > 0.0:
            current_slope = 1.0
        elif delta < 0.0:
            current_slope = -1.0
        else:
            current_slope = 0.0
        if current_slope != 0.0:
            if last_slope != 0.0 and current_slope != last_slope:
                extrema[idx] = signal_data[i-1]
                idx += 1
            last_slope = current_slope
    extrema[idx] = signal_data[n-1]
    idx += 1
    return extrema[:idx]


@njit(cache=True, fastmath=True)
def _rainflow_stack_damage_numba(pts, C, b):
    """Comptage rainflow ASTM E1049-85 strict (équivalent Downing-Socie 4 points).

    Reproduit au bit près le package iamlikeme/rainflow.
    Règle (sur les 3 derniers points A, B, C de la pile, avec
    Y = |B-A| et X = |C-B|) :
      - X < Y                 → pas encore de cycle, on lit le suivant.
      - X >= Y et ptr == 3    → DEMI-CYCLE (range Y) : segment B-A inclut
                                le tout 1er point lu (résidu en cours).
                                On retire seulement A.
      - X >= Y et ptr  > 3    → CYCLE COMPLET (range Y) : segment B-A
                                encadré par les points antérieurs. On
                                retire B et C.
    Convention amplitude : σ_a = Y/2. D = Σ n·σ_a^b / C.
    """
    n = len(pts)
    if n < 2:
        return 0.0
    stack = np.zeros(n)
    ptr = 0
    damage = 0.0
    for i in range(n):
        stack[ptr] = pts[i]
        ptr += 1
        while ptr >= 3:
            X = abs(stack[ptr-1] - stack[ptr-2])
            Y = abs(stack[ptr-2] - stack[ptr-3])
            if X < Y:
                break
            stress_amp = Y * 0.5
            if ptr == 3:
                # Demi-cycle : on évacue le 1er point du résidu
                if stress_amp > 1e-10:
                    damage += 0.5 * (stress_amp ** b) / C
                stack[0] = stack[1]
                stack[1] = stack[2]
                ptr -= 1
            else:
                # Cycle complet : on retire les 2 points internes
                if stress_amp > 1e-10:
                    damage += (stress_amp ** b) / C
                stack[ptr-3] = stack[ptr-1]
                ptr -= 2
    # Résidu final : tous demi-cycles
    for i in range(ptr - 1):
        stress_amp = abs(stack[i+1] - stack[i]) * 0.5
        if stress_amp > 1e-10:
            damage += 0.5 * (stress_amp ** b) / C
    return damage


def _rainflow_damage(z_t, C, b, rearrange=False):
    """Orchestrateur ASTM E1049 / NF A03-406.

    `rearrange` (défaut False) :
      False → ASTM E1049 strict (RECOMMANDÉ). Identique à iamlikeme/rainflow
              au bit près. Résidus non fermés comptés en demi-cycles (0.5).
      True  → réarrangement préalable depuis le pic absolu (DSF / MIL-STD-810).
              Ferme artificiellement le cycle majeur — sur-estime le dommage
              de quelques % (effet d'autant plus marqué que b est grand).
    """
    if z_t.size < 2:
        return 0.0
    pts = _rainflow_extrema_numba(np.ascontiguousarray(z_t, dtype=np.float64))
    if pts.size < 2:
        return 0.0
    if rearrange:
        max_idx = int(np.argmax(np.abs(pts)))
        pts = np.concatenate((pts[max_idx:], pts[:max_idx], pts[max_idx:max_idx+1]))
    return _rainflow_stack_damage_numba(pts, C, b)


def calculer_sdf_rainflow(reponse, sdf_b, sdf_C):
    """Dommage rainflow GLOBAL sur z(t) — un seul scalaire pour tout le signal.

    Référence : [1] §C.2 (p. 71, Figure C.1) — σ(t) = K·z(t), K=1 forfaitaire.
                [4] ASTM E1049, [5] AFNOR A03-406.

    Convention amplitude : σ_a = range/2  →  D = Σ n_i · σ_a,i^b / C.

    La grandeur d'entrée DOIT être z(t) (déplacement relatif, en m).
    """
    return _rainflow_damage(reponse, sdf_C, sdf_b)


def calculer_sdf_per_bloc(reponse, fs, Tb, clusters, sdf_b, sdf_C,
                           taille_bloc=None):
    """Dommage rainflow PAR BLOC — variable D_p(j) de la norme [1].

    Référence : [1] §C.5 (n-échantillons {D_p}), §C.10 (synthèse stochastique).

    Découpe z(t) en n_blocs blocs de taille `taille_bloc` (calé sur le découpage
    de l'excitation pour synchronisation avec `clusters`), applique un rainflow
    interne à chaque bloc (pas de cycles inter-blocs — granularité élémentaire
    du dommage = T_b).

    Retour
    ------
    sdf_blocs : np.ndarray (n_blocs,) — D_p(j) pour j = 1..n_blocs.
                Sert ensuite à :
                  - Σ D_p,j  → SDF MBD-AnnexeC empirique (par classe puis total),
                  - ajustement Kappa4 sur la distribution des D_p,
                  - projection TCL → log-normale (cf. calculer_projection_*).
    """
    n_total = len(reponse)
    if taille_bloc is None or taille_bloc <= 0:
        cible       = round(Tb * fs)
        diviseurs   = [d for d in _trouver_diviseurs(n_total) if d > 0]
        taille_bloc = min(diviseurs, key=lambda d: abs(d - cible)) if diviseurs else cible
    n_blocs = min(n_total // taille_bloc, len(clusters))
    sdf_blocs = np.zeros(n_blocs)

    for j in range(n_blocs):
        bloc = reponse[j * taille_bloc: (j + 1) * taille_bloc]
        sdf_blocs[j] = _rainflow_damage(bloc, sdf_C, sdf_b)

    return sdf_blocs

# =============================================================================
# SECTION 9 — PROJECTION CDF LONGUE DURÉE (L-moments)
# =============================================================================

def calculer_projection_lmoments(params, maxima, n_blocs_classe, total_blocs, Tb, T_proj, alfa):
    """Projection TVE du SRE d'une classe à la durée T_proj.

    Référence : [1] §C.8-C.9 — coefficient d'extrapolation M(j), critère M > 100
                pour TVE. Synthèse stochastique [1] §C.10.

    Méthode (théorie des valeurs extrêmes) :
        Soit X = Z_ext = max d'un bloc, ajusté Kappa4. Sur M tirages i.i.d. :
            F_Z_sup(z) = F_Z_ext(z)^M
        On veut SRE_α tel que P(Z_sup ≤ SRE_α) = α, soit F(SRE_α)^M = α
        ⇒ SRE_α = F⁻¹(α^(1/M)) = PPF_Kappa4(α^(1/M))

        M = (n_i / total_blocs) · T_proj / Tb
        (n_i / total_blocs = Occ(j), occurrence relative de la classe j)

    Retour
    ------
    (sre_proj, M) ou (None, None) si fit invalide ou M ≤ 0.
    """
    if not params.get('success') or total_blocs <= 0 or Tb <= 0:
        return None, None
    Tpj = (n_blocs_classe / total_blocs) * T_proj
    M   = Tpj / Tb
    if M <= 0:
        return None, None
    try:
        p_base    = alfa ** (1.0 / M)
        sre_proj  = loi_ppf(params, p_base)
        return sre_proj, M
    except Exception:
        return None, None


def calculer_projection_sdf_tcl(sdf_blocs, clusters, n_clusters, total_blocs, Tb, T_proj, alfa):
    """Projection TCL (log-normale) du dommage cumulé sur T_proj — moments empiriques.

    Référence : [1] §C.10 — critère M > 50 pour TCL, Tableau C.2 (loi finale Gauss).
                Conversion Gauss → log-normale ici car D ≥ 0.

    Méthode :
      Pour chaque classe i :
          M_i = (n_i / total_blocs) · T_proj / Tb
          μ_y_i = M_i · μ_emp(D_blocs_i)
          σ²_y_i = M_i · σ²_emp(D_blocs_i)        (TCL : sommation i.i.d.)
      Total :
          μ_tot = Σ μ_y_i, var_tot = Σ σ²_y_i
      Conversion Gauss → log-normale (D ≥ 0) :
          CV² = var_tot / μ_tot²
          σ_log² = ln(1 + CV²)
          μ_log  = ln(μ_tot) - σ_log²/2
          SDF_α = LogNormal.ppf(α, s=σ_log, scale=exp(μ_log))

    Cette branche utilise les moments EMPIRIQUES des D_p — voir
    calculer_projection_dmg_kappa4() utilisant les
    moments de la Kappa4 ajustée sur les D_p.
    """
    mu_list, sigma_sq_list = [], []
    for i in range(n_clusters):
        blocs_i = sdf_blocs[clusters[:len(sdf_blocs)] == i]
        if len(blocs_i) == 0:
            continue
        n_i   = len(blocs_i)
        Tpj   = (n_i / total_blocs) * T_proj
        M     = Tpj / Tb
        if M <= 0:
            continue
        mu_i    = float(np.mean(blocs_i))
        sigma_i = float(np.std(blocs_i, ddof=1)) if n_i > 1 else 0.0
        mu_list.append(M * mu_i)
        sigma_sq_list.append(M * sigma_i * sigma_i)

    if not mu_list:
        return None

    mu_tot  = sum(mu_list)
    var_tot = sum(sigma_sq_list)

    if mu_tot <= 0.0:
        return 0.0
    if var_tot <= 0.0:
        return float(mu_tot)

    cv2          = var_tot / (mu_tot * mu_tot)
    sigma_log_sq = math.log(1.0 + cv2)
    sigma_log    = math.sqrt(sigma_log_sq)
    mu_log       = math.log(mu_tot) - 0.5 * sigma_log_sq
    res          = float(scipy_lognorm.ppf(alfa, s=sigma_log,
                                            scale=math.exp(mu_log)))
    return max(0.0, res)


def _kappa4_mean_var(params, n_grid=None):
    # n_grid=None → utilise KAPPA4_MEAN_VAR_GRID (carte EXPERT).
    if n_grid is None:
        n_grid = KAPPA4_MEAN_VAR_GRID
    """Estime (μ, σ²) d'une Kappa4 ajustée par échantillonnage de la PPF.

    On évite scipy_kappa4(...).stats(moments='mv') qui est instable pour h<0
    et certaines combinaisons (k, h). À la place, on intègre F⁻¹ sur une
    grille uniforme de probabilités → moments empiriques de la quantile.
    """
    if not params.get('success'):
        return None, None
    try:
        u  = np.linspace(1.0 / (n_grid + 1), n_grid / (n_grid + 1), n_grid)
        if _kappa4_use_exact(params['k'], params['h']):
            x = _kappa4_ppf_exact(u, params['xi'], params['alpha'],
                                  params['k'], params['h'])
        else:
            x = scipy_kappa4(h=params['h'], k=params['k'],
                             loc=params['xi'],
                             scale=params['alpha']).ppf(u)
        x  = x[np.isfinite(x)]
        if x.size < n_grid // 4:
            return None, None
        mu_k4  = float(np.mean(x))
        var_k4 = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
        return mu_k4, var_k4
    except Exception:
        return None, None


def calculer_projection_dmg_kappa4(params_dmg_list, d_blocs_classes,
                                    n_clusters, total_blocs, Tb, T_proj, alfa):
    """Projection lognormale du dommage cumulé via les moments Kappa4 ajustés
    sur les D_bloc par classe (NF X50 144-3 Annexe C).

    Pour chaque classe i :
      M_i = (n_i / total_blocs) * T_proj / Tb
      μ_K4_i, σ²_K4_i estimés par intégration numérique de la PPF Kappa4.
      Contribution : μ_tot += M_i·μ_K4_i ; var_tot += M_i·σ²_K4_i (TCL).
    Si le fit Kappa4 a échoué pour une classe, on retombe sur les moments
    empiriques des D_bloc de cette classe (cohérent avec calculer_projection_sdf_tcl).
    """
    if total_blocs <= 0 or Tb <= 0 or T_proj is None or T_proj <= 0:
        return None

    mu_list, sigma_sq_list = [], []
    for i in range(n_clusters):
        d_blocs_i = (np.asarray(d_blocs_classes[i], dtype=float)
                     if i < len(d_blocs_classes) else np.array([]))
        n_i = len(d_blocs_i)
        if n_i == 0:
            continue
        Tpj = (n_i / total_blocs) * T_proj
        M   = Tpj / Tb
        if M <= 0:
            continue

        params_i = (params_dmg_list[i] if i < len(params_dmg_list)
                    else {'success': False})
        mu_i, var_i = _loi_mean_var(params_i)
        if mu_i is None or not np.isfinite(mu_i):
            mu_i  = float(np.mean(d_blocs_i))
            var_i = float(np.var(d_blocs_i, ddof=1)) if n_i > 1 else 0.0

        if not np.isfinite(mu_i):
            continue
        if var_i is None or not np.isfinite(var_i) or var_i < 0.0:
            var_i = 0.0

        mu_list.append(M * mu_i)
        sigma_sq_list.append(M * var_i)

    if not mu_list:
        return None

    mu_tot  = sum(mu_list)
    var_tot = sum(sigma_sq_list)

    if mu_tot <= 0.0:
        return 0.0
    if var_tot <= 0.0:
        return float(mu_tot)

    cv2          = var_tot / (mu_tot * mu_tot)
    sigma_log_sq = math.log(1.0 + cv2)
    sigma_log    = math.sqrt(sigma_log_sq)
    mu_log       = math.log(mu_tot) - 0.5 * sigma_log_sq
    res          = float(scipy_lognorm.ppf(alfa, s=sigma_log,
                                            scale=math.exp(mu_log)))
    return max(0.0, res)

# =============================================================================
# SECTION 9bis — QUALITY GATE IID (indépendance statistique des blocs par f0)
# =============================================================================
# Brique de validation post-traitement (cahier des charges). Aucune dépendance
# au domaine temporel ni au reste du pipeline : couplage lâche, opère
# uniquement sur le vecteur (1 f₀) ou la matrice [N blocs, M f₀] des valeurs
# extrêmes / dommages par bloc. Entièrement vectorisé numpy/scipy.

def _iid_ranks(X2):
    """Rangs (ex æquo moyennés) le long de l'axe 0 (blocs). X2 : (N, M)."""
    try:
        return sp_rankdata(X2, axis=0)
    except TypeError:                       # scipy < 1.10 : pas d'argument axis
        return np.column_stack([sp_rankdata(X2[:, j])
                                for j in range(X2.shape[1])])


def _iid_spearman_lag1(X2):
    """ρ de Spearman lag-1 par colonne = Pearson sur les rangs décalés.

    Travailler sur les RANGS plutôt que sur les valeurs brutes évite le biais
    des extrêmes de la Kappa-4 (cahier des charges §3, méthode 1).
    X2 : (N, M) → vecteur (M,). NaN si N < 3 ou variance de rang nulle."""
    N = X2.shape[0]
    out = np.full(X2.shape[1], np.nan)
    if N < 3:
        return out
    r  = _iid_ranks(X2)
    a  = r[:-1, :]
    b  = r[1:, :]
    da = a - a.mean(axis=0)
    db = b - b.mean(axis=0)
    num = (da * db).sum(axis=0)
    den = np.sqrt((da * da).sum(axis=0) * (db * db).sum(axis=0))
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _iid_runs_pvalue(X2):
    """p-value bilatérale du test des suites de Wald-Wolfowitz par colonne.

    Binarisation vs médiane locale (1 si xᵢ > médiane, 0 sinon), comptage des
    runs, statistique Z par approximation normale (cahier §3, méthode 2).
    X2 : (N, M) → vecteur (M,). NaN si dégénéré (n1=0, n2=0, var ≤ 0)."""
    N = X2.shape[0]
    out = np.full(X2.shape[1], np.nan)
    if N < 3:
        return out
    med  = np.median(X2, axis=0)
    bin_ = (X2 > med).astype(np.int8)             # 1 si > médiane, 0 sinon
    n1   = bin_.sum(axis=0).astype(float)         # nb au-dessus
    n2   = float(N) - n1                          # nb au niveau / en-dessous
    runs = 1.0 + (bin_[1:, :] != bin_[:-1, :]).sum(axis=0)
    prod = n1 * n2
    with np.errstate(divide='ignore', invalid='ignore'):
        mu  = 2.0 * prod / N + 1.0
        var = 2.0 * prod * (2.0 * prod - N) / (N * N * (N - 1.0))
        z   = (runs - mu) / np.sqrt(var)
    valid = (n1 > 0) & (n2 > 0) & np.isfinite(var) & (var > 0)
    out[valid] = 2.0 * sp_norm.sf(np.abs(z[valid]))
    return out


def quality_gate_iid(X, rho_max=None, pval_min=None, min_n=None):
    """Quality Gate IID — cahier des charges §3-4.

    Vérifie l'hypothèse d'indépendance des blocs temporels pour une (entrée
    1D) ou plusieurs (entrée 2D [N, M]) fréquences f₀, sans retour au domaine
    temporel :
      - Méthode 1 : autocorrélation lag-1 de Spearman sur les rangs ;
      - Méthode 2 : test des suites de Wald-Wolfowitz (médiane locale).

    Couplage lâche : ne dépend QUE de la matrice fournie.

    Retour : dict de scalaires (entrée 1D) ou de vecteurs (M,) (entrée 2D) :
      'rho'    autocorrélation lag-1 sur rangs,
      'pvalue' p-value du test des suites,
      'n'      taille d'échantillon,
      'tested' True si n ≥ min_n et métriques finies,
      'fail'   True si testé ET (|rho| > rho_max OU pvalue < pval_min).
    """
    rho_max  = IID_RHO_MAX    if rho_max  is None else rho_max
    pval_min = IID_PVALUE_MIN if pval_min is None else pval_min
    min_n    = IID_MIN_N      if min_n    is None else min_n

    arr = np.asarray(X, dtype=float)
    scalar = (arr.ndim == 1)
    if scalar:
        arr = arr[:, None]

    n      = np.isfinite(arr).sum(axis=0).astype(int)
    rho    = _iid_spearman_lag1(arr)
    pval   = _iid_runs_pvalue(arr)
    tested = (n >= min_n) & np.isfinite(rho) & np.isfinite(pval)
    fail   = np.zeros(arr.shape[1], dtype=bool)
    fail[tested] = ((np.abs(rho[tested]) > rho_max)
                    | (pval[tested] < pval_min))

    if scalar:
        return {'rho': float(rho[0]), 'pvalue': float(pval[0]),
                'n': int(n[0]), 'tested': bool(tested[0]),
                'fail': bool(fail[0])}
    return {'rho': rho, 'pvalue': pval, 'n': n,
            'tested': tested, 'fail': fail}


def _iid_verdict(frac_fail):
    """Statut global selon la fraction de f₀ hors-tolérance (cahier §5).
    🟢 GO / 🟡 WARNING (bande isolée) / 🔴 NO-GO (généralisé)."""
    if not np.isfinite(frac_fail) or frac_fail <= IID_FAIL_FRAC_MAX:
        return 'GO'
    if frac_fail < IID_NOGO_FRAC:
        return 'WARNING'
    return 'NO-GO'


def agreger_quality_gate(all_results, f0_spectrum, Tb, sdf_actif):
    """Agrège les diagnostics IID par f₀ en spectres [M] + verdict global.

    Construit, à partir des champs ``result['iid']`` posés dans ``traiter_f0``,
    les spectres ρ(f₀) et p-value(f₀) pour les branches SRE et SDF, calcule la
    fraction de f₀ hors-tolérance et en déduit le statut 🟢/🟡/🔴. Le run
    n'est jamais interrompu (décision : avertir + taguer + continuer)."""
    res_ok = [r for r in all_results if r.get('success') and 'iid' in r]
    f0s    = np.array([r['f0'] for r in res_ok], dtype=float)
    M      = len(res_ok)
    diag = {
        'enabled': True, 'f0': f0s,
        'status': 'GO', 'frac_fail': 0.0,
        'sre': {'rho': np.array([]), 'pvalue': np.array([]),
                'fail': np.array([], dtype=bool)},
        'sdf': {'rho': np.array([]), 'pvalue': np.array([]),
                'fail': np.array([], dtype=bool)},
        'per_f0_confidence': {},
    }
    if M == 0:
        diag['status'] = 'GO'
        return diag

    def _spec(branch):
        rho  = np.array([r['iid'].get(branch, {}).get('rho', np.nan)
                         for r in res_ok], dtype=float)
        pval = np.array([r['iid'].get(branch, {}).get('pvalue', np.nan)
                         for r in res_ok], dtype=float)
        fail = np.array([bool(r['iid'].get(branch, {}).get('fail', False))
                         for r in res_ok], dtype=bool)
        return {'rho': rho, 'pvalue': pval, 'fail': fail}

    diag['sre'] = _spec('sre')
    diag['sdf'] = _spec('sdf') if sdf_actif else diag['sdf']

    fail_any = diag['sre']['fail'].copy()
    if sdf_actif and diag['sdf']['fail'].size == M:
        fail_any = fail_any | diag['sdf']['fail']

    frac_fail = float(fail_any.sum()) / float(M) if M else 0.0
    status    = _iid_verdict(frac_fail)
    diag['frac_fail'] = frac_fail
    diag['status']    = status

    # f₀ fautives → confiance réduite (utilisé en hover du rapport principal).
    for r, bad in zip(res_ok, fail_any):
        diag['per_f0_confidence'][float(r['f0'])] = ('reduced' if bad
                                                     else 'ok')

    n_fail = int(fail_any.sum())
    if status == 'GO':
        logger.info("Quality Gate IID : 🟢 GO — indépendance validée "
                    "(%d/%d f₀ hors-tolérance, %.1f%%).",
                    n_fail, M, 100.0 * frac_fail)
    elif status == 'WARNING':
        bad_f0 = f0s[fail_any]
        plage  = (f"{bad_f0.min():.1f}–{bad_f0.max():.1f} Hz"
                  if bad_f0.size else "n/a")
        logger.warning("Quality Gate IID : 🟡 WARNING — indépendance perdue "
                        "sur une bande isolée (%d/%d f₀, %.1f%%, %s). "
                        "Inférence Kappa-4 conservée mais ces f₀ sont taguées "
                        "en confiance réduite.",
                        n_fail, M, 100.0 * frac_fail, plage)
    else:  # NO-GO
        tb_reco = 2.0 * Tb
        logger.warning("Quality Gate IID : 🔴 NO-GO — dépendance temporelle "
                        "généralisée (%d/%d f₀ hors-tolérance, %.1f%%). "
                        "ACTION CORRECTIVE : augmenter la durée de bloc "
                        "(ex. TB=%g s au lieu de %g s) pour englober la "
                        "traîne de la réponse dynamique, puis relancer "
                        "l'extraction SRE/SDF. Résultats Kappa-4 du run "
                        "courant à considérer comme NON FIABLES.",
                        n_fail, M, 100.0 * frac_fail, tb_reco, Tb)
    return diag

# =============================================================================
# SECTION 10 — TRAITEMENT D'UNE FRÉQUENCE f0
# =============================================================================

def traiter_f0(f0, excitation, clusters, n_clusters, Tb, Q, fs,
               prob_cible, option_cunnane, cunnane_a,
               sdf_b, sdf_C, sdf_enabled,
               min_points):
    """Pipeline complet pour une fréquence f₀ — fonction d'entrée des workers.

    Référence : [1] NF X50-144-3 §C — chaque appel reproduit pour une f₀ donnée
                les étapes 4-7 de la Figure C.3 (workflow Annexe C).

    Étapes :
      A. Réponse SDOF z(t) — FOH récursif Smallwood [8].
      B. Pseudo-accélération (2πf₀)²·z → calcul SRC + base maxima Z_ext.
      C. SDF Rainflow GLOBAL sur z(t) (un seul scalaire).
      D. Maxima par bloc T_b → {Z_ext(i)}, i = 1..n_blocs.
      E. SDF per-bloc → {D_p(i)} (rainflow interne à chaque bloc, [1] §C.5).
      F. Pour chaque classe i :
            - Branche SRE : Kappa4 sur Z_ext  → PPF Cunnane → SRE(f₀, classe)
            - Branche SDF : Kappa4 sur D_p    → moments → projection log-normale
      G. Synthèse :
            SRE(f₀)  = max_i SRE(f₀, classe_i)
            SDF(f₀) = Σᵢ Σⱼ D_p(i, j)        (= sdf_mbd_empirique)

    Retour
    ------
    dict avec clés :
        'f0', 'success', 'src', 'sre',
        'sdf_temporel', 'sdf_mbd_empirique', 'sdf_kappa4_empirical' (alias),
        'maxima_classes', 'params_list', 'valeurs_ppf_list',
        'rmse_list', 'msdi_list', 'prob_ppf_list',
        'd_blocs_classes', 'params_dmg_list', 'valeurs_ppf_dmg_list',
        'rmse_dmg_list', 'msdi_dmg_list',
        'sdf_per_bloc', 'clusters_trunc'      (si projection activée)
    """
    result = {'f0': f0, 'success': False}

    try:
        # NF X50-144-3 §C.2 : z(t) = déplacement relatif (m), σ(t) = K·z(t).
        # SRE/SRC s'expriment en pseudo-accélération (m/s²) = (2πf₀)²·z.
        # SDF (Basquin/Miner) s'applique sur σ ≈ K·z avec K=1 par défaut → rainflow sur z.
        reponse = reponse_sdof(excitation, f0=f0, Q=Q, fs=fs)
        contrainte = reponse * (4.0 * math.pi**2 * f0**2)
        result['src'] = float(np.max(np.abs(contrainte)))

        if sdf_enabled and HAS_RAINFLOW:
            result['sdf_temporel'] = calculer_sdf_rainflow(reponse, sdf_b, sdf_C)

        _, maxima_rep, _, n_blocs, taille_bloc_rep = extraire_caracteristiques(
            contrainte, fs, Tb_initial=Tb, feature_flags={}, min_ech=MIN_ECH_PAR_BLOC)

        min_len     = min(len(maxima_rep), len(clusters))
        maxima_rep  = maxima_rep[:min_len]
        cl_trunc    = clusters[:min_len]

        # SDF per-bloc calculé une fois pour toutes (avant la boucle classe).
        # Granularité élémentaire du dommage = bloc Tb (rainflow interne au bloc, sur z).
        sdf_blocs = None
        if sdf_enabled and HAS_RAINFLOW:
            sdf_blocs_full = calculer_sdf_per_bloc(reponse, fs, Tb, cl_trunc,
                                                    sdf_b, sdf_C,
                                                    taille_bloc=taille_bloc_rep)
            if sdf_blocs_full is not None:
                sdf_blocs = sdf_blocs_full[:min_len]

        # --- Quality Gate IID (SECTION 9bis) -----------------------------
        # Calculé ICI : maxima_rep / sdf_blocs / cl_trunc sont dans l'ordre
        # temporel des blocs (indispensable pour lag-1 / runs) et l'inférence
        # Kappa-4 par classe (boucle ci-dessous) n'a pas encore consommé les
        # données. Le masque booléen cl_trunc==i préserve l'ordre intra-classe.
        if IID_GATE_ENABLED:
            iid = {'sre': quality_gate_iid(maxima_rep),
                   'per_class': []}
            if sdf_blocs is not None and len(sdf_blocs) == min_len:
                iid['sdf'] = quality_gate_iid(sdf_blocs)
            for i in range(n_clusters):
                m_i = maxima_rep[cl_trunc == i]
                pc  = {'classe': i, 'n': int(m_i.size),
                       'sre': quality_gate_iid(m_i)}
                if sdf_blocs is not None and len(sdf_blocs) == min_len:
                    pc['sdf'] = quality_gate_iid(sdf_blocs[cl_trunc == i])
                iid['per_class'].append(pc)
            result['iid'] = iid

        result['final_n_clusters']  = n_clusters
        result['maxima_classes']    = []
        result['params_list']       = []
        result['valeurs_ppf_list']  = []
        result['rmse_list']         = []
        result['msdi_list']         = []
        result['prob_ppf_list']     = []
        # branche projection GEV 3 domaines (option METHODE_PROJECTION)
        result['params_gev_list']   = []
        # branche dommage (Kappa4 sur D_bloc)
        result['d_blocs_classes']      = []
        result['params_dmg_list']      = []
        result['valeurs_ppf_dmg_list'] = []
        result['rmse_dmg_list']        = []
        result['msdi_dmg_list']        = []
        # Diagnostic SRE Kappa4 par classe — rempli en parallèle de params_list /
        # valeurs_ppf_list pour expliquer les trous éventuels du spectre.
        result['sre_class_status']     = []
        sdf_mbd_empirique = 0.0

        for i in range(n_clusters):
            maxima_i = maxima_rep[cl_trunc == i]
            result['maxima_classes'].append(maxima_i.tolist())
            N_i = len(maxima_i)

            # --- Branche SRE : Kappa4 sur maxima (inchangée) -----------------
            params_i = {'success': False, 't3': np.nan, 't4': np.nan,
                        'fail_reason': 'not_run'}
            rmse_i = msdi_i = np.nan
            ppf_i  = None
            prob_eff = prob_cible
            class_status = 'ok'

            if N_i < min_points:
                class_status = f'n_lt_min:{N_i}'
            else:
                params_i = ajuster_loi(maxima_i)
                if params_i['success']:
                    rmse_i, msdi_i = loi_rmse_msdi(maxima_i, params_i)
                    if option_cunnane:
                        denom = N_i + 1.0 - 2.0 * cunnane_a
                        if abs(denom) > 1e-9:
                            prob_eff = float(np.clip((N_i - cunnane_a) / denom,
                                                      PROBA_CLIP_EPS, 1.0 - PROBA_CLIP_EPS))
                    ppf_i = loi_ppf(params_i, prob_eff)
                    if ppf_i is None or not np.isfinite(ppf_i):
                        class_status = 'ppf_nan'
                else:
                    class_status = f"fit:{params_i.get('fail_reason', 'unknown')}"

            result['params_list'].append(params_i)
            result['valeurs_ppf_list'].append(ppf_i)
            result['rmse_list'].append(rmse_i)
            result['msdi_list'].append(msdi_i)
            result['prob_ppf_list'].append(prob_eff)
            result['sre_class_status'].append(class_status)

            # --- Option 'gev_domaines' : ré-ajustement GEV (h=0) sur les
            # mêmes maxima, utilisé UNIQUEMENT pour la projection longue
            # durée ([2] §4.1). Le SRE à la durée du signal reste celui de
            # LOI_AJUSTEMENT ci-dessus.
            params_gev_i = None
            if (METHODE_PROJECTION == 'gev_domaines' and ENABLE_PROJECTION
                    and N_i >= min_points):
                params_gev_i = ajuster_gev_lmoments(maxima_i)
            result['params_gev_list'].append(params_gev_i)

            # --- Branche dommage : SDF MBD-AnnexeC + Kappa4 sur D_bloc --
            d_blocs_i = (sdf_blocs[cl_trunc == i] if sdf_blocs is not None
                         else np.array([]))
            result['d_blocs_classes'].append(d_blocs_i.tolist())

            params_dmg_i = {'success': False, 't3': np.nan, 't4': np.nan}
            rmse_dmg_i = msdi_dmg_i = np.nan
            ppf_dmg_i  = None

            if sdf_enabled and len(d_blocs_i) > 0:
                # Dommage MBD-AnnexeC = Σ D_bloc
                # /sdf_C déjà appliqué dans calculer_sdf_per_bloc.
                sdf_mbd_empirique += float(np.sum(d_blocs_i))

                if len(d_blocs_i) >= min_points:
                    # Filtre les D_bloc strictement > 0 pour stabilité Kappa4.
                    d_pos = d_blocs_i[np.isfinite(d_blocs_i) & (d_blocs_i > 0)]
                    if len(d_pos) >= min_points:
                        params_dmg_i = ajuster_loi(d_pos)
                        if params_dmg_i['success']:
                            rmse_dmg_i, msdi_dmg_i = loi_rmse_msdi(d_pos, params_dmg_i)
                            prob_eff_dmg = prob_cible
                            if option_cunnane:
                                Nd = len(d_pos)
                                denom_d = Nd + 1.0 - 2.0 * cunnane_a
                                if abs(denom_d) > 1e-9:
                                    prob_eff_dmg = float(np.clip(
                                        (Nd - cunnane_a) / denom_d,
                                        PROBA_CLIP_EPS, 1.0 - PROBA_CLIP_EPS))
                            ppf_dmg_i = loi_ppf(params_dmg_i, prob_eff_dmg)

            result['params_dmg_list'].append(params_dmg_i)
            result['valeurs_ppf_dmg_list'].append(ppf_dmg_i)
            result['rmse_dmg_list'].append(rmse_dmg_i)
            result['msdi_dmg_list'].append(msdi_dmg_i)

        # Alias rétrocompatible : nom historique conservé pour les exports/HTML.
        result['sdf_kappa4_empirical'] = sdf_mbd_empirique
        result['sdf_mbd_empirique']    = sdf_mbd_empirique

        ppf_valides = [p for p in result['valeurs_ppf_list']
                       if p is not None and np.isfinite(p)]
        result['sre'] = max(ppf_valides) if ppf_valides else None

        # Synthèse diagnostique : explique pourquoi result['sre'] vaut None
        # (ou confirme 'ok'). Groupe les statuts par catégorie pour un affichage compact.
        statuses    = result['sre_class_status']
        n_classes   = len(statuses)
        n_ok        = sum(1 for s in statuses if s == 'ok')
        n_fail      = n_classes - n_ok
        from collections import Counter
        # Catégorie = préfixe avant ':' (n_lt_min, fit, ppf_nan, ok)
        cat_counts  = Counter(s.split(':', 1)[0] for s in statuses if s != 'ok')
        # Détail par catégorie : liste des suffixes (N_i ou fail_reason)
        cat_details = {}
        for s in statuses:
            if s == 'ok':
                continue
            cat, _, suf = s.partition(':')
            cat_details.setdefault(cat, []).append(suf)
        if n_fail == 0:
            reason = 'ok'
        else:
            parts = []
            for cat, cnt in cat_counts.most_common():
                sufs = cat_details.get(cat, [])
                if cat == 'n_lt_min':
                    parts.append(f"{cnt}/{n_classes} classes N<min ({','.join(sufs)})")
                elif cat == 'fit':
                    # regroupe les fail_reason identiques
                    sub = Counter(sufs).most_common()
                    sub_str = ','.join(f"{r}×{c}" if c > 1 else r for r, c in sub)
                    parts.append(f"{cnt}/{n_classes} fit:{sub_str}")
                else:
                    parts.append(f"{cnt}/{n_classes} {cat}")
            reason = '; '.join(parts)
        result['sre_diag'] = {
            'reason':    reason,
            'per_class': statuses,
            'n_ok':      n_ok,
            'n_fail':    n_fail,
        }

        # Conserve sdf_per_bloc / clusters_trunc pour la projection TCL empirique.
        if sdf_enabled and sdf_blocs is not None and ENABLE_PROJECTION:
            result['sdf_per_bloc']   = sdf_blocs
            result['clusters_trunc'] = cl_trunc

        result['success'] = True

    except Exception as e:
        result['error_message'] = str(e)

    return result

# ---------------------------------------------------------------------------
# Workers multiprocessing (pickle-safe, top-level)
# ---------------------------------------------------------------------------

_MP_SIGNAL   = None
_MP_CLUSTERS = None
_MP_KWARGS   = None
_MP_SHM      = None


def _mp_worker_init(signal_ref, clusters, kwargs):
    """Initializer : attache shared_memory si tuple, sinon utilise ndarray pickle."""
    global _MP_SIGNAL, _MP_CLUSTERS, _MP_KWARGS, _MP_SHM
    if isinstance(signal_ref, tuple) and len(signal_ref) == 3:
        from multiprocessing import shared_memory as _shm
        name, shape, dtype_str = signal_ref
        _MP_SHM    = _shm.SharedMemory(name=name)
        _MP_SIGNAL = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_MP_SHM.buf)
    else:
        _MP_SIGNAL = signal_ref
    _MP_CLUSTERS = clusters
    _MP_KWARGS   = kwargs


def _mp_worker_traiter(f0):
    """Traite une fréquence dans le worker. Renvoie (result, timings_snapshot)."""
    global KAPPA4_TIMINGS
    prev = dict(KAPPA4_TIMINGS)
    for k in KAPPA4_TIMINGS:
        KAPPA4_TIMINGS[k] = 0 if isinstance(KAPPA4_TIMINGS[k], int) else 0.0
    try:
        res = traiter_f0(
            f0=f0,
            excitation=_MP_SIGNAL,
            clusters=_MP_CLUSTERS,
            **_MP_KWARGS,
        )
    finally:
        timings = dict(KAPPA4_TIMINGS)
        for k, v in prev.items():
            KAPPA4_TIMINGS[k] = v + timings[k]
    return res, timings

# =============================================================================
# SECTION 11 — EXPORTS CSV
# =============================================================================

def _fmt_compact(v):
    """Formate un nombre pour suffixes courts : 1.33→'1p33', 8.0→'8', 0.9→'0p9'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return str(v)
    if f == int(f):
        return str(int(f))
    return f"{f:g}".replace('.', 'p')


def _fmt_duration_compact(t_s):
    """Suffixe court pour une durée (s) : 36e6→'36Ms', 3600→'3p6ks', 120→'120s'."""
    try:
        t = float(t_s)
    except (TypeError, ValueError):
        return str(t_s)
    if not np.isfinite(t):
        return str(t_s)
    if t >= 1e6:
        return f"{t/1e6:g}Ms".replace('.', 'p')
    if t >= 1e3:
        return f"{t/1e3:g}ks".replace('.', 'p')
    return f"{t:g}s".replace('.', 'p')


def build_run_meta(fs, duree_mesure, n_k_final, num_f0, ts):
    """Dictionnaire des paramètres du run, sérialisé en JSON sidecar et utilisé
    pour suffixer les noms de colonnes CSV."""
    return {
        'version':            'v3.4',
        'date_run':           ts,
        'fichier_source':     CSV_FILEPATH,
        'fs_Hz':              float(fs),
        'duree_mesure_s':     float(duree_mesure),
        'Q':                  Q,
        'Tb_s':               TB,
        'f0_min_Hz':          F0_MIN,
        'f0_max_Hz':          F0_MAX,
        'delta_f0_Hz':        DELTA_F0,
        'num_f0':             int(num_f0),
        'K_means':            int(n_k_final),
        'P_cible':            PROBABILITE_CIBLE,
        'Cunnane_active':     bool(OPTION_CUNNANE),
        'Cunnane_a':          CUNNANE_A,
        'alpha_SRX_low':      ALPHA_SRX_LOW,
        'alpha_SRX_high':     ALPHA_SRX_HIGH,
        'SDF_enabled':        bool(SDF_ENABLED),
        'SDF_b_Basquin':      SDF_B,
        'SDF_C':              SDF_C,
        'projection_enabled': bool(ENABLE_PROJECTION),
        'T_projection_s':     DUREE_PROJECTION,
        'alfa_projection':    ALFA_PROJECTION,
        'methode_projection': METHODE_PROJECTION,
        'loi_ajustement':     LOI_AJUSTEMENT,
        'methode_Kappa4':     KAPPA4_METHOD,
    }


def _build_col_suffixes(meta):
    """Suffixes courts à concaténer dans les noms de colonnes selon les
    paramètres dont chaque grandeur dépend physiquement."""
    Q_s     = f"Q{meta['Q']}"
    Tb_s    = f"Tb{_fmt_compact(meta['Tb_s'])}s"
    Tmes_s  = f"Tmes{_fmt_compact(meta['duree_mesure_s'])}s"
    P_s     = f"P{_fmt_compact(meta['P_cible'])}"
    aL_s    = f"aL{_fmt_compact(meta['alpha_SRX_low'])}"
    aH_s    = f"aH{_fmt_compact(meta['alpha_SRX_high'])}"
    b_s     = f"b{_fmt_compact(meta['SDF_b_Basquin'])}"
    Tproj_s = f"T{_fmt_duration_compact(meta['T_projection_s'])}"
    a_s     = f"a{_fmt_compact(meta['alfa_projection'])}"
    return {
        'sdof':     f"{Q_s}_{Tb_s}",
        'sre':      f"{Q_s}_{Tb_s}_{P_s}_{Tmes_s}",
        'sre_dsp':  f"{Q_s}_{Tb_s}_{Tmes_s}",
        'srx_low':  f"{Q_s}_{Tb_s}_{aL_s}_{Tmes_s}",
        'srx_high': f"{Q_s}_{Tb_s}_{aH_s}_{Tmes_s}",
        'sdf':      f"{b_s}_{Q_s}_{Tb_s}_{Tmes_s}",
        'fit':      f"{b_s}_{Q_s}_{Tb_s}",
        'proj_sre': f"{Q_s}_{Tb_s}_{P_s}_{Tproj_s}_{a_s}",
        'proj_sre_dsp':  f"{Q_s}_{Tb_s}_{Tproj_s}",
        'proj_srx_low':  f"{Q_s}_{Tb_s}_{aL_s}_{Tproj_s}",
        'proj_srx_high': f"{Q_s}_{Tb_s}_{aH_s}_{Tproj_s}",
        'proj_M':   f"{Tproj_s}_{a_s}_{Tb_s}",
        'proj_sdf': f"{b_s}_{Q_s}_{Tb_s}_{Tproj_s}_{a_s}",
        'proj_dsp': f"{b_s}_{Q_s}_{Tb_s}_{Tproj_s}",
        'iid':      f"{Q_s}_{Tb_s}",
    }


def exporter_run_meta_json(filepath, meta):
    """Écrit le sidecar JSON des paramètres du run."""
    import json
    with open(filepath, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Sidecar paramètres : %s", filepath)


def exporter_csv_sre(filepath, results, sre_dsp, srx_low, srx_high,
                     f0_grid, meta):
    """CSV : SRC, SRE Kappa4, SRE DSP, SRX(α_low), SRX(α_high) — 1 ligne / f₀.

    Référence colonnes DSP : PR NORMDEF 0101 §5.4.2 (SRE) et §5.4.3 (SRX,
    formule [5.2]). Les paramètres de calcul (Q, Tb, P_cible, durée mesure,
    α_low, α_high) sont suffixés dans chaque nom de colonne pour pouvoir
    empiler des CSV de runs avec paramètres différents."""
    import pandas as pd
    sx = _build_col_suffixes(meta)
    sre_d = dict(zip(f0_grid, sre_dsp))
    xl_d  = dict(zip(f0_grid, srx_low))
    xh_d  = dict(zip(f0_grid, srx_high))
    def _iid_cols(r, branch):
        d = r.get('iid', {}).get(branch)
        if not isinstance(d, dict) or not d.get('tested'):
            return np.nan, np.nan, 'n_a'
        return (d.get('rho', np.nan), d.get('pvalue', np.nan),
                'FAIL' if d.get('fail') else 'ok')

    rows = []
    for r in sorted(results, key=lambda x: x['f0']):
        if not r['success']:
            continue
        rho, pval, stat = _iid_cols(r, 'sre')
        rows.append({
            'Frequence_Hz':                       r['f0'],
            f"SRC_{sx['sdof']}":                  r.get('src', np.nan),
            f"SRE_Kappa4_{sx['sre']}":            r.get('sre') if r.get('sre') is not None else np.nan,
            f"SRE_DSP_NormDef_{sx['sre_dsp']}":   sre_d.get(r['f0'], np.nan),
            f"SRX_alpha_low_{sx['srx_low']}":     xl_d.get(r['f0'],  np.nan),
            f"SRX_alpha_high_{sx['srx_high']}":   xh_d.get(r['f0'],  np.nan),
            f"IID_rho_lag1_{sx['iid']}":          rho,
            f"IID_pvalue_runs_{sx['iid']}":       pval,
            f"IID_status_{sx['iid']}":            stat,
        })
    pd.DataFrame(rows).to_csv(filepath, index=False, sep=';', decimal=',')
    logger.info("CSV SRE : %s", filepath)


def exporter_csv_sdf(filepath, results, sdf_spectral, f0_grid, meta):
    """CSV : SDF temporel rainflow, MBD-AnnexeC (Σ D_bloc), spectral Bendat,
    + RMSE/MSDI moyen du fit Kappa4 sur D_bloc. Paramètres (b Basquin, Q, Tb,
    durée mesure) suffixés dans les noms de colonnes."""
    import pandas as pd
    sx = _build_col_suffixes(meta)
    sp_d = dict(zip(f0_grid, sdf_spectral)) if sdf_spectral is not None else {}

    def _safe_mean(vals):
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                          dtype=float)
        return float(np.mean(arr)) if arr.size else np.nan

    rows = []
    for r in sorted(results, key=lambda x: x['f0']):
        if not r['success']:
            continue
        d_iid = r.get('iid', {}).get('sdf')
        if isinstance(d_iid, dict) and d_iid.get('tested'):
            iid_rho, iid_pval = d_iid.get('rho', np.nan), d_iid.get('pvalue', np.nan)
            iid_stat = 'FAIL' if d_iid.get('fail') else 'ok'
        else:
            iid_rho, iid_pval, iid_stat = np.nan, np.nan, 'n_a'
        rows.append({
            'Frequence_Hz':                              r['f0'],
            f"SDF_Temporel_Rainflow_{sx['sdf']}":        r.get('sdf_temporel', np.nan),
            f"SDF_MBD_AnnexeC_{sx['sdf']}":              r.get('sdf_mbd_empirique',
                                                              r.get('sdf_kappa4_empirical', np.nan)),
            f"SDF_Spectral_Bendat_{sx['sdf']}":          sp_d.get(r['f0'], np.nan),
            f"RMSE_K4_Dmg_moyen_{sx['fit']}":            _safe_mean(r.get('rmse_dmg_list', [])),
            f"MSDI_K4_Dmg_moyen_{sx['fit']}":            _safe_mean(r.get('msdi_dmg_list', [])),
            f"IID_rho_lag1_{sx['iid']}":                 iid_rho,
            f"IID_pvalue_runs_{sx['iid']}":              iid_pval,
            f"IID_status_{sx['iid']}":                   iid_stat,
        })
    pd.DataFrame(rows).to_csv(filepath, index=False, sep=';', decimal=',')
    logger.info("CSV SDF : %s", filepath)


def exporter_csv_projection(filepath, proj_results, meta,
                            sre_dsp_proj=None, srx_low_proj=None,
                            srx_high_proj=None, f0_grid=None):
    """CSV : SRE projeté, M, SDF projeté (TCL empirique, K4-LogN, DSP).

    Inclut également, à la durée T = DUREE_PROJECTION :
      - SRE DSP NORMDEF §5.4.2 projeté ;
      - SRX α_low / α_high NORMDEF §5.4.3 eq. [5.2] projetés.
    Ces grandeurs DSP ne dépendent que de n₀⁺·T_proj (z_eff stationnaire
    inchangé) ; elles sont alignées par fréquence via `f0_grid`.
    Paramètres de projection (T_proj, α, b, Q, Tb) suffixés dans les noms
    de colonnes."""
    import pandas as pd
    sx = _build_col_suffixes(meta)
    sre_dp_d = (dict(zip(f0_grid, sre_dsp_proj))
                if sre_dsp_proj is not None and f0_grid is not None else {})
    xl_dp_d  = (dict(zip(f0_grid, srx_low_proj))
                if srx_low_proj is not None and f0_grid is not None else {})
    xh_dp_d  = (dict(zip(f0_grid, srx_high_proj))
                if srx_high_proj is not None and f0_grid is not None else {})
    # Colonne 'GEV_domaine' présente uniquement en méthode 'gev_domaines'
    # (domaine d'attraction retenu pour la classe dimensionnante, [2] §4.1).
    has_gev = any(pr.get('gev_domaine') for pr in proj_results)
    def _row(pr):
        row = {
            'Frequence_Hz':                                 pr['f0'],
            f"SRE_Projection_{sx['proj_sre']}":             (pr.get('sre_proj_max')
                                                              if pr.get('sre_proj_max') is not None
                                                              else np.nan),
            f"SRE_DSP_NormDef_Proj_{sx['proj_sre_dsp']}":   sre_dp_d.get(pr['f0'], np.nan),
            f"SRX_alpha_low_Proj_{sx['proj_srx_low']}":     xl_dp_d.get(pr['f0'],  np.nan),
            f"SRX_alpha_high_Proj_{sx['proj_srx_high']}":   xh_dp_d.get(pr['f0'],  np.nan),
            f"M_blocs_projection_{sx['proj_M']}":           pr.get('M', np.nan),
            f"SDF_Proj_TCL_empirique_{sx['proj_sdf']}":     pr.get('sdf_proj_tcl', np.nan),
            f"SDF_Proj_Kappa4_LogN_{sx['proj_sdf']}":       pr.get('sdf_proj_k4_logn', np.nan),
            f"SDF_Proj_DSP_Bendat_{sx['proj_dsp']}":        pr.get('sdf_proj_dsp', np.nan),
        }
        if has_gev:
            row['GEV_domaine'] = pr.get('gev_domaine') or 'n_a'
        return row
    rows = [_row(pr) for pr in sorted(proj_results, key=lambda x: x['f0'])]
    pd.DataFrame(rows).to_csv(filepath, index=False, sep=';', decimal=',')
    logger.info("CSV Projection : %s", filepath)


def exporter_csv_sdf_kappa4_fit(filepath, results, meta):
    """CSV diagnostic : ajustement Kappa4 par classe sur les D_bloc.
    Une ligne par (f0, classe) avec n_points, ξ/α/k/h, t3/t4, RMSE, MSDI.
    Paramètres (b, Q, Tb) suffixés dans les noms de colonnes des grandeurs
    qui en dépendent."""
    import pandas as pd
    sx = _build_col_suffixes(meta)
    rows = []
    for r in sorted(results, key=lambda x: x['f0']):
        if not r.get('success'):
            continue
        f0 = r['f0']
        params_dmg_list = r.get('params_dmg_list', [])
        d_blocs_classes = r.get('d_blocs_classes', [])
        rmse_dmg_list   = r.get('rmse_dmg_list', [])
        msdi_dmg_list   = r.get('msdi_dmg_list', [])
        for i, p in enumerate(params_dmg_list):
            if not isinstance(p, dict):
                continue
            n_pts = len(d_blocs_classes[i]) if i < len(d_blocs_classes) else 0
            rows.append({
                'Frequence_Hz':           f0,
                'Classe':                 i,
                'N_Dblocs':               n_pts,
                'loi':                    p.get('loi', 'kappa4'),
                'fit_success':            bool(p.get('success', False)),
                f"xi_{sx['fit']}":        p.get('xi', np.nan),
                f"alpha_{sx['fit']}":     p.get('alpha', np.nan),
                f"k_{sx['fit']}":         p.get('k', np.nan),
                f"h_{sx['fit']}":         p.get('h', np.nan),
                't3':                     p.get('t3', np.nan),
                't4':                     p.get('t4', np.nan),
                f"RMSE_{sx['fit']}":      (rmse_dmg_list[i] if i < len(rmse_dmg_list) else np.nan),
                f"MSDI_{sx['fit']}":      (msdi_dmg_list[i] if i < len(msdi_dmg_list) else np.nan),
            })
    if not rows:
        logger.info("CSV SDF Kappa4 fit : aucune ligne (SDF désactivé ou aucun fit réussi).")
        return
    pd.DataFrame(rows).to_csv(filepath, index=False, sep=';', decimal=',')
    logger.info("CSV SDF Kappa4 fit : %s (%d lignes)", filepath, len(rows))


def exporter_csv_iid(filepath, results, meta):
    """CSV diagnostic Quality Gate IID — granularité par classe.
    Une ligne par (f0, classe) avec, pour les branches SRE et SDF, le ρ de
    Spearman lag-1 (sur rangs), la p-value du test des suites, n et le statut
    (ok / FAIL / n_a). Paramètres (Q, Tb) suffixés dans les noms de colonnes."""
    import pandas as pd
    sx = _build_col_suffixes(meta)

    def _trip(d):
        if not isinstance(d, dict) or not d.get('tested'):
            return (d.get('rho', np.nan) if isinstance(d, dict) else np.nan,
                    d.get('pvalue', np.nan) if isinstance(d, dict) else np.nan,
                    'n_a')
        return (d.get('rho', np.nan), d.get('pvalue', np.nan),
                'FAIL' if d.get('fail') else 'ok')

    rows = []
    for r in sorted(results, key=lambda x: x['f0']):
        if not r.get('success') or 'iid' not in r:
            continue
        f0  = r['f0']
        iid = r['iid']
        # Ligne "globale" (toutes classes confondues) : Classe = -1
        g_sre = _trip(iid.get('sre', {}))
        g_sdf = _trip(iid.get('sdf', {}))
        rows.append({
            'Frequence_Hz': f0, 'Classe': -1,
            'N': iid.get('sre', {}).get('n', np.nan),
            f"IID_SRE_rho_lag1_{sx['iid']}":    g_sre[0],
            f"IID_SRE_pvalue_runs_{sx['iid']}": g_sre[1],
            f"IID_SRE_status_{sx['iid']}":      g_sre[2],
            f"IID_SDF_rho_lag1_{sx['iid']}":    g_sdf[0],
            f"IID_SDF_pvalue_runs_{sx['iid']}": g_sdf[1],
            f"IID_SDF_status_{sx['iid']}":      g_sdf[2],
        })
        for pc in iid.get('per_class', []):
            c_sre = _trip(pc.get('sre', {}))
            c_sdf = _trip(pc.get('sdf', {}))
            rows.append({
                'Frequence_Hz': f0, 'Classe': pc.get('classe'),
                'N': pc.get('n', np.nan),
                f"IID_SRE_rho_lag1_{sx['iid']}":    c_sre[0],
                f"IID_SRE_pvalue_runs_{sx['iid']}": c_sre[1],
                f"IID_SRE_status_{sx['iid']}":      c_sre[2],
                f"IID_SDF_rho_lag1_{sx['iid']}":    c_sdf[0],
                f"IID_SDF_pvalue_runs_{sx['iid']}": c_sdf[1],
                f"IID_SDF_status_{sx['iid']}":      c_sdf[2],
            })
    if not rows:
        logger.info("CSV IID Quality Gate : aucune ligne (gate désactivé ?).")
        return
    pd.DataFrame(rows).to_csv(filepath, index=False, sep=';', decimal=',')
    logger.info("CSV IID Quality Gate : %s (%d lignes)", filepath, len(rows))


def exporter_csv_debug_analytic(filepath, results, meta):
    """CSV diagnostic : une ligne par (f0, classe) avec les métriques internes
    de direct_pwm_analytic (L-moments, warm_start, résidus fsolve, g1-g2, ξ/α/k/h,
    exit_reason). Les paramètres finaux du fit (suffixés Q, Tb) sont conditionnés
    par la mécanique SDOF puisque le fit porte sur les maxima."""
    import pandas as pd
    sx = _build_col_suffixes(meta)
    rows = []
    for r in sorted(results, key=lambda x: x['f0']):
        f0 = r.get('f0')
        params_list = r.get('params_list', [])
        maxima_classes = r.get('maxima_classes', [])
        for i, p in enumerate(params_list):
            if not isinstance(p, dict):
                continue
            dbg = p.get('_debug') if isinstance(p, dict) else None
            if dbg is None:
                continue
            n_pts = len(maxima_classes[i]) if i < len(maxima_classes) else np.nan
            row = {
                'Frequence_Hz':                  f0,
                'Classe':                        i,
                'N_points':                      n_pts,
                'fit_success':                   p.get('success', False),
                f"xi_final_{sx['sdof']}":        p.get('xi', np.nan),
                f"alpha_final_{sx['sdof']}":     p.get('alpha', np.nan),
                f"k_final_{sx['sdof']}":         p.get('k', np.nan),
                f"h_final_{sx['sdof']}":         p.get('h', np.nan),
                'ks':                            p.get('ks', np.nan),
                'pearson_r':                     p.get('pearson_r', np.nan),
            }
            for key in ('n_data', 'l1', 'l2', 't3', 't4',
                        'warm_start', 'fsolve_ier',
                        'fsolve_sol_k', 'fsolve_sol_h',
                        'fsolve_res1', 'fsolve_res2', 'fsolve_res_norm2',
                        'tau3_calc', 'tau4_calc',
                        'g1', 'g2', 'g1_minus_g2',
                        'xi', 'alpha', 'exit_reason'):
                row[f'dbg_{key}'] = dbg.get(key)
            rows.append(row)

    if not rows:
        logger.info("Debug Kappa4 : aucune ligne à exporter (KAPPA4_DEBUG_ANALYTIC=False ?)")
        return
    pd.DataFrame(rows).to_csv(filepath, index=False, sep=';', decimal=',')
    logger.info("CSV Debug Kappa4 : %s (%d lignes)", filepath, len(rows))

# =============================================================================
# SECTION 12 — RAPPORT HTML (Plotly interactif)
# =============================================================================

def _kappa4_tau_curve_for_h(h_val, k_grid=None):
    """Échantillonne la courbe (τ3, τ4) théorique Kappa4 pour h fixé en faisant
    varier k. Remplace l'usage de la table polynomiale Mielke."""
    if k_grid is None:
        k_grid = np.linspace(-0.95, 4.5, 250)
    t3, t4 = [], []
    for k in k_grid:
        a, b = _tau3_tau4_from_kh_analytic(float(k), float(h_val))
        if np.isfinite(a) and np.isfinite(b) and -1.0 < a < 1.0:
            t3.append(a); t4.append(b)
    if not t3:
        return np.array([]), np.array([])
    arr = np.array(sorted(zip(t3, t4)))
    return arr[:, 0], arr[:, 1]


def _axe_key(prefix, row):
    """Clé d'axe Plotly pour un sous-graphe (col=1) : 'xaxis'/'xaxis2'…"""
    return f"{prefix}axis" if row == 1 else f"{prefix}axis{row}"


def _ajouter_boutons_echelle(n_rows, x=1.005, y_top=1.0):
    """Renvoie une liste d'`updatemenus` Plotly : pour chaque sous-graphe,
    4 boutons (X lin/log, Y lin/log) agissant par relayout ciblé sur l'axe
    du graphe concerné. Réutilisé par generer_html et generer_html_details
    (l'appelant concatène à ses propres updatemenus éventuels)."""
    menus = []
    step = 1.0 / max(n_rows, 1)
    for i in range(1, n_rows + 1):
        xk, yk = _axe_key('x', i), _axe_key('y', i)
        yc = y_top - (i - 1) * step - step * 0.5
        menus.append(dict(
            type='buttons', direction='right', showactive=False,
            x=x, y=yc, xanchor='left', yanchor='middle',
            pad=dict(r=2, t=2), font=dict(size=9),
            bgcolor='white', bordercolor='lightgray',
            buttons=[
                dict(label=f'G{i} X-lin', method='relayout',
                     args=[{f'{xk}.type': 'linear'}]),
                dict(label='X-log', method='relayout',
                     args=[{f'{xk}.type': 'log'}]),
                dict(label='Y-lin', method='relayout',
                     args=[{f'{yk}.type': 'linear'}]),
                dict(label='Y-log', method='relayout',
                     args=[{f'{yk}.type': 'log'}]),
            ]))
    return menus


def _ecrire_html_avec_cadre(fig, filepath, config_info=None):
    """Écrit le HTML Plotly avec un cadre de paramètres STATIQUE et REPLIABLE
    en tête de page (<details>), au lieu d'une annotation Plotly opaque qui
    gêne la lecture du graphe. Conserve plotly.js en ligne (hors-connexion)."""
    import plotly.io as pio
    html = pio.to_html(fig, full_html=True, include_plotlyjs=True)
    if config_info:
        rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0;font-weight:bold;"
            f"white-space:nowrap'>{k}</td>"
            f"<td style='padding:2px 0'>{v}</td></tr>"
            for k, v in config_info.items())
        bloc = (
            "<details open style=\"font-family:Segoe UI,Arial,sans-serif;"
            "margin:10px;border:1px solid #c8c8c8;border-radius:6px;"
            "padding:6px 12px;background:#f6f6f6;max-width:1000px\">"
            "<summary style='cursor:pointer;font-weight:bold;font-size:14px'>"
            "Paramètres du calcul (cliquer pour replier / déplier)</summary>"
            f"<table style='font-size:12px;margin-top:6px;"
            f"border-collapse:collapse'>{rows}</table></details>")
        low = html.lower()
        idx = low.find('<body>')
        if idx != -1:
            pos = idx + len('<body>')
            html = html[:pos] + bloc + html[pos:]
        else:
            html = bloc + html
    with open(filepath, 'w', encoding='utf-8') as fh:
        fh.write(html)


def generer_html(filepath, results, sre_dsp, srx_low, srx_high, sdf_spectral,
                 f0_grid, proj_results=None, config_info=None,
                 sre_dsp_proj=None, srx_low_proj=None, srx_high_proj=None,
                 alpha_srx_low=None, alpha_srx_high=None, iid_diag=None):
    """Rapport HTML interactif Plotly (4 graphiques max).

    Affiche en regard du SRE MBD-Kappa4 les estimateurs analytiques NORMDEF :
      - SRE DSP (NORMDEF §5.4.2)
      - SRX à risque α_low  — dimensionnement enveloppe haute
      - SRX à risque α_high — comparaison vs SRC d'un choc (NORMDEF §5.4.3)
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly non installé — rapport HTML non généré.")
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    def _get(lst, f0, key, default=np.nan):
        for r in lst:
            if r['f0'] == f0 and r['success']:
                v = r.get(key)
                return v if v is not None else default
        return default

    f0_list = sorted(r['f0'] for r in results if r['success'])
    sre_d_d    = dict(zip(f0_grid, sre_dsp))
    xl_d       = dict(zip(f0_grid, srx_low))
    xh_d       = dict(zip(f0_grid, srx_high))
    sp_d       = dict(zip(f0_grid, sdf_spectral)) if sdf_spectral is not None else {}
    sre_d_proj = dict(zip(f0_grid, sre_dsp_proj))  if sre_dsp_proj  is not None else {}
    xl_proj_d  = dict(zip(f0_grid, srx_low_proj))  if srx_low_proj  is not None else {}
    xh_proj_d  = dict(zip(f0_grid, srx_high_proj)) if srx_high_proj is not None else {}

    sre_k4  = [_get(results, f, 'sre')            for f in f0_list]
    src_v   = [_get(results, f, 'src')             for f in f0_list]

    # Diagnostic SRE Kappa4 : raison synthétique + détail par classe.
    # Aligné sur f0_list pour customdata Plotly.
    def _sre_diag(f0):
        for r in results:
            if r['f0'] == f0 and r['success']:
                return r.get('sre_diag') or {}
        return {}
    sre_reasons       = [(_sre_diag(f).get('reason') or 'n/a') for f in f0_list]
    sre_per_class     = [(_sre_diag(f).get('per_class') or []) for f in f0_list]
    # Texte de hover détaillé pour les f₀ en échec : liste compacte des statuts par classe.
    def _per_class_str(pc):
        if not pc:
            return ''
        return ' | '.join(f"cl{i}:{s}" for i, s in enumerate(pc))
    sre_per_class_str = [_per_class_str(pc) for pc in sre_per_class]
    # Confiance IID par f₀ (suggestion : surfacée dans le hover du SRE).
    _iid_conf_map = (iid_diag.get('per_f0_confidence', {})
                     if iid_diag is not None else {})
    iid_conf = [_iid_conf_map.get(float(f), 'n/a') for f in f0_list]
    sre_cd   = np.column_stack([np.asarray(sre_reasons, dtype=object),
                                np.asarray(iid_conf, dtype=object)])
    sdf_t   = [_get(results, f, 'sdf_temporel')   for f in f0_list]
    sdf_k   = [_get(results, f, 'sdf_kappa4_empirical') for f in f0_list]
    sre_d   = [sre_d_d.get(f, np.nan) for f in f0_list]
    xl_v    = [xl_d.get(f,  np.nan) for f in f0_list]
    xh_v    = [xh_d.get(f,  np.nan) for f in f0_list]
    sre_d_p = [sre_d_proj.get(f, np.nan) for f in f0_list] if sre_d_proj else None
    xl_v_p  = [xl_proj_d.get(f,  np.nan) for f in f0_list] if xl_proj_d  else None
    xh_v_p  = [xh_proj_d.get(f,  np.nan) for f in f0_list] if xh_proj_d  else None
    sdf_sp  = [sp_d.get(f, np.nan) for f in f0_list]
    aL_lbl  = f"{alpha_srx_low:.2f}"  if alpha_srx_low  is not None else "low"
    aH_lbl  = f"{alpha_srx_high:.2f}" if alpha_srx_high is not None else "high"

    sre_mbd_proj = None
    if proj_results:
        _pd = {pr['f0']: pr.get('sre_proj_max') for pr in proj_results}
        sre_mbd_proj = [_pd.get(f) for f in f0_list]

    has_sdf  = sdf_spectral is not None
    has_proj = bool(proj_results)
    n_rows   = 2 + int(has_sdf) + int(has_proj)

    titles = ['SRC & SRE Kappa4 — Spectre de réponse',
              'SRE & SRX — comparatif (SRC / MBD Kappa4 / DSP NORMDEF §5.4.2-5.4.3)']
    if has_sdf:  titles.append('SDF — Spectre de dommage par fatigue')
    if has_proj:
        _meth = ('GEV 3 domaines' if METHODE_PROJECTION == 'gev_domaines'
                 else 'puissance F^M')
        titles.append(f'SRE MBD - Projeté (T={DUREE_PROJECTION:.0e}s, '
                      f'α={ALFA_PROJECTION}, méthode {_meth})')

    # Le graphe comparatif SRE & SRX (row 2) est rehaussé de +30 %.
    _row_units = [1.0] * n_rows
    _row_units[1] = 1.3
    _row_heights = [u / sum(_row_units) for u in _row_units]
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles,
                        vertical_spacing=0.07, row_heights=_row_heights)

    fig.add_trace(go.Scatter(x=f0_list, y=src_v,  name='SRC (max réponse)',
                             line=dict(color='forestgreen', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sre_k4, name='SRE Kappa4',
                             line=dict(color='royalblue', width=2),
                             customdata=sre_cd,
                             hovertemplate=('f₀=%{x:.2f} Hz<br>SRE=%{y:.3g}'
                                            '<br>statut: %{customdata[0]}'
                                            '<br>IID: %{customdata[1]}'
                                            '<extra></extra>')),
                  row=1, col=1)

    # Marqueurs ❌ aux f₀ où SRE Kappa4 est manquant — positionnés sur SRC pour
    # rester visibles dans l'échelle du graphe.
    fail_x, fail_y, fail_hover = [], [], []
    for f, s, src, reason, pcs in zip(f0_list, sre_k4, src_v,
                                       sre_reasons, sre_per_class_str):
        if s is None or (isinstance(s, float) and not np.isfinite(s)):
            fail_x.append(f)
            fail_y.append(src if (src is not None and np.isfinite(src)) else 0.0)
            fail_hover.append(f"{reason}<br>{pcs}" if pcs else reason)
    if fail_x:
        fig.add_trace(go.Scatter(
            x=fail_x, y=fail_y,
            mode='markers+text',
            name='SRE Kappa4 — échecs (hover pour la raison)',
            marker=dict(symbol='x', color='red', size=10, line=dict(width=2)),
            customdata=fail_hover,
            hovertemplate=('f₀=%{x:.2f} Hz<br>SRE Kappa4: non calculable'
                           '<br>%{customdata}<extra></extra>')),
            row=1, col=1)

    fig.add_trace(go.Scatter(x=f0_list, y=src_v,  name='SRC',
                             line=dict(color='lightgray', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sre_k4, name='SRE MBD Kappa4 (Cunnane)',
                             line=dict(color='royalblue', width=2),
                             customdata=sre_cd,
                             hovertemplate=('f₀=%{x:.2f} Hz<br>SRE=%{y:.3g}'
                                            '<br>statut: %{customdata[0]}'
                                            '<br>IID: %{customdata[1]}'
                                            '<extra></extra>')),
                  row=2, col=1)
    if sre_mbd_proj is not None:
        fig.add_trace(go.Scatter(x=f0_list, y=sre_mbd_proj,
                                 name=f'SRE MBD projeté (T={DUREE_PROJECTION:.0e}s)',
                                #  line=dict(color='royalblue', width=2, dash='dash')),
                                 line=dict(color='steelblue', width=2)),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sre_d,
                             name='SRE DSP (NORMDEF §5.4.2)',
                             line=dict(color='seagreen', width=2)), row=2, col=1)
    if sre_d_p is not None:
        fig.add_trace(go.Scatter(x=f0_list, y=sre_d_p,
                                 name='SRE DSP projeté',
                                 line=dict(color='seagreen', width=2, dash='dash')),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=xl_v,
                             name=f'SRX α={aL_lbl} (dimensionnement)',
                             line=dict(color='darkorange', width=2)), row=2, col=1)
    if xl_v_p is not None:
        fig.add_trace(go.Scatter(x=f0_list, y=xl_v_p,
                                 name=f'SRX α={aL_lbl} projeté',
                                 line=dict(color='darkorange', width=2, dash='dash')),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=xh_v,
                             name=f'SRX α={aH_lbl} (vs choc)',
                             line=dict(color='firebrick', width=2)), row=2, col=1)
    if xh_v_p is not None:
        fig.add_trace(go.Scatter(x=f0_list, y=xh_v_p,
                                 name=f'SRX α={aH_lbl} projeté',
                                 line=dict(color='firebrick', width=2, dash='dash')),
                      row=2, col=1)
    fig.update_yaxes(type='log', row=2, col=1)

    row = 3
    if has_sdf:
        def _clean_pos_basic(arr):
            a = np.asarray(arr, dtype=float)
            return np.where((a > 0) & np.isfinite(a), a, np.nan)
        sdf_t_p  = _clean_pos_basic(sdf_t)
        sdf_k_p  = _clean_pos_basic(sdf_k)
        sdf_sp_p = _clean_pos_basic(sdf_sp)
        fig.add_trace(go.Scatter(x=f0_list, y=sdf_t_p,  name='SDF Rainflow (global)',
                                 line=dict(color='crimson', width=2)), row=row, col=1)
        fig.add_trace(go.Scatter(x=f0_list, y=sdf_k_p,  name='SDF MBD-AnnexeC (Σ D_bloc)',
                                 line=dict(color='orange', width=2)), row=row, col=1)
        fig.add_trace(go.Scatter(x=f0_list, y=sdf_sp_p, name='SDF Bendat spectral',
                                 line=dict(color='purple', dash='dash')), row=row, col=1)
        fig.update_yaxes(type='log', title_text='SDF (log)', row=row, col=1)
        row += 1

    if has_proj:
        pd_ = {pr['f0']: pr.get('sre_proj_max') for pr in proj_results}
        sre_proj = [pd_.get(f) for f in f0_list]
        fig.add_trace(go.Scatter(x=f0_list, y=sre_k4,   name='SRE MBD - Temps du signal',
                                 line=dict(color='steelblue', width=1),
                                 showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=f0_list, y=sre_proj, name='SRE MBD - Projeté',
                                 line=dict(color='crimson', width=2)), row=row, col=1)
        row += 1

    # Boutons d'échelle log/linéaire par graphique (X et Y indépendants).
    # Le nom du fichier analysé (≤35 caractères) apparaît dans le titre.
    _nom_fic = (config_info or {}).get('Fichier', '')
    fig.update_layout(
        title=(f"MBD — Analyse Spectrale — {_nom_fic}  "
               f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]"),
        height=int(320 * sum(_row_units)),
        margin=dict(r=120),
        updatemenus=_ajouter_boutons_echelle(n_rows),
    )
    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Fréquence f₀ (Hz)", row=i, col=1)

    # Cadre des paramètres : bloc HTML statique repliable (non opaque),
    # remplace l'ancienne annotation Plotly qui masquait le graphe.
    _ecrire_html_avec_cadre(fig, filepath, config_info)
    logger.info("Rapport HTML : %s", filepath)


def generer_html_details(filepath, results, sre_dsp, srx_low, srx_high,
                          sdf_spectral, f0_grid, config_info=None,
                          proj_results=None, t_mesure=None, t_proj=None,
                          alpha_srx_low=None, alpha_srx_high=None,
                          iid_diag=None):
    """Rapport HTML de diagnostic (τ3/τ4, CDF par classe, comparatifs SRE/SDF).

    Affiche le comparatif SRE incluant les estimateurs analytiques NORMDEF
    (SRE §5.4.2 et SRX §5.4.3 à deux niveaux α)."""
    if not HAS_PLOTLY:
        logger.warning("Plotly non installé — rapport détails non généré.")
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    results_ok = [r for r in results if r.get('success')]
    if not results_ok:
        logger.warning("Aucun résultat exploitable pour le rapport détails.")
        return
    results_ok.sort(key=lambda r: r['f0'])
    f0_list = [r['f0'] for r in results_ok]
    sre_d_d = dict(zip(f0_grid, sre_dsp))
    xl_d    = dict(zip(f0_grid, srx_low))
    xh_d    = dict(zip(f0_grid, srx_high))
    sp_d    = dict(zip(f0_grid, sdf_spectral)) if sdf_spectral is not None else {}
    aL_lbl  = f"{alpha_srx_low:.2f}"  if alpha_srx_low  is not None else "low"
    aH_lbl  = f"{alpha_srx_high:.2f}" if alpha_srx_high is not None else "high"

    if t_mesure and t_proj and t_mesure > 0:
        ratio_rf = t_proj / t_mesure
    else:
        ratio_rf = None

    proj_d = {}
    if proj_results:
        for pr in proj_results:
            proj_d[pr['f0']] = pr

    def _safe_mean(vals):
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                          dtype=float)
        return float(np.mean(arr)) if arr.size else np.nan

    msdi_mean = [_safe_mean(r.get('msdi_list', [])) for r in results_ok]
    rmse_mean = [_safe_mean(r.get('rmse_list', [])) for r in results_ok]

    has_iid = (iid_diag is not None
               and len(np.asarray(iid_diag.get('f0', []))) > 0)

    subplot_titles = [
        'Diagramme τ3/τ4 — un point par (f0, classe) + surbrillance f0',
        'CDF par classe (ECDF vs Kappa4 ajusté, RMSE | MSDI dans la légende)',
        'Comparatif SRE : SRC / MBD Kappa4 / SRE DSP / SRX α_low / SRX α_high (+ MSDI moyen)',
        'Comparatif SDF projeté (axe log) : MBD TCL empirique / MBD K4-LogN / spectral DSP / rainflow brut & projeté',
    ]
    specs = [
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
    ]
    if has_iid:
        subplot_titles.append(
            'Quality Gate IID — ρ Spearman lag-1 (sur rangs) & p-value '
            'test des suites vs f0 (seuils en pointillés ; statut %s)'
            % iid_diag.get('status', ''))
        specs.append([{"secondary_y": False}])
    n_rows_det = len(subplot_titles)
    # Le graphe « Comparatif SRE » (row 3) est rehaussé de +30 %.
    _ru_det = [1.0] * n_rows_det
    _ru_det[2] = 1.3
    _rh_det = [u / sum(_ru_det) for u in _ru_det]
    fig = make_subplots(
        rows=n_rows_det, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.07,
        specs=specs,
        row_heights=_rh_det,
    )

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # --- Row 1 : diagramme τ3/τ4 ---------------------------------------------
    # Courbes théoriques Kappa4 calculées analytiquement (variation de k à h fixé).
    for h_val, color, extra in [(-1.0, '#888888', ' (GLO — limite haute)'),
                                (0.0, '#4444aa', ' (GEV)'),
                                (1.0, '#44aa44', ' (GPD)'),
                                (5.0, '#aa4444', '')]:
        xs_th, ys_th = _kappa4_tau_curve_for_h(h_val)
        if xs_th.size:
            fig.add_trace(go.Scatter(
                x=xs_th, y=ys_th,
                mode='lines', line=dict(color=color, width=1, dash='dot'),
                name=f'Kappa4 théorique h={h_val}{extra}',
                legendgroup='tau_theo',
            ), row=1, col=1)

    # Domaine de validité (τ3, τ4) — [2] eq. 15 / Figure 10 :
    #   limite BASSE (toutes lois) : τ4 = (5τ3² − 1)/4 (parabole) ;
    #   limite HAUTE du domaine Kappa4 : courbe h = −1 (GLO),
    #   τ4 = (5τ3² + 1)/6, déjà tracée ci-dessus parmi les courbes à h fixé.
    tau3_grid = np.linspace(-0.999, 0.999, 500)
    fig.add_trace(go.Scatter(
        x=tau3_grid, y=(5 * tau3_grid ** 2 - 1) / 4,
        mode='lines', line=dict(color='black', width=1),
        name='Limite basse τ4 = (5τ3²−1)/4 ([2] eq. 15)',
        legendgroup='tau_theo',
    ), row=1, col=1)

    n_k_max = max(len(r.get('params_list', [])) for r in results_ok)
    for ci in range(n_k_max):
        xs, ys, txt = [], [], []
        for r in results_ok:
            pl = r.get('params_list', [])
            if ci >= len(pl):
                continue
            p = pl[ci]
            t3, t4 = p.get('t3'), p.get('t4')
            if t3 is None or t4 is None or not (np.isfinite(t3) and np.isfinite(t4)):
                continue
            mc = r.get('maxima_classes', [])
            n_pts = len(mc[ci]) if ci < len(mc) else 0
            xs.append(t3); ys.append(t4)
            txt.append(f"f0={r['f0']:.1f} Hz | classe {ci} | N={n_pts} | "
                       f"success={p.get('success')}")
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(size=5, color=palette[ci % len(palette)], opacity=0.7),
                name=f'Classe {ci}', text=txt, hoverinfo='text',
                legendgroup=f'class_{ci}',
            ), row=1, col=1)

    tau_hl_idx = []
    for r in results_ok:
        xs_h, ys_h, txt_h = [], [], []
        for ci, p in enumerate(r.get('params_list', [])):
            t3, t4 = p.get('t3'), p.get('t4')
            if t3 is None or t4 is None or not (np.isfinite(t3) and np.isfinite(t4)):
                continue
            xs_h.append(t3); ys_h.append(t4)
            txt_h.append(f"f0={r['f0']:.1f} Hz | classe {ci}")
        fig.add_trace(go.Scatter(
            x=xs_h, y=ys_h, mode='markers',
            marker=dict(size=16, color='rgba(0,0,0,0)',
                        line=dict(color='black', width=2)),
            name=f'Sélection f0={r["f0"]:.1f} Hz',
            text=txt_h, hoverinfo='text',
            legendgroup='tau_highlight', showlegend=False,
            visible=False,
        ), row=1, col=1)
        tau_hl_idx.append(len(fig.data) - 1)

    fig.update_xaxes(title_text='τ3 (L-skewness)', row=1, col=1)
    fig.update_yaxes(title_text='τ4 (L-kurtosis)', row=1, col=1)

    # --- Row 2 : CDF par classe, dropdown f0 ---------------------------------
    cdf_trace_groups = []
    for r in results_ok:
        group_idx = []
        rmse_list = r.get('rmse_list', [])
        msdi_list = r.get('msdi_list', [])
        for ci, (maxima_i, params_i) in enumerate(zip(
                r.get('maxima_classes', []), r.get('params_list', []))):
            if not maxima_i or len(maxima_i) < 2:
                continue
            maxima_arr = np.sort(np.asarray(maxima_i, dtype=float))
            n     = len(maxima_arr)
            ecdf  = np.arange(1, n + 1) / n
            color = palette[ci % len(palette)]

            rmse_i = rmse_list[ci] if ci < len(rmse_list) else np.nan
            msdi_i = msdi_list[ci] if ci < len(msdi_list) else np.nan
            rmse_s = f"{rmse_i:.4f}" if rmse_i is not None and np.isfinite(rmse_i) else "—"
            msdi_s = f"{msdi_i:.4f}" if msdi_i is not None and np.isfinite(msdi_i) else "—"

            fig.add_trace(go.Scatter(
                x=maxima_arr, y=ecdf, mode='markers',
                marker=dict(size=4, color=color),
                name=f'Classe {ci} — ECDF (RMSE={rmse_s} | MSDI={msdi_s})',
                legendgroup=f'cdf_class_{ci}',
                visible=False, showlegend=(len(cdf_trace_groups) == 0),
            ), row=2, col=1)
            group_idx.append(len(fig.data) - 1)

            if params_i.get('success'):
                try:
                    x_grid   = np.linspace(maxima_arr[0], maxima_arr[-1], 200)
                    if params_i.get('loi') == 'rayleigh_gen':
                        a   = float(params_i['rg_alpha'])
                        lam = float(params_i['rg_lambda'])
                        z   = np.clip(lam * x_grid, 0.0, None)
                        cdf_theo = np.power(1.0 - np.exp(-(z * z)), a)
                        nom_loi  = 'Rayleigh généralisée ajustée'
                    elif _kappa4_use_exact(params_i['k'], params_i['h']):
                        cdf_theo = _kappa4_cdf_exact(
                            x_grid, params_i['xi'], params_i['alpha'],
                            params_i['k'], params_i['h'])
                        nom_loi  = 'Kappa4 ajusté'
                    else:
                        cdf_theo = scipy_kappa4(
                            h=params_i['h'], k=params_i['k'],
                            loc=params_i['xi'], scale=params_i['alpha']
                        ).cdf(x_grid)
                        nom_loi  = 'Kappa4 ajusté'
                    fig.add_trace(go.Scatter(
                        x=x_grid, y=cdf_theo, mode='lines',
                        line=dict(color=color, width=2),
                        name=f'Classe {ci} — {nom_loi}',
                        legendgroup=f'cdf_class_{ci}',
                        visible=False, showlegend=False,
                    ), row=2, col=1)
                    group_idx.append(len(fig.data) - 1)
                except Exception:
                    pass
        cdf_trace_groups.append(group_idx)

    if cdf_trace_groups:
        for idx in cdf_trace_groups[0]:
            fig.data[idx].visible = True
    if tau_hl_idx:
        fig.data[tau_hl_idx[0]].visible = True

    fig.update_xaxes(title_text='Maxima (contrainte)', row=2, col=1)
    fig.update_yaxes(title_text='CDF', row=2, col=1)

    # --- Row 3 : SRE comparatif + MSDI moyen ---------------------------------
    def _get(lst, f0, key, default=np.nan):
        for r in lst:
            if r['f0'] == f0 and r['success']:
                v = r.get(key)
                return v if v is not None else default
        return default

    src_v  = [_get(results_ok, f, 'src') for f in f0_list]
    sre_k4 = [_get(results_ok, f, 'sre') for f in f0_list]
    sre_d  = [sre_d_d.get(f, np.nan) for f in f0_list]
    xl_v   = [xl_d.get(f,  np.nan) for f in f0_list]
    xh_v   = [xh_d.get(f,  np.nan) for f in f0_list]

    fig.add_trace(go.Scatter(x=f0_list, y=src_v,  name='SRC (max réponse)',
                             line=dict(color='forestgreen', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sre_k4, name='SRE MBD (Kappa4)',
                             line=dict(color='royalblue', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sre_d,  name='SRE DSP (NORMDEF §5.4.2)',
                             line=dict(color='seagreen', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=xl_v,
                             name=f'SRX α={aL_lbl} (dimensionnement)',
                             line=dict(color='darkorange', dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=xh_v,
                             name=f'SRX α={aH_lbl} (vs choc)',
                             line=dict(color='firebrick', dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=msdi_mean, name='MSDI moyen (classes)',
                             mode='lines+markers',
                             line=dict(color='plum', width=1, dash='dashdot'),
                             marker=dict(size=2)),
                  row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text='Fréquence f0 (Hz)', row=3, col=1)
    fig.update_yaxes(title_text='SRE / SRC', row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text='MSDI', row=3, col=1, secondary_y=True)

    # --- Row 4 : SDF comparatif projeté + MSDI moyen -------------------------
    def _clean_pos(arr):
        a = np.asarray(arr, dtype=float)
        return np.where((a > 0) & np.isfinite(a), a, np.nan)

    sdf_rf_brut = [_get(results_ok, f, 'sdf_temporel') for f in f0_list]
    if ratio_rf is not None:
        sdf_rf_proj = [v * ratio_rf if (v is not None and np.isfinite(v)) else np.nan
                       for v in sdf_rf_brut]
    else:
        sdf_rf_proj = [np.nan] * len(f0_list)

    if proj_d:
        sdf_mbd_proj   = [proj_d.get(f, {}).get('sdf_proj_tcl', np.nan) for f in f0_list]
        sdf_k4logn_proj = [proj_d.get(f, {}).get('sdf_proj_k4_logn', np.nan) for f in f0_list]
        sdf_dsp_proj   = [proj_d.get(f, {}).get('sdf_proj_dsp', np.nan) for f in f0_list]
    else:
        logger.info("Rapport détails : proj_results absent — SDF row 4 en valeurs non projetées.")
        sdf_mbd_proj   = [_get(results_ok, f, 'sdf_kappa4_empirical') for f in f0_list]
        sdf_k4logn_proj = [np.nan] * len(f0_list)
        sdf_dsp_proj   = [sp_d.get(f, np.nan) for f in f0_list]

    sdf_mbd_proj    = _clean_pos(sdf_mbd_proj)
    sdf_k4logn_proj = _clean_pos(sdf_k4logn_proj)
    sdf_dsp_proj    = _clean_pos(sdf_dsp_proj)
    sdf_rf_brut     = _clean_pos(sdf_rf_brut)
    sdf_rf_proj     = _clean_pos(sdf_rf_proj)

    fig.add_trace(go.Scatter(x=f0_list, y=sdf_mbd_proj,
                             name='SDF MBD projeté (TCL empirique)',
                             line=dict(color='orange', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sdf_k4logn_proj,
                             name='SDF MBD K4-LogN projeté',
                             line=dict(color='gold', width=2, dash='dot')), row=4, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sdf_dsp_proj,
                             name='SDF spectral projeté (DSP)',
                             line=dict(color='purple', width=2, dash='dash')),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sdf_rf_brut,
                             name='SDF rainflow brut (non projeté)',
                             line=dict(color='crimson', width=1, dash='dot')),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=sdf_rf_proj,
                             name='SDF rainflow projeté',
                             line=dict(color='crimson', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=f0_list, y=msdi_mean,
                             name='MSDI moyen (classes)',
                             mode='lines+markers',
                             line=dict(color='plum', width=1, dash='dashdot'),
                             marker=dict(size=2), showlegend=False),
                  row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text='Fréquence f0 (Hz)', row=4, col=1)
    fig.update_yaxes(title_text='SDF (log)', type='log',
                     row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text='MSDI', row=4, col=1, secondary_y=True)

    # --- Row 5 : Quality Gate IID (ρ lag-1 & p-value vs f0) ------------------
    if has_iid:
        if0   = np.asarray(iid_diag['f0'], dtype=float)
        order = np.argsort(if0)
        if0   = if0[order]

        def _ord(branch, key):
            arr = np.asarray(iid_diag.get(branch, {}).get(key, []),
                             dtype=float)
            return arr[order] if arr.size == order.size else None

        for branch, dash, cset in (('sre', 'solid', ('#1f77b4', '#7fb3d5')),
                                   ('sdf', 'dot',   ('#d62728', '#e8888a'))):
            rho_b  = _ord(branch, 'rho')
            pval_b = _ord(branch, 'pvalue')
            up     = branch.upper()
            if rho_b is not None:
                fig.add_trace(go.Scatter(
                    x=if0, y=rho_b, name=f'ρ lag-1 {up}',
                    mode='lines', line=dict(color=cset[0], width=1.5,
                                            dash=dash)),
                    row=5, col=1)
            if pval_b is not None:
                fig.add_trace(go.Scatter(
                    x=if0, y=pval_b, name=f'p-value runs {up}',
                    mode='lines', line=dict(color=cset[1], width=1.5,
                                            dash=dash)),
                    row=5, col=1)

        for yv, txt in ((IID_RHO_MAX,  f'|ρ| max = {IID_RHO_MAX}'),
                        (-IID_RHO_MAX, None),
                        (IID_PVALUE_MIN, f'p min = {IID_PVALUE_MIN}')):
            fig.add_hline(y=yv, line=dict(color='gray', width=1, dash='dash'),
                          annotation_text=txt, annotation_position='right',
                          row=5, col=1)
        fig.update_xaxes(title_text='Fréquence f0 (Hz)', row=5, col=1)
        fig.update_yaxes(title_text='ρ lag-1  /  p-value', row=5, col=1)

    sdf_yaxis_key = 'yaxis5.type'

    total_traces = len(fig.data)
    buttons = []
    for gi, (f0, group_idx) in enumerate(zip(f0_list, cdf_trace_groups)):
        vis = [True] * total_traces
        for gj, other in enumerate(cdf_trace_groups):
            state = (gj == gi)
            for idx in other:
                vis[idx] = state
        for hj, hl_idx in enumerate(tau_hl_idx):
            vis[hl_idx] = (hj == gi)
        buttons.append(dict(label=f"f0 = {f0:.1f} Hz", method='update',
                            args=[{'visible': vis}]))

    scale_buttons = [
        dict(label='SDF : log', method='relayout',
             args=[{sdf_yaxis_key: 'log'}]),
        dict(label='SDF : linéaire', method='relayout',
             args=[{sdf_yaxis_key: 'linear'}]),
    ]

    updatemenus = []
    if buttons:
        updatemenus.append(dict(
            buttons=buttons, direction='down', showactive=True,
            x=1.02, y=0.72, xanchor='left', yanchor='top',
            bgcolor='white', bordercolor='lightgray',
        ))
        fig.add_annotation(x=1.02, y=0.76, xref='paper', yref='paper',
                           text='<b>Sélection f0 (CDF + τ3/τ4)</b>', showarrow=False,
                           font=dict(size=11), xanchor='left')

    updatemenus.append(dict(
        buttons=scale_buttons, direction='down', showactive=True,
        x=1.02, y=0.20, xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='lightgray',
    ))
    fig.add_annotation(x=1.02, y=0.24, xref='paper', yref='paper',
                       text='<b>Échelle SDF (row 4)</b>', showarrow=False,
                       font=dict(size=11), xanchor='left')

    # Boutons d'échelle log/lin par graphique (placés à droite des menus
    # existants — sélection f0 / échelle SDF — pour éviter le recouvrement).
    updatemenus = updatemenus + _ajouter_boutons_echelle(n_rows_det, x=1.16)
    if updatemenus:
        fig.update_layout(updatemenus=updatemenus)

    # Le nom du fichier analysé (≤35 caractères) apparaît dans le titre.
    _nom_fic = (config_info or {}).get('Fichier', '')
    fig.update_layout(
        title=f"MBD — Rapport Détails — {_nom_fic}  "
              f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]",
        height=int(380 * sum(_ru_det)), margin=dict(r=300),
    )

    # Cadre des paramètres : bloc HTML statique repliable (non opaque).
    _ecrire_html_avec_cadre(fig, filepath, config_info)
    logger.info("Rapport HTML détails : %s", filepath)

# =============================================================================
# SECTION 13 — PROGRAMME PRINCIPAL
# =============================================================================

def main(use_mp=False):
    np.random.seed(RANDOM_SEED)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Dossier de résultats dédié au run --------------------------------
    # Un sous-dossier est créé à chaque calcul dans OUTPUT_FOLDER :
    #   <AAAAMMJJ_HHMMSS>_<nom fichier ≤35 car.>_<infos de calcul>
    # infos de calcul = plage de fréquences, Q, Tb, b (Basquin),
    # durée de projection, α de projection.
    base_csv      = os.path.splitext(os.path.basename(CSV_FILEPATH))[0]
    nom_fichier35 = ''.join(c if (c.isalnum() or c in '-_') else '_'
                            for c in base_csv)[:35]
    infos_calc = (f"f{F0_MIN:.0f}-{F0_MAX:.0f}"
                  f"_Q{_fmt_compact(Q)}_Tb{_fmt_compact(TB)}"
                  f"_b{_fmt_compact(SDF_B)}"
                  f"_T{_fmt_duration_compact(DUREE_PROJECTION)}"
                  f"_a{_fmt_compact(ALFA_PROJECTION)}")
    run_dir = os.path.join(OUTPUT_FOLDER, f"{ts}_{nom_fichier35}_{infos_calc}")
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  MBD Kappa4 — dommage NF X50 144-3 Annexe C")
    logger.info("  Fix 2026-05-01 : SDF (rainflow + Bendat) calculé sur z(t)")
    logger.info("                   = déplacement relatif (m), plus σ=ω²·z.")
    logger.info("=" * 60)

    # Garde-fou : la branche dommage MBD-AnnexeC repose sur le rainflow par bloc.
    # Implémentation Numba (ASTM E1049 / AFNOR A03-406) — voir SECTION 8.
    global SDF_ENABLED
    if SDF_ENABLED and not HAS_RAINFLOW:
        logger.warning("Numba indisponible — SDF désactivé. "
                       "Installer via : pip install numba")
        SDF_ENABLED = False

    if KAPPA4_DEBUG_ANALYTIC:
        logger.info("   ⚠ KAPPA4_DEBUG_ANALYTIC=True → CSV Kappa4_Debug_* sera exporté")

    # Pré-chauffage JIT Numba : compile une fois dans le process principal
    # (le cache disque est ensuite réutilisé par chaque worker du pool).
    if SDF_ENABLED:
        _warmup = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 0.0], dtype=np.float64)
        _ = _rainflow_damage(_warmup, 1.0, 8.0)

    # 1. Import du signal
    logger.info("1. Import du signal : %s", CSV_FILEPATH)
    t, signal = importer_signal_csv(CSV_FILEPATH, skip_rows=CSV_SKIP_ROWS,
                                     delimiter=CSV_DELIMITER)
    fs = float(1.0 / np.mean(np.diff(t)))
    logger.info("   fs=%.1f Hz | %d points | durée=%.2f s", fs, len(signal), t[-1] - t[0])

    # 2. Extraction des features de l'excitation
    logger.info("2. Extraction des features de l'excitation (Tb=%.3f s)...", TB)
    features_exc, maxima_exc, used_names, n_blocs_exc, _ = extraire_caracteristiques(
        signal, fs, Tb_initial=TB, feature_flags=FEATURE_FLAGS, min_ech=MIN_ECH_PAR_BLOC)
    logger.info("   %d blocs | %d features actives : %s", n_blocs_exc, len(used_names), used_names)

    # 3. Classification K-Means sur l'excitation
    logger.info("3. Classification K-Means...")
    if N_CLUSTERS == 1 or features_exc.shape[0] < 2 or len(used_names) == 0:
        clusters = np.zeros(n_blocs_exc, dtype=int)
        n_k_final = 1
        logger.info("   K=1 : tous les blocs dans une seule classe.")
    else:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        scaler     = StandardScaler()
        feat_sc    = scaler.fit_transform(features_exc)
        n_k        = N_CLUSTERS

        if AUTO_SELECT_K:
            scores = {}
            for kv in K_RANGE:
                if 2 <= kv < feat_sc.shape[0]:
                    try:
                        lbl = KMeans(n_clusters=kv, random_state=RANDOM_SEED,
                                     n_init=10).fit_predict(feat_sc)
                        scores[kv] = silhouette_score(feat_sc, lbl)
                    except Exception:
                        pass
            if scores:
                n_k = max(scores, key=scores.get)
                logger.info("   K optimal (Silhouette) = %d", n_k)

        while n_k >= 1:
            km      = KMeans(n_clusters=n_k, random_state=RANDOM_SEED, n_init=10)
            clusters = km.fit_predict(feat_sc)
            counts   = np.bincount(clusters)
            if np.all(counts >= MIN_SAMPLES_PER_CLUSTER) or n_k == 1:
                break
            n_k -= 1
        n_k_final = n_k
        logger.info("   K=%d | tailles : %s", n_k_final, np.bincount(clusters).tolist())

    # 4. Boucle principale sur le spectre de fréquences
    # DELTA_F0 = pas en Hz (correction V3.5 : l'ancienne formule multipliait
    # au lieu de diviser — identique pour DELTA_F0=1, faux sinon).
    num_f0     = int(round((F0_MAX - F0_MIN) / DELTA_F0)) + 1
    f0_spectrum = np.linspace(F0_MIN, F0_MAX, num_f0)

    logger.info("   Méthode Kappa4 (SRE & dommage) : %s",
                KAPPA4_METHOD)

    kwargs_f0 = dict(
        n_clusters=n_k_final, Tb=TB, Q=Q, fs=fs,
        prob_cible=PROBABILITE_CIBLE, option_cunnane=OPTION_CUNNANE, cunnane_a=CUNNANE_A,
        sdf_b=SDF_B, sdf_C=SDF_C, sdf_enabled=SDF_ENABLED,
        min_points=MIN_POINTS_KAPPA4,
    )

    all_results = []
    if use_mp:
        n_workers = N_WORKERS if N_WORKERS is not None else _auto_n_workers()
        logger.info("4. Traitement de %d fréquences (%.1f—%.1f Hz) — parallèle "
                    "(%d workers, signal en shared_memory)...",
                    num_f0, F0_MIN, F0_MAX, n_workers)

        from multiprocessing import shared_memory as _shm
        signal_c = np.ascontiguousarray(signal, dtype=np.float64)
        shm_blk  = _shm.SharedMemory(create=True, size=signal_c.nbytes)
        shm_view = np.ndarray(signal_c.shape, dtype=signal_c.dtype,
                              buffer=shm_blk.buf)
        shm_view[:] = signal_c
        signal_ref = (shm_blk.name, signal_c.shape, signal_c.dtype.str)

        chunksize = max(1, num_f0 // (n_workers * 4))
        try:
            with mp.Pool(n_workers, initializer=_mp_worker_init,
                         initargs=(signal_ref, clusters, kwargs_f0)) as pool:
                for res, timings in tqdm(
                        pool.imap_unordered(_mp_worker_traiter, f0_spectrum,
                                            chunksize=chunksize),
                        total=num_f0, desc="Traitement f0 (MP)"):
                    all_results.append(res)
                    for k, v in timings.items():
                        KAPPA4_TIMINGS[k] = KAPPA4_TIMINGS[k] + v
        finally:
            shm_blk.close()
            try:
                shm_blk.unlink()
            except FileNotFoundError:
                pass
        all_results.sort(key=lambda r: r['f0'])
    else:
        logger.info("4. Traitement de %d fréquences (%.1f—%.1f Hz) — séquentiel...",
                    num_f0, F0_MIN, F0_MAX)
        for f0 in tqdm(f0_spectrum, desc="Traitement f0"):
            res = traiter_f0(f0=f0, excitation=signal, clusters=clusters,
                             **kwargs_f0)
            all_results.append(res)

    successful = [r for r in all_results if r['success']]
    logger.info("   %d / %d fréquences réussies.", len(successful), num_f0)

    # 4bis. Quality Gate IID — validation de l'hypothèse d'indépendance des
    # blocs avant de faire confiance à l'inférence Kappa-4 (cahier §5).
    iid_diag = None
    if IID_GATE_ENABLED:
        logger.info("4bis. Quality Gate IID (indépendance statistique)...")
        iid_diag = agreger_quality_gate(all_results, f0_spectrum, TB,
                                        SDF_ENABLED and HAS_RAINFLOW)

    # 5. SRE / SRX analytiques depuis la DSP (NORMDEF §5.4.2 et §5.4.3)
    logger.info("5. SRE & SRX analytiques (DSP Welch)...")
    _sdf_b_sre = SDF_B if SDF_ENABLED else None
    _T_proj_sre    = DUREE_PROJECTION if ENABLE_PROJECTION else None
    _alfa_proj_sre = ALFA_PROJECTION  if ENABLE_PROJECTION else None
    (sre_dsp, srx_low, srx_high,
     sre_dsp_proj, srx_low_proj, srx_high_proj,
     sdf_spectral) = calculer_sre_analytique(
        signal, fs, Q, f0_spectrum,
        alpha_srx_low=ALPHA_SRX_LOW, alpha_srx_high=ALPHA_SRX_HIGH,
        sdf_b=_sdf_b_sre, sdf_C=SDF_C,
        T_proj=_T_proj_sre, alfa_proj=_alfa_proj_sre)

    # 6. Projection CDF longue durée
    proj_results = []
    if ENABLE_PROJECTION and successful:
        logger.info("6. Projections CDF (T=%.0f s, α=%.2f)...", DUREE_PROJECTION, ALFA_PROJECTION)
        total_blocs = len(clusters)

        duree_mesure    = len(signal) / fs
        ratio_dsp       = DUREE_PROJECTION / duree_mesure if duree_mesure > 0 else 1.0
        sdf_sp_dict     = dict(zip(f0_spectrum, sdf_spectral)) if sdf_spectral is not None else {}

        for r in tqdm(successful, desc="Projections"):
            f0              = r['f0']
            maxima_classes  = r.get('maxima_classes',  [])
            params_list     = r.get('params_list',     [])
            params_gev_list = r.get('params_gev_list', [])

            sre_proj_max = None
            M_best       = None
            dom_best     = None
            n_classes_ok = 0
            n_classes_proj_ok = 0

            for i, (params, maxima) in enumerate(zip(params_list, maxima_classes)):
                n_blocs_i = len(maxima)
                if isinstance(params, dict) and params.get('success'):
                    n_classes_ok += 1
                if METHODE_PROJECTION == 'gev_domaines':
                    # Projection par max-stabilité GEV (3 domaines, [2] §4.1).
                    p_gev = (params_gev_list[i]
                             if i < len(params_gev_list) else None)
                    sre_p, M, dom = calculer_projection_gev_domaines(
                        p_gev, n_blocs_i, total_blocs, TB,
                        DUREE_PROJECTION, ALFA_PROJECTION)
                else:
                    # Méthode historique : F^M sur la loi LOI_AJUSTEMENT.
                    sre_p, M = calculer_projection_lmoments(
                        params, maxima, n_blocs_i, total_blocs, TB,
                        DUREE_PROJECTION, ALFA_PROJECTION)
                    dom = None
                if sre_p is not None and np.isfinite(sre_p):
                    n_classes_proj_ok += 1
                    if sre_proj_max is None or sre_p > sre_proj_max:
                        sre_proj_max = sre_p
                        M_best       = M
                        dom_best     = dom

            if KAPPA4_DEBUG_ANALYTIC:
                logger.info(
                    "f0=%.2f Hz : %d/%d classes avec fit OK, %d/%d projections valides, "
                    "SRE_proj_max=%s",
                    f0, n_classes_ok, len(params_list),
                    n_classes_proj_ok, len(params_list),
                    (f"{sre_proj_max:.3f}" if sre_proj_max is not None else "None"))

            sdf_proj_tcl    = None
            sdf_proj_k4_logn = None
            if SDF_ENABLED and HAS_RAINFLOW:
                sdf_blocs = r.get('sdf_per_bloc')
                cl_trunc  = r.get('clusters_trunc')
                if sdf_blocs is not None and cl_trunc is not None and len(sdf_blocs) > 0:
                    sdf_proj_tcl = calculer_projection_sdf_tcl(
                        sdf_blocs, cl_trunc, n_k_final, total_blocs, TB,
                        DUREE_PROJECTION, ALFA_PROJECTION)

                # Projection K4 sur D_bloc (NF X50 144-3 Annexe C).
                params_dmg_list  = r.get('params_dmg_list', [])
                d_blocs_classes  = r.get('d_blocs_classes', [])
                if params_dmg_list and d_blocs_classes:
                    sdf_proj_k4_logn = calculer_projection_dmg_kappa4(
                        params_dmg_list, d_blocs_classes,
                        n_k_final, total_blocs, TB,
                        DUREE_PROJECTION, ALFA_PROJECTION)

            sdf_proj_dsp = sdf_sp_dict.get(f0, np.nan)
            if np.isfinite(sdf_proj_dsp):
                sdf_proj_dsp *= ratio_dsp

            proj_results.append({
                'f0':              f0,
                'sre_proj_max':    sre_proj_max,
                'M':               M_best,
                'gev_domaine':     dom_best,
                'sdf_proj_tcl':    sdf_proj_tcl,
                'sdf_proj_k4_logn': sdf_proj_k4_logn,
                'sdf_proj_dsp':    sdf_proj_dsp,
            })

    # 7. Exports CSV
    logger.info("7. Export des résultats CSV...")
    duree_mesure = len(signal) / fs if fs else float('nan')
    meta_run = build_run_meta(fs, duree_mesure, n_k_final, num_f0, ts)

    tag = (f"v3p5_f{F0_MIN:.0f}-{F0_MAX:.0f}"
           f"_Q{Q}_Tb{_fmt_compact(TB)}"
           f"_b{_fmt_compact(SDF_B)}"
           f"_T{_fmt_duration_compact(DUREE_PROJECTION)}"
           f"_a{_fmt_compact(ALFA_PROJECTION)}")

    meta_json = os.path.join(run_dir, f"params_{tag}_{ts}.json")
    exporter_run_meta_json(meta_json, meta_run)

    sre_csv = os.path.join(run_dir, f"SRE_{tag}_{ts}.csv")
    exporter_csv_sre(sre_csv, all_results, sre_dsp, srx_low, srx_high,
                     f0_spectrum, meta=meta_run)

    if SDF_ENABLED:
        sdf_csv = os.path.join(run_dir, f"SDF_{tag}_{ts}.csv")
        exporter_csv_sdf(sdf_csv, all_results, sdf_spectral, f0_spectrum,
                         meta=meta_run)
        sdf_fit_csv = os.path.join(run_dir, f"SDF_Kappa4_Fit_{tag}_{ts}.csv")
        exporter_csv_sdf_kappa4_fit(sdf_fit_csv, all_results, meta=meta_run)

    if IID_GATE_ENABLED:
        iid_csv = os.path.join(run_dir, f"IID_QualityGate_{tag}_{ts}.csv")
        exporter_csv_iid(iid_csv, all_results, meta=meta_run)

    if proj_results:
        proj_csv = os.path.join(run_dir, f"SRE_Projection_{tag}_{ts}.csv")
        exporter_csv_projection(proj_csv, proj_results, meta=meta_run,
                                sre_dsp_proj=sre_dsp_proj,
                                srx_low_proj=srx_low_proj,
                                srx_high_proj=srx_high_proj,
                                f0_grid=f0_spectrum)

    if KAPPA4_DEBUG_ANALYTIC:
        dbg_csv = os.path.join(run_dir, f"Kappa4_Debug_{tag}_{ts}.csv")
        exporter_csv_debug_analytic(dbg_csv, all_results, meta=meta_run)

    # --- Récap timing direct_pwm_analytic ---
    n  = KAPPA4_TIMINGS.get('analytic_n', 0)
    ts_cum = KAPPA4_TIMINGS.get('analytic', 0.0)
    if n > 0:
        logger.info("Temps cumulé Kappa4 direct_pwm_analytic : "
                    "%.2f s (%d appels, %.3f ms/appel)",
                    ts_cum, n, 1000.0 * ts_cum / n)

    # 8. Rapport HTML
    logger.info("8. Génération du rapport HTML...")
    html_path = os.path.join(run_dir, f"Rapport_{tag}_{ts}.html")
    config_info = {
        'Fichier':       nom_fichier35,
        'Fichier CSV':   CSV_FILEPATH,
        'fs':            f"{fs:.1f} Hz",
        'Loi':           ('Rayleigh généralisée' if LOI_AJUSTEMENT == 'rayleigh_gen'
                          else 'Kappa4'),
        'Tb':            f"{TB} s",
        'Q':             Q,
        'Fréquences':    f"{F0_MIN}–{F0_MAX} Hz ({num_f0} pts)",
        'K-Means':       f"K = {n_k_final}",
        'P cible':       PROBABILITE_CIBLE,
        'Cunnane':       f"a = {CUNNANE_A}" if OPTION_CUNNANE else "non",
        'SDF b':         SDF_B if SDF_ENABLED else "désactivé",
        'SRX α_low':     ALPHA_SRX_LOW,
        'SRX α_high':    ALPHA_SRX_HIGH,
        'Projection':    (f"T = {DUREE_PROJECTION:.2e} s, α = {ALFA_PROJECTION}, "
                          f"méthode = {'GEV 3 domaines' if METHODE_PROJECTION == 'gev_domaines' else 'puissance F^M'}"
                          if ENABLE_PROJECTION else "désactivée"),
        'Méthode Kappa4': KAPPA4_METHOD,
    }
    if iid_diag is not None:
        _ico = {'GO': '🟢', 'WARNING': '🟡', 'NO-GO': '🔴'}.get(
            iid_diag['status'], '')
        config_info['Quality Gate IID'] = (
            f"{_ico} {iid_diag['status']} "
            f"({100.0 * iid_diag['frac_fail']:.1f}% f₀ hors-tolérance)")
    generer_html(html_path, all_results, sre_dsp, srx_low, srx_high, sdf_spectral,
                 f0_spectrum, proj_results=proj_results or None,
                 config_info=config_info,
                 sre_dsp_proj=sre_dsp_proj,
                 srx_low_proj=srx_low_proj,
                 srx_high_proj=srx_high_proj,
                 alpha_srx_low=ALPHA_SRX_LOW,
                 alpha_srx_high=ALPHA_SRX_HIGH,
                 iid_diag=iid_diag)

    html_details = os.path.join(run_dir, f"Rapport_Details_{tag}_{ts}.html")
    _t_mesure = len(signal) / fs if fs else None
    generer_html_details(html_details, all_results, sre_dsp, srx_low, srx_high,
                         sdf_spectral, f0_spectrum, config_info=config_info,
                         proj_results=proj_results or None,
                         alpha_srx_low=ALPHA_SRX_LOW,
                         alpha_srx_high=ALPHA_SRX_HIGH,
                         t_mesure=_t_mesure,
                         t_proj=DUREE_PROJECTION if ENABLE_PROJECTION else None,
                         iid_diag=iid_diag)

    logger.info("=" * 60)
    logger.info("  Terminé. Fichiers dans : %s/", run_dir)
    logger.info("=" * 60)


def main_mp():
    """Variante multiprocess — parallélise la boucle des fréquences f0."""
    return main(use_mp=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    (main_mp() if USE_MULTIPROCESS else main())
    logger.info("Temps total : %.1f s", time.perf_counter() - t0)
    print(f"\nTemps total : {time.perf_counter() - t0:.1f} s")
