# -*- coding: utf-8 -*-
"""
MBD Kappa4 — Script mono-fichier autoporteur (V3.6)
======================================================

Outil de calcul des spectres SRC / SRE & SRX / SDF d'un signal d'accélération
mesuré, avec projection longue durée. Implémente la méthode « MBD non
corrélée » de la norme NF X50-144-3 (2021) Annexe C, avec deux lois
d'ajustement au choix :
  - Kappa4 à 4 paramètres (Hosking [6]), ajustée par L-moments analytiques
    ([2] Colin 2023) — loi par défaut, domaine (τ3, τ4) le plus large ;
  - Rayleigh généralisée à 2 paramètres ([3] Clou & Lelan, DGA TT / CFM 2025),
    F(x ; α, λ) = (1 − e^(−(λx)²))^α — PPF analytique exacte, domaine
    restreint mais estimation très stable d'une fréquence à l'autre.

Le fichier est AUTOPORTEUR : aucun module compagnon, aucune donnée externe.
Tous les réglages sont regroupés en SECTION 1 (usage courant) puis dans la
carte « PARAMÈTRES EXPERT » qui la suit (réglages numériques fins).

----------------------------------------------------------------------
Ce que fait la V3.6, en 6 points
----------------------------------------------------------------------
1. Identification Kappa4 EN DEUX ÉTAPES, avec contrôle du domaine de h
   (fonction `kappa4_from_lmoments`, SECTION 6) :
       étape 1 — la FORME (k, h) est résolue à partir des seuls ratios de
                 L-moments (τ3, τ4) : système 2×2 τ3(k,h)=τ3̂, τ4(k,h)=τ4̂ ;
       étape 2 — la POSITION et l'ÉCHELLE (ξ, α) sont déduites de (L1, L2) et
                 de la forme obtenue, en forme close ([2] eq. 33-34).
   La correspondance (k, h) → (τ3, τ4) n'est injective que pour h ≥ −1 : une
   racine h < KAPPA4_H_MIN est rejetée, et le solveur est relancé sur des
   amorces situées dans h ∈ [−1, 0). Si le couple (τ3, τ4) mesuré tombe
   au-dessus de la courbe h = −1 (limite haute du domaine, loi logistique
   généralisée), la loi retenue est celle du BORD du domaine : h* = −1, au
   point de cette courbe le plus proche du point mesuré en distance
   euclidienne ([2] §4.1, éq. 28-29), d'où k* en forme close.

2. Réponse 1-DDL amorcée (SECTION 5, option SDOF_AMORCAGE) : le filtre
   récursif de Smallwood démarre dans un état stationnaire et non dans l'état
   « excitation constante = x[0] », qui injecte aux basses f₀ une oscillation
   libre parasite d'amplitude ≈ |x[0]|.

3. Découpage en blocs T_b exact (SECTION 4) : chaque bloc compte round(T_b·fs)
   échantillons, quelle que soit la longueur du signal (la fin de signal qui ne
   remplit pas un bloc est ignorée). Le coefficient d'extrapolation M est
   calculé avec cette durée de bloc effective.

4. Contrôle de stationnarité AVANT calcul (SECTION 9ter), bâti sur la Quality
   Gate IID : sondes f₀ sur les maxima de réponse, coefficient de variation du
   RMS par bloc, gaussianité. Verdict STATIONNAIRE / NON_STATIONNAIRE /
   DOUTEUX, seuils entièrement réglables par l'utilisateur (SECTION 1).

5. Calcul EN UNE SEULE CLASSE quand le signal est jugé stationnaire
   (AUTO_SELECT_K + AUTO_K_SELON_STATIONNARITE) : pas de partition artificielle
   d'un signal homogène, donc statistiques de blocs maximales pour l'inférence.

6. Synthèse des classes par PRODUIT des répartitions (SECTION 9quater,
   SYNTHESE_CLASSES='produit') : P(Z_sup ≤ z) = Π_j F_j(z)^{M_j} conformément
   à [1] §C.10 et [2] éq. 37, au lieu du maximum des quantiles par classe —
   pour le SRE à la durée du signal comme pour le SRE projeté. Une classe sans
   loi exploitable, donc absente de la synthèse, est signalée.

----------------------------------------------------------------------
Calcul du dommage (SDF) selon NF X50-144-3 §C.10-C.11
----------------------------------------------------------------------
  - SDF empirique = somme des dommages-bloc (comptage rainflow par bloc) : la
    granularité élémentaire du dommage est le bloc T_b, le dommage d'une
    classe est la somme des D_bloc (règle de Miner).
  - Ajustement de la loi sur la distribution des D_bloc par classe, en
    parallèle de la branche SRE qui travaille, elle, sur les maxima.
  - Projection longue durée du dommage par TCL/log-normale alimentée par les
    moments de la loi ajustée sur les D_bloc (somme de variables i.i.d. →
    log-normale, et non F^M).

----------------------------------------------------------------------
Pipeline complet
----------------------------------------------------------------------
  1. Import CSV (encodage et séparateur auto-détectés)  → SECTION 3
  2. Extraction des features par bloc T_b               → SECTION 4
  3. Diagnostic de stationnarité (sondes f₀)            → SECTION 9ter
  4. Classification K-Means des blocs d'excitation      → main()
  5. Réponse 1-DDL (FOH + lfilter, Smallwood récursif)  → SECTION 5
  6. Maxima par bloc + rainflow par bloc                → SECTIONS 4, 8
  7. Ajustement de la loi (L-moments analytiques)       → SECTION 6
  8. SRE / SRX / SDF analytiques depuis la DSP Welch    → SECTION 7
  9. Projection CDF longue durée + synthèse des classes → SECTIONS 9, 9quater
 10. Boucle f₀ multiprocess (shared_memory)             → SECTION 10
 11. Exports CSV + rapports HTML interactifs            → SECTIONS 11, 12

Usage:
    python mbd_simple-multi-process_v3_6.py

Dépendances : numpy, scipy, scikit-learn, pandas, tqdm, plotly, numba
              (psutil optionnel pour détection cœurs physiques)

----------------------------------------------------------------------
Références documentaires (cf. README-3.6.md pour pages détaillées)
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
# Comptage rainflow par pile (stack) compilé Numba — ASTM E1049 [4] /
# AFNOR A03-406 [5], convention AMPLITUDE (σ_a = range/2). HAS_RAINFLOW est le
# drapeau lu par le pipeline pour activer la branche SDF temporelle.
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
# Les noms des CSV de sortie sont suffixés par les paramètres du run ; un
# sidecar JSON (params_*.json) enregistre la configuration complète.
# -----------------------------------------------------------------------------

# --- Fichier CSV d'entrée ----------------------------------------------------
# Format attendu : 2 colonnes (temps en s, accélération en m/s²).
# Encodage et séparateur décimal auto-détectés (utf-8/latin-1/cp1252, ',' ou '.').
CSV_FILEPATH    = r"C:\Users\aaaaaa.csv"


CSV_SKIP_ROWS   = 10        # Lignes d'en-tête à sauter (≥ 0)
CSV_DELIMITER   = ";"       # Délimiteur — typiquement ";" (FR) ou "," (US)

# TRIM_DEBUT_S : durée (s) supprimée en début de signal AVANT tout calcul.
#                Plage 0 à quelques secondes. 0 (défaut) = aucun retrait.
#                Utile pour un signal de synthèse dont le tout début contient
#                le régime transitoire du générateur (filtrage, mise en
#                régime d'un bruit coloré) : ces premiers échantillons ne sont
#                pas représentatifs du processus et contaminent les maxima du
#                premier bloc. Inutile sur un enregistrement d'essai.
TRIM_DEBUT_S    = 0.0

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
TB  = 1.28

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
# N_CLUSTERS              : K imposé quand AUTO_SELECT_K=False.
#                            1 = mono-classe (pas de partition).
#                            Plage 1–10 ; au-delà, les statistiques par classe
#                            deviennent trop pauvres pour l'inférence (n < 40).
# AUTO_SELECT_K           : True ⇒ K est choisi automatiquement et N_CLUSTERS
#                            est ignoré :
#                              - K = 1 si le diagnostic de stationnarité conclut
#                                STATIONNAIRE (cf. AUTO_K_SELON_STATIONNARITE) ;
#                              - sinon K maximisant le score Silhouette dans
#                                K_RANGE.
#                            ATTENTION : la silhouette compare la distance d'un
#                            bloc à sa propre classe et à la classe voisine ;
#                            elle n'est pas définie pour K = 1 et découpe donc
#                            TOUJOURS le signal, même parfaitement homogène.
#                            C'est la raison d'être du lien avec le test de
#                            stationnarité ci-dessous.
#                            Si aucune feature n'est activée dans FEATURE_FLAGS,
#                            rms + kurtosis + crest_factor sont utilisées
#                            d'office (features d'énergie et de forme du bloc).
# AUTO_K_SELON_STATIONNARITE : True (défaut) ⇒ lorsque AUTO_SELECT_K=True, un
#                            verdict STATIONNAIRE impose K = 1. False ⇒
#                            silhouette seule. Sans effet si AUTO_SELECT_K=False.
# K_RANGE                 : plage testée pour K optimal (AUTO_SELECT_K=True et
#                            signal non stationnaire). range(2, 7) permet de
#                            détecter 2 régimes seulement (rupture de niveau),
#                            cas le plus fréquent en essais.
# MIN_SAMPLES_PER_CLUSTER : si une classe a moins de blocs que ce seuil, K est
#                            décrémenté automatiquement (jusqu'à K=1). Doit
#                            rester ≥ MIN_POINTS_KAPPA4 (40), sinon les fits par
#                            classe échouent.
# SYNTHESE_CLASSES        : mode de synthèse des classes, pour le SRE à la
#                            durée du signal comme pour le SRE projeté.
#                            - 'produit' (défaut) : [1] §C.10 p. 86 et [2]
#                              fig. 15 / éq. 37, P(Z_sup ≤ z) = Π_j F_j(z)^{M_j}.
#                              Chaque classe contribue au risque à hauteur de
#                              son nombre d'occurrences M_j (blocs mesurés de
#                              la classe à la durée du signal, Occ(j)·T_v/T_b
#                              en projection).
#                            - 'max' : maximum des quantiles par classe. Toujours
#                              ≤ au quantile du produit, donc NON conservatif.
#                            Sans effet si une seule classe.
#                            Une classe sans loi exploitable (moins de
#                            MIN_POINTS_KAPPA4 blocs, ajustement en échec) ne
#                            peut pas entrer dans la synthèse : le spectre est
#                            alors sous-estimé. Le cas est signalé au journal,
#                            dans le cadre des rapports HTML et dans la colonne
#                            Classes_exclues des CSV.
N_CLUSTERS                 = 1
AUTO_SELECT_K              = False
AUTO_K_SELON_STATIONNARITE = True
K_RANGE                    = range(2, 7)
MIN_SAMPLES_PER_CLUSTER    = 40
SYNTHESE_CLASSES           = 'produit'

# --- Contrôle de stationnarité (critère utilisateur, Quality Gate IID) -------
# Diagnostic calculé AVANT la boucle f₀ (quelques secondes), journalisé,
# exporté (CSV Stationnarite_*, sidecar JSON, cadre des rapports HTML).
#
# POURQUOI : l'inférence MBD par L-moments suppose que les maxima de blocs sont
# indépendants et identiquement distribués (i.i.d.). Deux causes de violation
# se traitent de façons OPPOSÉES :
#   - T_b trop court devant la mémoire de l'oscillateur τ = Q/(π·f₀) : les blocs
#     successifs se recouvrent en énergie → il faut ALLONGER T_b ;
#   - enveloppe lente du signal (changement de régime, dérive, modulation
#     d'énergie) : allonger T_b n'y change rien → il faut CLASSER les blocs.
# Le diagnostic ci-dessous sépare les deux, en plaçant les sondes f₀ au-delà de
# la zone où la mémoire de l'oscillateur pourrait à elle seule expliquer une
# corrélation (f₀ ≥ 3·Q/(π·T_b)).
#
# TROIS INDICATEURS, combinés en un verdict :
#   (a) Quality Gate IID sur les maxima de réponse de N sondes f₀ : corrélation
#       de Spearman lag-1 sur les rangs + test des suites de Wald-Wolfowitz.
#       On retient la FRACTION de sondes en échec.
#   (b) coefficient de variation (CV = écart-type / moyenne) du RMS de
#       l'excitation calculé bloc par bloc : mesure directe de la modulation
#       d'énergie du signal.
#   (c) kurtosis et asymétrie de l'excitation : qualifient le caractère
#       gaussien, qui conditionne la comparaison au SRX analytique.
#
# VERDICT : NON_STATIONNAIRE si (a) ET (b) dépassent leurs seuils ;
#           STATIONNAIRE si aucun des deux ; DOUTEUX sinon (indicateurs
#           contradictoires — la recommandation invite alors à comparer les
#           deux voies).
#
# STATIONNARITE_ENABLED     : True (défaut) / False. False ⇒ aucun diagnostic,
#                              aucun CSV Stationnarite_*, et AUTO_SELECT_K
#                              retombe sur la silhouette seule.
# STATIONNARITE_N_SONDES    : nombre de fréquences sondes log-espacées entre
#                              max(F0_MIN, 3·Q/(π·T_b)) et F0_MAX.
#                              Plage 4–20. 8 (défaut) : coût ≈ 8 réponses 1-DDL,
#                              soit quelques secondes. Augmenter à 12–16 pour un
#                              spectre très large (3 décades) ; descendre à 4–6
#                              pour un signal très long si le temps compte.
# STATIONNARITE_FRAC_ECHEC  : fraction de sondes en échec IID au-delà de laquelle
#                              l'indicateur (a) est déclaré positif.
#                              Plage 0,1–0,5. 0,3 (défaut) = un tiers des sondes.
#                              Plus BAS ⇒ plus sévère (classe plus souvent) ;
#                              plus HAUT ⇒ plus tolérant (mono-classe plus
#                              souvent). 0,5 est un réglage « ne classer que si
#                              la moitié du spectre est touchée ».
# STATIONNARITE_CV_RMS_MAX  : seuil du CV du RMS par bloc pour l'indicateur (b).
#                              Plage 0,05–0,40. 0,15 (défaut). Ordres de grandeur
#                              observés : bruit stationnaire large bande ≈ 0,05
#                              à 0,10 (le CV résiduel vient du seul échantillon-
#                              nage de T_b) ; signal à deux régimes de niveau,
#                              ou à enveloppe modulée ≥ 0,25. Un T_b court
#                              augmente mécaniquement le CV : si T_b < 0,2 s,
#                              remonter le seuil vers 0,20–0,25.
# STATIONNARITE_KURT_TOL    : tolérance |kurtosis − 3| pour qualifier
#                              « gaussien ». Plage 0,2–1,5. 0,5 (défaut).
#                              Un signal de choc ou impulsionnel monte vite
#                              au-delà de 4–5.
# STATIONNARITE_SKEW_TOL    : tolérance |asymétrie| pour « gaussien ».
#                              Plage 0,1–1,0. 0,3 (défaut).
STATIONNARITE_ENABLED     = True
STATIONNARITE_N_SONDES    = 8
STATIONNARITE_FRAC_ECHEC  = 0.3
STATIONNARITE_CV_RMS_MAX  = 0.15
STATIONNARITE_KURT_TOL    = 0.5
STATIONNARITE_SKEW_TOL    = 0.3
#
# RÉGLAGE « une seule classe si le signal est stationnaire » :
#     STATIONNARITE_ENABLED      = True   (défaut)
#     AUTO_SELECT_K              = True
#     AUTO_K_SELON_STATIONNARITE = True   (défaut)
# Un verdict STATIONNAIRE impose alors K = 1 ; sinon K est choisi dans K_RANGE
# par le score Silhouette. Le critère de stationnarité est celui défini par
# STATIONNARITE_FRAC_ECHEC (sondes IID) et STATIONNARITE_CV_RMS_MAX (énergie
# par bloc) ci-dessus : c'est là que se règle la sévérité du basculement.
# Le verdict est journalisé, exporté en CSV et rappelé dans les rapports HTML,
# avec une recommandation en clair.

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
#   - 'puissance'     : élévation directe à la puissance M
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
#                       aucune colonne ni section n'est ajoutée aux sorties.
# IID_RHO_MAX        : seuil |ρ| Spearman lag-1 acceptable. Plage 0,1–0,3 ;
#                       0,2 par défaut.
# IID_PVALUE_MIN     : seuil de p-value du test des suites. Plage 0,01–0,10 ;
#                       0,05 par défaut.
# IID_FAIL_FRAC_MAX  : fraction max de f₀ hors-tolérance pour rester 🟢 GO
#                       Plage 0,02–0,10 ; 5 % du spectre par défaut.
# IID_NOGO_FRAC      : au-delà de cette fraction de f₀ en échec ⇒ 🔴 NO-GO
#                       (dépendance généralisée). Entre les deux ⇒ 🟡 WARNING
#                       (bande isolée, souvent les basses fréquences).
# IID_MIN_N          : taille d'échantillon minimale pour qu'un test soit
#                       jugé fiable (sinon NaN, colonne ignorée du verdict).
#
# Les deux rapports HTML contiennent, en tête de page, un bloc « Contrat IID »
# qui commente ces résultats en clair (fonction commenter_quality_gate) : test
# en cause, bandes de fréquences touchées, part attribuable au hasard, cause
# probable (T_b trop court / enveloppe lente / fluctuation), conséquence sur
# le SRE et action à mener.
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
#                               (0.1, 0.1) = amorce neutre au centre du domaine ;
#                               les cas difficiles sont relevés par le retry si
#                               KAPPA4_RETRY_ENABLED=True.
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

# --- Domaine de h : unicité de la solution en L-moments ----------------------
# La forme (k, h) est identifiée par l'ÉTAPE 1 du fit, qui résout
#   τ3(k, h) = τ3̂  et  τ4(k, h) = τ4̂.
# Cette application n'est inversible de façon UNIQUE que sur h ≥ −1 : pour
# h < −1, deux couples (k, h) très différents peuvent reproduire exactement le
# même (τ3, τ4) — par exemple (k, h) = (0,30 ; −0,50) et (−0,03 ; −3,70) — mais
# donnent des queues de distribution, donc des SRE projetés, sans rapport.
# fsolve n'a aucune raison de préférer l'une ou l'autre. Hosking (routine
# PELKAP de [6]) et [2] §4.1 (h ∈ [−1, 10]) restreignent donc la recherche.
#
# KAPPA4_H_MIN               : borne basse admissible de h. −1 (défaut) =
#                               domaine d'unicité. None = aucun contrôle (la
#                               solution retenue peut alors être parasite).
#                               Une racine h < KAPPA4_H_MIN est rejetée et la
#                               résolution reprend sur KAPPA4_DOMAIN_RETRY_STARTS.
#                               Si le point (τ3, τ4) mesuré est AU-DESSUS de la
#                               courbe h = −1 (loi logistique généralisée,
#                               τ4 = (1 + 5τ3²)/6, limite haute du domaine), il
#                               n'existe aucune solution admissible : on applique
#                               le repli de [2] §4.1 (éq. 28-29) — la loi du
#                               BORD, h* = −1, au point de la courbe le plus
#                               proche du point mesuré (distance euclidienne),
#                               signalée par fail_reason='ok_h_borne' dans les
#                               exports.
#                               Intervient : kappa4_from_lmoments.
# KAPPA4_DOMAIN_RETRY_STARTS : amorces (k, h) situées dans h ∈ [−1, 0), essayées
#                               uniquement après rejet d'une racine hors domaine.
KAPPA4_H_MIN                = -1.0
KAPPA4_DOMAIN_RETRY_STARTS  = [(0.0, -0.5), (0.3, -0.5), (0.0, -0.9), (0.5, -0.9)]

# --- Amorçage de la réponse 1-DDL (transitoire de démarrage) -----------------
# SDOF_AMORCAGE              : True (défaut) / False.
#                               L'état initial « naturel » d'un filtre récursif
#                               (lfilter_zi · x[0]) est l'état d'équilibre d'une
#                               excitation CONSTANTE égale à x[0] depuis
#                               toujours. Pour un enregistrement d'essai c'est
#                               faux : l'oscillateur part d'un déplacement
#                               statique x[0]/ω₀² et libère une oscillation libre
#                               d'amplitude ≈ |x[0]| en pseudo-accélération, qui
#                               décroît en e^{−t/τ} avec τ = Q/(π·f₀) (0,64 s à
#                               5 Hz pour Q = 10). Aux basses f₀, où la réponse
#                               stationnaire est plus faible que |x[0]|, ce
#                               transitoire FIXE le maximum du premier bloc,
#                               donc le SRC et la queue des maxima.
#                               True ⇒ le filtre est d'abord amorcé sur le début
#                               du signal renversé dans le temps (même DSP,
#                               raccord continu en x[0]) ; son état final sert
#                               d'état initial. La sortie garde exactement la
#                               longueur de l'entrée (découpage en blocs
#                               inchangé) et coïncide avec l'état initial simple
#                               au-delà de quelques τ.
# SDOF_AMORCAGE_N_TAU        : durée d'amorçage en constantes de temps τ.
#                               Plage 4 à 20. 8 (défaut) ⇒ résidu e^{−8} ≈
#                               3·10⁻⁴ du transitoire. Bornée à la longueur du
#                               signal. Intervient : reponse_sdof.
SDOF_AMORCAGE               = True
SDOF_AMORCAGE_N_TAU         = 8.0

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
#                               rainflow). 10 = valeur par défaut.
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

def _taille_bloc(n, fs, Tb, min_ech=None):
    """Nombre d'échantillons par bloc : round(T_b·fs), jamais autre chose.

    La durée de bloc est une donnée de la méthode : c'est elle qui fixe le
    coefficient d'extrapolation M = T_proj / T_b ([2] §4.1). Elle ne doit donc
    pas dépendre de la longueur du signal. Quand celle-ci n'est pas un multiple
    de la taille de bloc, le signal est tronqué au multiple inférieur : on
    ignore au plus (taille − 1) échantillons en fin de signal, soit moins d'un
    bloc. (Chercher à la place un diviseur exact de la longueur du signal peut
    donner des blocs très différents de T_b — 1,65 s pour 1,28 s demandés sur
    un signal de 7 679 991 échantillons — alors que M reste calculé avec T_b.)

    Bornée par MIN_ECH_PAR_BLOC (carte EXPERT) et par la longueur du signal.
    """
    if min_ech is None:
        min_ech = MIN_ECH_PAR_BLOC
    return int(max(1, min(max(round(Tb * fs), min_ech), n)))


def extraire_caracteristiques(signal, fs, Tb_initial=0.05, feature_flags=None, min_ech=None):
    """Découpe le signal en blocs de durée T_b et calcule des features par bloc.

    Référence : [1] NF X50-144-3 §C.5 — n-échantillons par bloc T_b.
    La taille de bloc est round(T_b·fs) échantillons (cf. `_taille_bloc`) ; la
    fin du signal qui ne remplit pas un bloc entier est ignorée.

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
    taille = _taille_bloc(n, fs, Tb_initial, min_ech)
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
    formules directement utilisables par scipy.signal.lfilter.

    ÉTAT INITIAL (option SDOF_AMORCAGE, carte EXPERT). L'état `lfilter_zi · x[0]`
    est l'état d'équilibre d'une excitation CONSTANTE égale à x[0] depuis
    toujours. Pour un signal enregistré en cours d'essai, cette hypothèse est
    fausse : l'oscillateur démarre avec un déplacement statique x[0]/ω₀² au lieu
    d'un état aléatoire stationnaire, et libère une oscillation libre d'amplitude
    ≈ |x[0]| (en pseudo-accélération) qui décroît en e^{−t/τ} avec
    τ = 1/(ξ·ω₀) = Q/(π·f₀), soit 0,64 s à 5 Hz pour Q = 10. Aux basses f₀, où
    la réponse stationnaire σ_z est plus faible que |x[0]|, ce transitoire fixe
    le maximum du premier bloc, donc le SRC et la queue des maxima.

    Avec SDOF_AMORCAGE=True, le filtre est d'abord amorcé sur le début du signal
    RENVERSÉ dans le temps — même densité spectrale, raccord continu en x[0] —
    pendant SDOF_AMORCAGE_N_TAU·τ ; son état final sert d'état initial, le
    transitoire s'étant amorti pendant l'amorçage (e^{−8} ≈ 3·10⁻⁴). La sortie
    conserve exactement la longueur de l'entrée, donc le découpage en blocs est
    inchangé, et coïncide avec l'autre état initial au-delà de quelques τ.

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
    excitation = np.asarray(excitation, dtype=float)

    n_amorce = 0
    if SDOF_AMORCAGE and excitation.size > 2:
        tau = 1.0 / (xi * omega0)                      # = Q/(π·f₀)
        n_amorce = int(min(excitation.size - 1,
                           math.ceil(float(SDOF_AMORCAGE_N_TAU) * tau * fs)))
    if n_amorce >= 2:
        # Amorçage sur x[n_amorce], …, x[1] (ordre renversé), puis enchaînement
        # sur x[0], x[1], … avec l'état final : équivalent au filtrage du signal
        # préfixé par son propre début renversé, sans recopier tout le signal.
        amorce = excitation[n_amorce:0:-1]
        zi = lfilter_zi(b_c, a_c) * amorce[0]
        _, zf = lfilter(b_c, a_c, amorce, zi=zi)
        y, _ = lfilter(b_c, a_c, excitation, zi=zf)
        return y

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


# --- Contrôle du domaine de h (unicité de la solution en L-moments) ----------

def _kappa4_h_hors_domaine(h):
    """Vrai si la racine h sort du domaine d'unicité (h < KAPPA4_H_MIN).

    Voir la carte EXPERT, bloc « Domaine de h » : sous cette borne, plusieurs
    couples (k, h) reproduisent le même (τ3, τ4) tout en décrivant des queues
    de distribution différentes."""
    return (KAPPA4_H_MIN is not None and np.isfinite(h)
            and float(h) < float(KAPPA4_H_MIN) - 1e-6)


def _kappa4_au_dessus_glo(t3, t4):
    """Vrai si le point (τ3, τ4) mesuré est AU-DESSUS de la courbe h = −1.

    Cette courbe, τ4 = (1 + 5τ3²)/6, est celle de la loi logistique généralisée
    (GLO) : c'est la limite HAUTE du domaine Kappa4 restreint à h ≥ −1 dans le
    diagramme des L-moments (h croît vers le bas : h = 0 → GEV, h = 1 → Pareto
    généralisée). Au-dessus, aucun couple (k, h) admissible ne reproduit
    exactement (τ3, τ4). N'a de sens que pour KAPPA4_H_MIN = −1."""
    return (KAPPA4_H_MIN is not None and abs(float(KAPPA4_H_MIN) + 1.0) < 1e-12
            and np.isfinite(t3) and np.isfinite(t4)
            and float(t4) > (1.0 + 5.0 * float(t3) ** 2) / 6.0 + 1e-12)


def _kappa4_sur_borne_h(l1, l2, t3, t4, h_fix):
    """Loi du BORD du domaine, repli de [2] §4.1 pour un point hors domaine.

    [2] éq. 28 retient h* = argmin_{h ∈ [−1, 10]} de la distance EUCLIDIENNE,
    dans le diagramme des L-moments, entre le point mesuré (t3, t4) et la
    courbe (τ3(h), τ4(h)). Quand le point est hors domaine, ce minimum est
    atteint AU BORD : h* = KAPPA4_H_MIN, et le point d'optimisation (τ3*, τ4*)
    de l'éq. 29 est le point de cette courbe le plus proche du point mesuré.
    k* s'en déduit par τ3(k*, h*) = τ3* (éq. 32).

    Ni τ3 ni τ4 mesurés ne sont alors reproduits exactement, ce qui est
    normal : le point est en dehors du domaine atteignable par la loi. Retenir
    k* = −t3 (projection verticale sur la courbe) conserverait l'asymétrie
    mesurée mais n'est pas le minimum de distance : pour une asymétrie
    positive, il donne une queue plus légère, donc un SRE projeté plus bas.

    Pour h = −1 (GLO), Hosking & Wallis donnent τ3 = −k et τ4 = (1 + 5k²)/6 :
    le carré de la distance est un polynôme en x = τ3*, dont la dérivée
        (25/18)·x³ + [1 + (5/3)·(1/6 − t4)]·x − t3 = 0
    se résout en forme close ; parmi les racines réelles de (−1, 1), on garde
    celle de distance minimale, et k* = −x. Pour une autre borne, le minimum
    est cherché numériquement le long de la courbe paramétrée par k.

    Retour : (k, ξ, α, τ3_loi, τ4_loi) ou None si aucun point admissible /
    échelle invalide."""
    h_fix, t3, t4 = float(h_fix), float(t3), float(t4)
    if abs(h_fix + 1.0) < 1e-12:
        a, b = 1.0 / 6.0, 5.0 / 6.0            # τ4 = a + b·τ3² sur la GLO
        racines = np.roots([2.0 * b * b, 0.0, 1.0 + 2.0 * b * (a - t4), -t3])
        cand = [float(r.real) for r in racines
                if abs(r.imag) < 1e-9 and -1.0 < r.real < 1.0]
        if not cand:
            return None
        x = min(cand, key=lambda v: (v - t3) ** 2 + (a + b * v * v - t4) ** 2)
        k = -x
    else:
        from scipy.optimize import minimize_scalar
        k_hi = (min(10.0, -1.0 / h_fix - 1e-3) if h_fix < 0 else 10.0)

        def d2(kk):
            t3c, t4c = _tau3_tau4_from_kh_analytic(float(kk), h_fix)
            if not (np.isfinite(t3c) and np.isfinite(t4c)):
                return np.inf
            return (t3c - t3) ** 2 + (t4c - t4) ** 2
        # Balayage puis affinage local : la distance peut avoir plusieurs
        # minima locaux le long de la courbe.
        ks = np.linspace(-0.999, k_hi, 400)
        ds = np.array([d2(kk) for kk in ks])
        if not np.isfinite(ds).any():
            return None
        i = int(np.argmin(ds))
        try:
            opt = minimize_scalar(d2, bounds=(ks[max(i - 1, 0)],
                                              ks[min(i + 1, ks.size - 1)]),
                                  method='bounded',
                                  options={'xatol': 1e-12})
            k = float(opt.x) if opt.fun <= ds[i] else float(ks[i])
        except Exception:
            k = float(ks[i])
    ls = _fit_loc_scale(l1, l2, k, h_fix)
    if ls is None:
        return None
    tau3_c, tau4_c = _tau3_tau4_from_kh_analytic(k, h_fix)
    return k, float(ls[0]), float(ls[1]), tau3_c, tau4_c


def kappa4_from_lmoments(l1, l2, t3, t4, warm_start=None, debug=None):
    """Identification Kappa4 EN DEUX ÉTAPES à partir des L-moments.

    Référence : [2] §4.1, eq. 28-34 — loi de Hosking [6].

    C'est le cœur de l'inférence MBD. L'entrée est le quadruplet de L-moments
    (L1, L2, τ3, τ4) et non l'échantillon, parce que la structure du problème
    est séparable :

    ÉTAPE 1 — la FORME (k, h) ne dépend QUE des ratios (τ3, τ4).
        On résout le système 2×2
              τ3_loi(k, h) = τ3̂
              τ4_loi(k, h) = τ4̂
        par scipy.optimize.fsolve, amorcé sur `warm_start`
        (défaut KAPPA4_WARM_START_INITIAL).
        Trois garde-fous successifs :
          a) fsolve doit converger (ier = 1) ET le résidu² recalculé doit
             rester sous KAPPA4_RESIDUAL_TOL — sinon on rejoue sur les amorces
             KAPPA4_RETRY_WARM_STARTS (4 quadrants, maxfev étendu) ;
          b) la racine doit vérifier h ≥ KAPPA4_H_MIN (domaine d'unicité) —
             sinon elle est rejetée et les amorces KAPPA4_DOMAIN_RETRY_STARTS,
             situées dans h ∈ [−1, 0), sont ajoutées à la liste d'essais ;
          c) si (τ3̂, τ4̂) est au-dessus de la courbe h = −1, aucun couple
             admissible n'existe : on prend la loi du bord, h* = −1, au point
             de la courbe le plus proche de (τ3̂, τ4̂) en distance euclidienne
             ([2] éq. 28-29, `_kappa4_sur_borne_h`), signalée par
             fail_reason='ok_h_borne'.

    ÉTAPE 2 — la POSITION ξ et l'ÉCHELLE α se déduisent en forme close de
        (L1, L2) et de la forme obtenue ([2] eq. 33-34, fonction
        `_fit_loc_scale`) : α = L2 / (g1 − g2), ξ = L1 − α·(1 − g1)/k.
        Aucune optimisation supplémentaire n'est nécessaire, ce qui explique
        que la qualité du fit se joue entièrement à l'étape 1.

    Paramètres
    ----------
    l1, l2     : L-moments d'ordre 1 et 2 (l2 > 0 attendu).
    t3, t4     : L-skewness et L-kurtosis cibles.
    warm_start : (k0, h0) optionnel — défaut KAPPA4_WARM_START_INITIAL.
    debug      : dict optionnel, rempli en place avec la trace de résolution
                 (voir `ajuster_kappa4_pwm_analytic`).

    Retour
    ------
    dict : 'xi', 'alpha', 'k', 'h' (paramètres), 't3', 't4' (entrée),
           'tau3', 'tau4' (recalculés depuis (k*, h*) — vérification),
           'success', 'fail_reason', 'solve_attempt', 'solve_recovered',
           'loi'='kappa4'. Ne lève jamais : tout échec est rapporté par
           success=False et fail_reason.
    """
    from scipy.optimize import fsolve

    res = {'xi': np.nan, 'alpha': np.nan, 'k': np.nan, 'h': np.nan,
           't3': np.nan, 't4': np.nan, 'tau3': None, 'tau4': None,
           'success': False, 'fail_reason': 'not_run', 'loi': 'kappa4'}
    dbg = debug if isinstance(debug, dict) else None

    def _sortie(reason):
        """Termine en consignant la raison dans le dict de debug."""
        res['fail_reason'] = reason
        if dbg is not None:
            dbg['exit_reason'] = reason
        return res

    try:
        l1 = float(l1); l2 = float(l2); t3 = float(t3); t4 = float(t4)
        res['t3'], res['t4'] = t3, t4
        if dbg is not None:
            dbg['l1'] = l1
            dbg['l2'] = l2
            dbg['t3'] = t3 if np.isfinite(t3) else np.nan
            dbg['t4'] = t4 if np.isfinite(t4) else np.nan
        if (not all(np.isfinite(v) for v in (l1, l2, t3, t4))
                or abs(l2) < KAPPA4_L2_MIN_FOR_FIT):
            return _sortie('l2_or_tau_invalid')

        # ------------------------------------------------------------------
        # ÉTAPE 1 — forme (k, h) depuis (τ3, τ4) seuls.
        # ------------------------------------------------------------------
        def equations(params):
            k, h = params
            tau3_c, tau4_c = _tau3_tau4_from_kh_analytic(k, h)
            if not (np.isfinite(tau3_c) and np.isfinite(tau4_c)):
                return [1e6, 1e6]
            return [tau3_c - t3, tau4_c - t4]

        x0_initial = (tuple(warm_start)
                      if (warm_start is not None and all(np.isfinite(warm_start)))
                      else tuple(KAPPA4_WARM_START_INITIAL))
        if dbg is not None:
            dbg['warm_start'] = (float(x0_initial[0]), float(x0_initial[1]))

        # 200·(N+1) = budget maxfev par défaut de fsolve pour N = 2 inconnues.
        attempts = [(x0_initial, int(200 * (2 + 1)))]
        if KAPPA4_RETRY_ENABLED:
            attempts += [(tuple(x0), int(KAPPA4_RETRY_MAXFEV))
                         for x0 in KAPPA4_RETRY_WARM_STARTS]

        sol, ier, res_norm2 = None, None, np.inf
        last_exc_name, retry_idx = None, -1
        h_hors_domaine = False
        for idx, (x0, maxfev) in enumerate(attempts):
            try:
                sol_t, _info, ier_t, _msg = fsolve(
                    equations, x0, full_output=True,
                    xtol=float(KAPPA4_ANALYTIC_XTOL), maxfev=int(maxfev))
            except Exception as exc:
                last_exc_name = type(exc).__name__
                continue
            if ier_t != 1:
                if ier is None:          # garde la 1ʳᵉ trace d'échec
                    ier = ier_t
                continue
            r_chk = equations((float(sol_t[0]), float(sol_t[1])))
            rn2 = ((r_chk[0] ** 2 + r_chk[1] ** 2)
                   if all(np.isfinite(r_chk)) else np.inf)
            if not np.isfinite(rn2) or rn2 > KAPPA4_RESIDUAL_TOL:
                ier, res_norm2 = ier_t, rn2
                continue
            # Garde-fou (b) : racine hors du domaine d'unicité de h.
            if _kappa4_h_hors_domaine(sol_t[1]):
                if not h_hors_domaine:
                    h_hors_domaine = True
                    attempts.extend((tuple(x0d), int(KAPPA4_RETRY_MAXFEV))
                                    for x0d in KAPPA4_DOMAIN_RETRY_STARTS)
                continue
            sol, ier, res_norm2, retry_idx = sol_t, ier_t, rn2, idx
            break

        if sol is None:
            # Garde-fou (c) : point hors domaine ⇒ loi du bord h = KAPPA4_H_MIN.
            if h_hors_domaine or _kappa4_au_dessus_glo(t3, t4):
                borne = _kappa4_sur_borne_h(l1, l2, t3, t4, KAPPA4_H_MIN)
                if borne is not None:
                    k_b, xi_b, al_b, t3c_b, t4c_b = borne
                    res.update({'xi': xi_b, 'alpha': al_b, 'k': k_b,
                                'h': float(KAPPA4_H_MIN), 'tau3': t3c_b,
                                'tau4': t4c_b, 'success': True,
                                'h_borne': True, 'fail_reason': 'ok_h_borne'})
                    if dbg is not None:
                        dbg.update({'fsolve_sol_k': float(k_b),
                                    'fsolve_sol_h': float(KAPPA4_H_MIN),
                                    'tau3_calc': float(t3c_b),
                                    'tau4_calc': float(t4c_b),
                                    'xi': float(xi_b), 'alpha': float(al_b),
                                    'exit_reason': 'ok_h_borne'})
                    return res
                return _sortie('h_out_of_domain')
            if last_exc_name is not None and ier is None:
                return _sortie(f'fsolve_exception:{last_exc_name}')
            if ier is None:
                return _sortie('fsolve_no_attempt')
            if not np.isfinite(res_norm2) or res_norm2 > KAPPA4_RESIDUAL_TOL:
                if dbg is not None:
                    dbg['fsolve_res_norm2'] = (float(res_norm2)
                                               if np.isfinite(res_norm2) else np.inf)
                return _sortie('residual_too_large')
            if dbg is not None:
                dbg['fsolve_ier'] = int(ier)
            return _sortie(f'fsolve_not_converged(ier={ier})')

        k_star, h_star = float(sol[0]), float(sol[1])
        res['solve_attempt']   = retry_idx
        res['solve_recovered'] = (retry_idx > 0)
        if dbg is not None:
            r_chk = equations((k_star, h_star))
            dbg.update({'fsolve_ier': int(ier),
                        'fsolve_sol_k': k_star, 'fsolve_sol_h': h_star,
                        'fsolve_res1': (float(r_chk[0])
                                        if np.isfinite(r_chk[0]) else np.nan),
                        'fsolve_res2': (float(r_chk[1])
                                        if np.isfinite(r_chk[1]) else np.nan),
                        'fsolve_res_norm2': (float(res_norm2)
                                             if np.isfinite(res_norm2) else np.inf)})

        # ------------------------------------------------------------------
        # ÉTAPE 2 — position ξ et échelle α depuis (L1, L2) et la forme.
        # ------------------------------------------------------------------
        if dbg is not None:
            loc_scale, ls_diag = _fit_loc_scale(l1, l2, k_star, h_star,
                                                return_diag=True)
            for key in ('g1', 'g2', 'g1_minus_g2'):
                if key in ls_diag:
                    dbg[key] = ls_diag[key]
        else:
            loc_scale = _fit_loc_scale(l1, l2, k_star, h_star)
        if loc_scale is None:
            return _sortie('loc_scale_invalid')
        xi_star, alpha_star = loc_scale

        tau3_calc, tau4_calc = _tau3_tau4_from_kh_analytic(k_star, h_star)
        res.update({'xi': float(xi_star), 'alpha': float(alpha_star),
                    'k': k_star, 'h': h_star,
                    'tau3': tau3_calc, 'tau4': tau4_calc, 'success': True,
                    'fail_reason': 'ok_retry' if res['solve_recovered'] else 'ok'})
        if dbg is not None:
            dbg.update({'xi': float(xi_star), 'alpha': float(alpha_star),
                        'tau3_calc': float(tau3_calc),
                        'tau4_calc': float(tau4_calc),
                        'exit_reason': 'ok'})
    except Exception as exc:
        return _sortie(f'outer_exception:{type(exc).__name__}')
    return res


def ajuster_kappa4_pwm_analytic(data, warm_start=None):
    """Ajustement Kappa4 d'un ÉCHANTILLON : L-moments puis identification.

    Enchaîne simplement :
        1. `calculer_lmoments(data)`  → (L1, L2, τ3, τ4) empiriques,
           estimateurs à faible variance (probability-weighted moments) ;
        2. `kappa4_from_lmoments(...)` → (ξ, α, k, h) en deux étapes.

    L'identité `ajuster_kappa4_pwm_analytic(x)` ≡
    `kappa4_from_lmoments(*calculer_lmoments(x))` est garantie : toute la
    logique d'inférence est dans la seconde fonction.

    Retour
    ------
    dict : mêmes clés que `kappa4_from_lmoments`, plus
        '_t_ms'   : durée du fit en ms (ajoutée par `_fit_analytic`)
        '_debug'  : trace de résolution si KAPPA4_DEBUG_ANALYTIC=True.
    """
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
    except Exception as exc:
        res = {'xi': np.nan, 'alpha': np.nan, 'k': np.nan, 'h': np.nan,
               't3': np.nan, 't4': np.nan, 'tau3': None, 'tau4': None,
               'success': False, 'loi': 'kappa4',
               'fail_reason': f'lmoments_exception:{type(exc).__name__}'}
        if debug_on:
            dbg['exit_reason'] = res['fail_reason']
            res['_debug'] = dbg
        return res

    res = kappa4_from_lmoments(l1, l2, t3, t4, warm_start=warm_start, debug=dbg)
    if debug_on:
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
        # Même découpage que extraire_caracteristiques.
        taille_bloc = _taille_bloc(n_total, fs, Tb)
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
# Brique de validation post-traitement. Aucune dépendance
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
    des extrêmes de la loi ajustée (méthode 1 de la Quality Gate).
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
    runs, statistique Z par approximation normale (méthode 2 de la Quality Gate).
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
    """Quality Gate IID — test d'indépendance d'une série de blocs.

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
    """Statut global selon la fraction de f₀ hors-tolérance.
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
        # 'tested' et 'n' servent au commentaire des rapports HTML
        # (commenter_quality_gate) : fréquences réellement testées, taille N.
        tested = np.array([bool(r['iid'].get(branch, {}).get('tested', False))
                           for r in res_ok], dtype=bool)
        n_b = np.array([r['iid'].get(branch, {}).get('n', 0)
                        for r in res_ok], dtype=int)
        return {'rho': rho, 'pvalue': pval, 'fail': fail,
                'tested': tested, 'n': n_b}

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


# --- Commentaire en clair de la Quality Gate (bloc des rapports HTML) --------

def _iid_taux_hasard(n_blocs):
    """Fraction de f₀ en échec attendue sous indépendance PARFAITE des blocs.

    Les deux tests ont un taux de fausse alarme propre : le test des suites
    rejette à tort environ IID_PVALUE_MIN des séries indépendantes, et |ρ|
    dépasse IID_RHO_MAX par simple fluctuation quand N est petit (écart-type
    de ρ ≈ 1/√N). Ce taux est le niveau de référence auquel comparer la
    fraction observée. Il est obtenu en appliquant la porte elle-même à du
    bruit blanc de même longueur N (forme asymptotique au-delà de 2000 blocs,
    où seule la fausse alarme du test des suites subsiste). Il dépend de N :
    ≈ 22 % pour N = 40, ≈ 9 % pour N = 90, 4 à 6 % au-delà de 300 blocs.
    Retourne NaN si N est trop petit pour être testé."""
    n_blocs = int(n_blocs)
    if n_blocs < max(int(IID_MIN_N), 3):
        return np.nan
    if n_blocs > 2000:
        return float(min(1.0, IID_PVALUE_MIN
                         + 2.0 * sp_norm.sf(IID_RHO_MAX * math.sqrt(n_blocs))))
    rng = np.random.default_rng(RANDOM_SEED)
    # 4000 à 10000 tirages : erreur d'échantillonnage ≤ 0,4 point sur le taux.
    n_tirages = int(min(10000, max(4000, 4_000_000 // n_blocs)))
    g = quality_gate_iid(rng.standard_normal((n_blocs, n_tirages)))
    return float(np.mean(g['fail']))


def _iid_bandes(f0, fail):
    """Regroupe les f₀ en échec en bandes de fréquences contiguës.
    Retour : liste de (f_début, f_fin, nombre de f₀)."""
    bandes, debut = [], None
    for i, bad in enumerate(fail):
        if bad and debut is None:
            debut = i
        if debut is not None and (not bad or i == len(fail) - 1):
            fin = i if bad else i - 1
            bandes.append((float(f0[debut]), float(f0[fin]), fin - debut + 1))
            debut = None
    return bandes


def commenter_quality_gate(iid_diag, results, Tb, Q, duree_mesure=None,
                           stat_diag=None, tb_effectif=None):
    """Explication en clair des résultats de la Quality Gate IID.

    Produit le contenu du bloc « Contrat IID » placé en tête des rapports
    HTML : le verdict seul (GO / WARNING / NO-GO) ne dit ni QUEL test échoue,
    ni OÙ, ni POURQUOI, ni ce que cela change pour le SRE. Le commentaire est
    construit à partir des résultats du run, rubrique par rubrique :

      - verdict et règle de décision ;
      - ce que le contrat vérifie (hypothèse i.i.d. de [2] §4.1, éq. 35-36) ;
      - données testées (N blocs, durée de bloc EFFECTIVE) ;
      - résultat test par test, pour les maxima (SRE) et les dommages (SDF) ;
      - localisation en fréquence des échecs ;
      - part attribuable au hasard (cf. `_iid_taux_hasard`) ;
      - cause probable : mémoire de l'oscillateur (τ = Q/(π·f₀) > T_b/3),
        enveloppe lente du signal, ou simple fluctuation statistique ;
      - conséquence sur les résultats et action recommandée ;
      - détail par classe et sondes du contrôle de stationnarité.

    Paramètres
    ----------
    iid_diag     : dict renvoyé par `agreger_quality_gate`.
    results      : liste des résultats par f₀ (pour le détail par classe).
    Tb, Q        : durée de bloc demandée (s) et surtension.
    duree_mesure : durée du signal (s) — sert à chiffrer le nombre de blocs
                   qu'aurait un T_b allongé. Optionnel.
    stat_diag    : dict de `detecter_non_stationnarite`, optionnel.
    tb_effectif  : durée de bloc effective (s), taille entière en échantillons
                   divisée par fs. Défaut : Tb.

    Retour : dict ordonné {rubrique: texte HTML}, None si rien à commenter.
    Le dict porte aussi la clé '_status' (lue pour la couleur du bloc).
    """
    if not iid_diag:
        return None
    f0 = np.asarray(iid_diag.get('f0', []), dtype=float)
    if f0.size == 0:
        return None
    order = np.argsort(f0)
    f0 = f0[order]

    def _branche(nom):
        d = iid_diag.get(nom, {})
        if np.asarray(d.get('fail', [])).size != order.size or 'tested' not in d:
            return None
        return {k: np.asarray(d[k])[order]
                for k in ('rho', 'pvalue', 'fail', 'tested', 'n')}

    sre, sdf = _branche('sre'), _branche('sdf')
    if sre is None:
        return None
    fail_any   = sre['fail'].copy()
    tested_any = sre['tested'].copy()
    if sdf is not None:
        fail_any   |= sdf['fail']
        tested_any |= sdf['tested']
    n_f0, n_fail, n_test = f0.size, int(fail_any.sum()), int(tested_any.sum())
    status    = iid_diag.get('status', 'GO')
    frac_fail = float(iid_diag.get('frac_fail', 0.0))

    def _pc(x):
        return f"{100.0 * x:.1f} %"

    def _pv(nom, x):
        return f"{nom} &lt; 0.0001" if x < 1e-4 else f"{nom} = {x:.4f}"

    def _n_independantes(masque):
        """Nombre approximatif de fréquences INDÉPENDANTES dans un ensemble de
        f₀ : les réponses à des f₀ voisines sont corrélées sur la largeur de
        bande f₀/Q, soit environ Q·ln(f_max/f_min) bandes distinctes."""
        f_t = f0[masque]
        if f_t.size == 0:
            return 1.0
        return float(np.clip(float(Q) * math.log(max(f_t.max() / f_t.min(),
                                                     1.0)), 1.0, f_t.size))

    # --- Données testées : N blocs et durée de bloc effective ---------------
    n_ref = sre['n'][sre['tested']] if sre['tested'].any() else sre['n']
    n_blocs = int(np.median(n_ref)) if n_ref.size else 0
    tb_eff = float(tb_effectif) if tb_effectif else float(Tb)
    donnees = (f"N = {n_blocs} blocs par fréquence, dans l'ordre du temps ; "
               f"{n_test} fréquences testées sur {n_f0}")
    if n_test < n_f0:
        donnees += (f" ({n_f0 - n_test} non testées : moins de {IID_MIN_N} "
                    "blocs ou série constante)")
    donnees += f". Durée de bloc : {tb_eff:.4g} s"
    if abs(tb_eff / float(Tb) - 1.0) > 0.01:
        donnees += (f" — <b>différente du T_b demandé ({Tb:g} s)</b> : la "
                    "taille de bloc est bornée par MIN_ECH_PAR_BLOC ou par la "
                    "longueur du signal. La zone « mémoire » ci-dessous est "
                    "évaluée avec la durée effective.")
    else:
        donnees += "."

    # --- Résultat test par test ----------------------------------------------
    def _detail(b, nom):
        t = b['tested']
        if not t.any():
            return f"<b>{nom}</b> : aucune fréquence testable."
        rho, pv = b['rho'], b['pvalue']
        e_rho = t & (np.abs(rho) > IID_RHO_MAX)
        e_run = t & (pv < IID_PVALUE_MIN)
        i_r = int(np.nanargmax(np.where(t, np.abs(rho), np.nan)))
        i_p = int(np.nanargmin(np.where(t, pv, np.nan)))
        txt = (f"<b>{nom}</b> : {int(b['fail'].sum())} f₀ en échec sur "
               f"{int(t.sum())} testées — {int((e_rho & ~e_run).sum())} par la "
               f"corrélation seule, {int((e_run & ~e_rho).sum())} par le test "
               f"des suites seul, {int((e_rho & e_run).sum())} par les deux. "
               f"|ρ| médian = {np.nanmedian(np.abs(rho[t])):.3f}, maximum = "
               f"{abs(rho[i_r]):.3f} à {f0[i_r]:.1f} Hz (seuil {IID_RHO_MAX}). "
               f"{_pv('p-value minimale', pv[i_p])} à {f0[i_p]:.1f} Hz "
               f"(seuil {IID_PVALUE_MIN}).")
        bad = b['fail']
        if bad.any():
            n_bad, n_pos = int(bad.sum()), int((rho[bad] > 0).sum())
            if n_pos >= 0.75 * n_bad:
                sens = ("persistance : un bloc fort est suivi d'un bloc fort, "
                        "signature d'une mémoire ou d'une enveloppe lente")
            elif n_pos <= 0.25 * n_bad:
                sens = ("alternance : blocs forts et faibles se succèdent, "
                        "signature d'une composante périodique de période "
                        "voisine de 2·T_b")
            else:
                sens = ("pas de signe dominant, comme on l'attend d'écarts "
                        "dus au hasard")
            txt += (f" Sur les f₀ en échec, ρ est positif {n_pos} fois sur "
                    f"{n_bad} — {sens}.")
        return txt

    details = _detail(sre, "Maxima par bloc (SRE)")
    if sdf is not None:
        details += "<br>" + _detail(sdf, "Dommages par bloc (SDF)")

    # --- Localisation en fréquence -------------------------------------------
    bandes = _iid_bandes(f0, fail_any)
    if not bandes:
        localisation = "Aucune fréquence en échec."
    else:
        larges  = [b for b in bandes if b[2] > 1]
        isolees = [b for b in bandes if b[2] == 1]
        morceaux = [f"{a:g}–{b:g} Hz ({n} f₀)" for a, b, n in larges[:12]]
        if len(larges) > 12:
            morceaux.append(f"… et {len(larges) - 12} autres bandes")
        if isolees:
            liste = ", ".join(f"{a:g}" for a, _, _ in isolees[:15])
            if len(isolees) > 15:
                liste += ", …"
            morceaux.append(f"{len(isolees)} fréquence(s) isolée(s) : "
                            f"{liste} Hz")
        localisation = (" ; ".join(morceaux) + ". Des f₀ voisines donnent des "
                        "réponses presque identiques : un échec touche donc "
                        "normalement plusieurs f₀ consécutives, et une bande "
                        "compte pour UN événement, pas pour autant d'échecs "
                        "que de f₀.")

    # --- Zone « mémoire de l'oscillateur » ------------------------------------
    # τ = Q/(π·f₀) > T_b/3 ⇔ f₀ < 3·Q/(π·T_b) : la réponse d'un bloc déborde
    # sur le suivant. Même critère que les sondes du contrôle de stationnarité.
    # Ce n'est pas une frontière nette (corrélation résiduelle e^{−T_b/τ} ≈ 5 %
    # à f_mem, négligeable à 2·f_mem) : si la bande en échec se prolonge au-delà
    # de f_mem, la zone est étendue jusqu'à sa fin, sans dépasser 2·f_mem.
    f_mem  = 3.0 * float(Q) / (math.pi * tb_eff)
    bas    = tested_any & (f0 < f_mem)
    n_bas, n_fail_bas = int(bas.sum()), int((fail_any & bas).sum())
    memoire = n_bas >= 1 and n_fail_bas >= max(1.0, 0.5 * n_bas)
    f_lim = f_mem
    if memoire:
        for a, b, _n in bandes:
            if a < f_mem <= b:
                f_lim = min(b, 2.0 * f_mem)
        lim_txt = f"{f_lim:.1f}" if f_lim == f_mem else f"{f_lim:.4g}"
        haut = tested_any & (f0 > f_lim)
        ou_haut = f"au-dessus de {lim_txt} Hz"
    else:
        # Zone mémoire non incriminée : le hasard est évalué sur tout le spectre.
        haut = tested_any.copy()
        ou_haut = "sur l'ensemble du spectre"
    n_haut, n_fail_haut = int(haut.sum()), int((fail_any & haut).sum())
    frac_haut = (n_fail_haut / n_haut) if n_haut else 0.0

    # --- Part attribuable au hasard ------------------------------------------
    # Deux branches testées ⇒ la fausse alarme de l'union est au plus celle de
    # deux tests indépendants. La dispersion σ tient compte du nombre de
    # fréquences réellement indépendantes (cf. _n_independantes).
    def _hasard_ref(n):
        p = _iid_taux_hasard(n)
        return p if (sdf is None or not np.isfinite(p)) else 1.0 - (1.0 - p) ** 2

    p0_any = _hasard_ref(n_blocs)
    hasard_ok = False
    if np.isfinite(p0_any) and n_haut > 0:
        n_ind  = _n_independantes(haut)
        sigma  = math.sqrt(p0_any * (1.0 - p0_any) / n_ind)
        hasard_ok = frac_haut <= p0_any + 2.0 * sigma
        hasard = (
            f"Avec N = {n_blocs} blocs, des blocs parfaitement indépendants "
            f"produisent à eux seuls {'au plus ' if sdf is not None else ''}"
            f"≈ {_pc(p0_any)} de f₀ en échec (porte appliquée à du bruit "
            "blanc de même longueur : le test des suites, au seuil "
            f"{IID_PVALUE_MIN}, rejette à tort environ une série "
            "indépendante sur vingt). Les réponses à des f₀ voisines étant "
            "très corrélées, le spectre ne contient qu'environ "
            f"Q·ln(f_max/f_min) ≈ {n_ind:.0f} fréquences indépendantes : "
            f"cette fraction fluctue de ± {_pc(sigma)} (un écart-type) d'un "
            f"signal à l'autre. <b>Observé {ou_haut} : "
            f"{_pc(frac_haut)}</b> ({n_fail_haut} f₀ sur {n_haut}) — "
            + ("<b>compatible avec le seul hasard</b> (moins de deux "
               "écarts-types au-dessus du niveau attendu)."
               if hasard_ok else
               "<b>au-dessus de ce que le hasard explique</b> (plus de deux "
               "écarts-types au-dessus du niveau attendu).")
            + " Ordres de grandeur indicatifs.")
        if (hasard_ok and status == 'WARNING'
                and p0_any + sigma > IID_FAIL_FRAC_MAX):
            hasard += (f" À noter : le seuil GO ({_pc(IID_FAIL_FRAC_MAX)}) est "
                       "du même ordre que ce niveau de fausse alarme ; un "
                       "statut WARNING peut donc apparaître sur un signal "
                       "parfaitement indépendant.")
    else:
        hasard = ("Non évaluée (trop peu de blocs, ou aucune fréquence hors "
                  "de la zone mémoire).")

    # --- Détail par classe (l'ajustement est fait classe par classe) ---------
    par_classe = {}
    for r in results:
        if not r.get('success') or 'iid' not in r:
            continue
        for pc in r['iid'].get('per_class', []):
            d_sre, d_sdf = pc.get('sre', {}), pc.get('sdf', {})
            acc = par_classe.setdefault(pc.get('classe'), [0, 0, []])
            if d_sre.get('tested') or d_sdf.get('tested'):
                acc[0] += 1
                acc[1] += int(bool(d_sre.get('fail') or d_sdf.get('fail')))
            acc[2].append(pc.get('n', 0))
    multi = len(par_classe) > 1
    classes_saines, lignes_classes = False, []
    if multi:
        n_ind_tot, saines = _n_independantes(tested_any), []
        for ci in sorted(par_classe):
            n_t, n_e, ns = par_classe[ci]
            n_c = int(np.median(ns)) if ns else 0
            p0_c = _hasard_ref(n_c)
            if not n_t or not np.isfinite(p0_c):
                lignes_classes.append(f"classe {ci} (N = {n_c} blocs) : non "
                                      f"testée, moins de {IID_MIN_N} blocs")
                continue
            sig_c = math.sqrt(p0_c * (1.0 - p0_c) / n_ind_tot)
            saine = (n_e / n_t) <= p0_c + 2.0 * sig_c
            saines.append(saine)
            lignes_classes.append(
                f"classe {ci} (N = {n_c} blocs) : {n_e} f₀ en échec sur {n_t} "
                f"({_pc(n_e / n_t)}) pour ≈ {_pc(p0_c)} attendus du seul "
                f"hasard — <b>{'au niveau du hasard' if saine else 'au-dessus du hasard'}</b>")
        classes_saines = bool(saines) and all(saines)

    # --- Cause probable, conséquence, action ---------------------------------
    # Trois causes distinguées, parce qu'elles appellent des actions OPPOSÉES :
    # mémoire de l'oscillateur (allonger T_b), enveloppe lente du signal
    # (classer les blocs), fluctuation statistique (ne rien faire). Quand
    # l'enveloppe lente est en cause, elle explique aussi les échecs des basses
    # fréquences : la mémoire n'est alors citée qu'en second.
    non_stat  = (stat_diag is not None
                 and stat_diag.get('verdict') == 'NON_STATIONNAIRE')
    enveloppe = non_stat and n_fail_haut > 0 and not hasard_ok
    note_classes = (
        "À l'intérieur de chaque classe, la fraction en échec est au niveau "
        "du hasard (rubrique « Par classe ») : l'ajustement classe par classe "
        "et la synthèse des classes ne sont pas remis en cause par le verdict "
        "global, qui porte sur la série de tous les blocs.")
    causes, consequences, actions = [], [], []
    if memoire:
        tb_reco = 3.0 * float(Q) / (math.pi * float(f0[tested_any].min()))
        n_reco = (f" (soit environ {duree_mesure / tb_reco:.0f} blocs au lieu "
                  f"de {n_blocs})" if duree_mesure else "")
    if memoire and not enveloppe:
        causes.append(
            "<b>Mémoire de l'oscillateur</b> : "
            + (f"la seule f₀ située sous {f_mem:.1f} Hz est en échec"
               if n_bas == 1 else
               f"{n_fail_bas} des {n_bas} f₀ situées sous {f_mem:.1f} Hz "
               "sont en échec")
            + (f", et la bande en échec se prolonge jusqu'à {lim_txt} Hz "
               "(le critère n'est pas une frontière nette, l'effet s'atténue "
               "progressivement)" if f_lim > f_mem else "")
            + f". Sous {f_mem:.1f} Hz, la constante de temps de l'oscillateur "
            "τ = Q/(π·f₀) dépasse T_b/3 : la réponse d'un bloc déborde sur le "
            "suivant et les maxima successifs se ressemblent. C'est un effet "
            "du réglage de T_b, pas une propriété du signal.")
        consequences.append(
            f"Jusqu'à {lim_txt} Hz, l'échantillon contient moins de maxima "
            "réellement indépendants que N : l'ajustement de la loi y est "
            "moins précis. Les extrêmes arrivant groupés, la projection "
            "F^M (M tirages indépendants) tend à majorer le long terme.")
        actions.append(
            f"Porter T_b à au moins {tb_reco:.2f} s = 3·Q/(π·f₀_min){n_reco}, "
            f"ou relever F0_MIN au-dessus de {lim_txt} Hz.")
    if n_fail_haut == 0:
        if memoire and n_haut:
            causes.append(f"Aucun échec au-dessus de {lim_txt} Hz.")
    elif hasard_ok:
        causes.append(
            ("Sur le reste du spectre, " if memoire else "")
            + "<b>fluctuation statistique</b> : la fraction en échec est au "
            "niveau produit par des blocs indépendants ; il n'y a pas de "
            "cause physique à rechercher.")
        consequences.append(
            "Les échecs restants sont au niveau du hasard : ils n'appellent "
            "pas de réserve sur les valeurs de SRE"
            + (" et de SDF" if sdf is not None else "") + ".")
        actions.append(
            "Pour lever le doute sur une bande précise, relancer avec un T_b "
            "légèrement différent : un échec dû au hasard ne se reproduit "
            "pas aux mêmes fréquences.")
    elif enveloppe:
        causes.append(
            "<b>Enveloppe lente du signal</b> : le contrôle de stationnarité "
            "conclut NON_STATIONNAIRE (changement de régime, dérive ou "
            "modulation d'énergie). Les blocs voisins partagent le même "
            "niveau d'énergie, ce qui corrèle leurs maxima à toutes les "
            "fréquences. Allonger T_b ne corrige pas cet effet."
            + (f" Les f₀ situées sous {f_mem:.1f} Hz cumulent en outre "
               "l'effet de mémoire de l'oscillateur (τ = Q/(π·f₀) &gt; T_b/3)."
               if memoire else ""))
        if not multi:
            consequences.append(
                "Une loi unique ajustée sur tous les blocs mélange des "
                "régimes d'amplitude différents ; sa queue est tirée vers le "
                "bas par les blocs faibles et le SRE projeté est sous-estimé.")
            actions.append(
                "Classer les blocs : AUTO_SELECT_K=True (features rms, "
                "kurtosis, crest_factor) avec SYNTHESE_CLASSES='produit', puis "
                "vérifier la porte IID classe par classe (rubrique « Par "
                "classe » du rapport et fichier IID_QualityGate_*.csv).")
        elif classes_saines:
            consequences.append(
                "Le classement des blocs répond à cette non-stationnarité. "
                + note_classes)
            actions.append(
                "Aucune action sur le classement. Lire la porte IID classe "
                "par classe plutôt que sur la série de tous les blocs.")
        else:
            consequences.append(
                "Malgré le classement, au moins une classe reste au-dessus du "
                "niveau du hasard (rubrique « Par classe ») : elle mélange "
                "encore des régimes différents, et la loi ajustée sur cette "
                "classe sous-estime le long terme.")
            actions.append(
                "Augmenter le nombre de classes (N_CLUSTERS, ou K_RANGE avec "
                "AUTO_SELECT_K=True) ou revoir les features de classement, "
                "puis contrôler à nouveau classe par classe.")
        if memoire:
            actions.append(
                f"Si les f₀ situées sous {f_mem:.1f} Hz restent en échec une "
                "fois les classes en place (lecture classe par classe) : "
                f"porter T_b à au moins {tb_reco:.2f} s = "
                f"3·Q/(π·f₀_min){n_reco}.")
    else:
        causes.append(
            ("Sur le reste du spectre, " if memoire else "")
            + "<b>dépendance localisée</b> : les échecs dépassent le niveau du "
            "hasard sans que le contrôle de stationnarité conclue à un "
            "changement de régime. Causes usuelles : une composante "
            "périodique lente ou une raie dont la période est comparable à "
            "T_b, un battement entre raies voisines, ou un évènement "
            "transitoire isolé qui marque plusieurs blocs consécutifs.")
        consequences.append(
            "Sur les bandes listées, l'hypothèse d'indépendance n'est pas "
            "établie : le SRE à la durée du signal reste représentatif de la "
            "mesure, mais le SRE projeté y est à lire avec réserve.")
        actions.append(
            "Examiner le graphe 5 du rapport de diagnostic sur ces bandes ; "
            "relancer avec un T_b différent (une raie cesse d'être "
            "synchrone, un régime lent persiste) ; si l'échec persiste, "
            "essayer le classement des blocs (AUTO_SELECT_K=True).")
        if multi and classes_saines:
            consequences.append(note_classes)
    classes_ok = multi and classes_saines
    if status == 'NO-GO' and not classes_ok:
        if memoire and not enveloppe and (n_fail_haut == 0 or hasard_ok):
            # Seule la mémoire est en cause : la réserve se limite à sa zone.
            consequences.insert(0, "<b>Résultats à considérer comme non "
                                   f"fiables jusqu'à {lim_txt} Hz</b> tant "
                                   "que T_b n'est pas allongé ; pas de "
                                   "réserve au-dessus.")
        else:
            consequences.insert(0, "<b>Résultats de ce calcul à considérer "
                                   "comme non fiables</b> tant que la cause "
                                   "n'est pas traitée.")
    if n_fail:
        consequences.append("Les f₀ en échec sont repérées par la mention "
                            "« indépendance IID : reduced » dans l'infobulle "
                            "des courbes SRE.")
    else:
        consequences = ["Aucune réserve : l'hypothèse d'indépendance est "
                        "tenue à toutes les fréquences testées."]
        actions = ["Aucune."]

    _ico = {'GO': '🟢', 'WARNING': '🟡', 'NO-GO': '🔴'}.get(status, '')
    _sens = {
        'GO': "L'hypothèse d'indépendance des blocs est tenue sur l'ensemble "
              "du spectre : l'ajustement par L-moments et la projection "
              "reposent sur une base saine.",
        'WARNING': "L'indépendance des blocs n'est pas établie sur une partie "
                   "du spectre. Les résultats sont conservés ; les rubriques "
                   "ci-dessous indiquent si ces échecs appellent une réserve.",
        'NO-GO': "La dépendance entre blocs est généralisée sur la série de "
                 "tous les blocs. "
                 + ("Les classes, sur lesquelles porte l'ajustement, sont "
                    "saines : voir « Par classe »." if classes_ok else
                    "L'inférence par L-moments n'est pas fiable en l'état."),
    }.get(status, '')
    com = {
        '_status': status,
        'Verdict': (
            f"<b>{_ico} {status}</b> — {n_fail} f₀ en échec sur {n_f0} "
            f"({_pc(frac_fail)}). Règle : GO jusqu'à {_pc(IID_FAIL_FRAC_MAX)}, "
            f"NO-GO à partir de {_pc(IID_NOGO_FRAC)}, WARNING entre les deux. "
            f"{_sens}"),
        'Ce que le contrat vérifie': (
            "La méthode MBD suppose que les maxima par bloc (et les dommages "
            "par bloc) sont indépendants et identiquement distribués : c'est "
            "cette hypothèse qui autorise l'ajustement de la loi par "
            "L-moments et la projection F_Zsup = F_Zmax^M ([2] §4.1, éq. 35 "
            "et 36). Deux tests sont appliqués à chaque fréquence, sur la "
            "série des blocs prise dans l'ordre du temps : (1) corrélation de "
            "Spearman entre un bloc et le suivant, calculée sur les rangs — "
            f"échec si |ρ| &gt; {IID_RHO_MAX} ; (2) test des suites de "
            "Wald-Wolfowitz par rapport à la médiane — échec si p-value "
            f"&lt; {IID_PVALUE_MIN}. Une fréquence est en échec dès que l'un "
            "des deux tests échoue."),
        'Données testées': donnees,
        'Résultat test par test': details,
        'Localisation des échecs': localisation,
        'Part attribuable au hasard': hasard,
        'Cause probable': "<br>".join(causes) if causes else "Sans objet.",
        'Conséquence sur les résultats': "<br>".join(consequences),
        'Action recommandée': "<br>".join(actions),
    }

    if multi:
        com['Par classe'] = (
            "Le verdict porte sur la série de TOUS les blocs, alors que la "
            "loi est ajustée classe par classe : c'est l'indépendance à "
            "l'intérieur de chaque classe qui conditionne l'ajustement. Le "
            "niveau du hasard est plus élevé pour une classe peu peuplée "
            "(|ρ| dépasse plus facilement son seuil quand N est petit).<br>"
            + "<br>".join(lignes_classes) + ".<br>Une série globale en échec "
            "avec des classes au niveau du hasard indique que le classement "
            "a absorbé le changement de régime.")

    # --- Sondes du contrôle de stationnarité (mêmes tests, f₀ choisies) ------
    if stat_diag is not None and stat_diag.get('sondes'):
        sondes  = stat_diag['sondes']
        testees = [s for s in sondes if s.get('tested')]
        echecs  = [s for s in testees if s.get('fail')]
        txt = (f"Le contrôle de stationnarité applique les deux mêmes tests à "
               f"{len(sondes)} fréquences sondes, toutes placées au-dessus de "
               "la zone mémoire : un échec y désigne le signal, pas T_b. "
               f"{len(echecs)} sonde(s) en échec sur {len(testees)}")
        if testees:
            txt += (f" ({_pc(len(echecs) / len(testees))} ; seuil "
                    f"{_pc(STATIONNARITE_FRAC_ECHEC)})")
        if echecs:
            txt += " : " + ", ".join(
                f"{s['f0']:.1f} Hz (ρ = {s['rho']:+.2f}, "
                f"{_pv('p', s['pvalue'])})" for s in echecs)
        txt += (f". CV du RMS par bloc = {stat_diag['cv_rms_blocs']:.3f} "
                f"(seuil {STATIONNARITE_CV_RMS_MAX}). Verdict : "
                f"<b>{stat_diag['verdict']}</b> — il faut que les DEUX "
                "indicateurs dépassent leur seuil pour conclure à la "
                "non-stationnarité.")
        com['Sondes de stationnarité'] = txt

    com['Où lire le détail'] = (
        "Graphe 5 du rapport de diagnostic (Rapport_Details) : ρ et p-value "
        "en fonction de f₀, avec les seuils en pointillés. Infobulle "
        "« indépendance IID » des courbes SRE. Fichier IID_QualityGate_*.csv : "
        "valeurs par fréquence et par classe (Classe = −1 pour la série de "
        "tous les blocs).")
    return com


# =============================================================================
# SECTION 9ter — CONTRÔLE DE STATIONNARITÉ (critère utilisateur)
# =============================================================================
#
# Diagnostic calculé AVANT la boucle f₀, en quelques secondes. Il INFORME
# l'utilisateur : le choix de la voie de calcul (nombre de classes, loi,
# méthode de projection) reste le sien. La seule décision automatique est
# K = 1 lorsque AUTO_SELECT_K=True et AUTO_K_SELON_STATIONNARITE=True.
#
# Les seuils sont ceux de la SECTION 1 (STATIONNARITE_*), où leur plage et
# leur sens physique sont documentés.

def detecter_non_stationnarite(signal, fs, Tb, Q, f0_min, f0_max,
                               n_sondes=None):
    """Verdict STATIONNAIRE / NON_STATIONNAIRE / DOUTEUX + recommandation.

    Trois indicateurs, décrits en SECTION 1 :

    1. Quality Gate IID (`quality_gate_iid`) appliquée aux maxima de réponse
       par bloc, pour N fréquences sondes log-espacées. Une enveloppe lente
       (changement de régime, dérive, modulation d'énergie) corrèle les maxima
       de blocs successifs : le ρ de Spearman lag-1 monte et le test des suites
       échoue. Les sondes démarrent à f₀ ≥ 3·Q/(π·T_b) pour que la mémoire de
       l'oscillateur (τ = Q/(π·f₀)) reste courte devant T_b : ainsi un échec ne
       peut PAS être imputé à un T_b trop court, seulement au signal lui-même.
    2. Coefficient de variation du RMS de l'excitation par bloc T_b : mesure
       directe de la modulation d'énergie.
    3. Kurtosis et asymétrie de l'excitation : caractère gaussien.

    Paramètres
    ----------
    signal   : array — excitation (m/s²).
    fs       : float — fréquence d'échantillonnage (Hz).
    Tb       : float — durée de bloc T_b (s).
    Q        : float — coefficient de surtension de l'oscillateur étalon.
    f0_min, f0_max : float — bornes du spectre analysé (Hz).
    n_sondes : int optionnel — défaut STATIONNARITE_N_SONDES.

    Retour
    ------
    dict : 'verdict', 'gaussien', 'frac_sondes_echec_iid', 'n_sondes_testees',
           'cv_rms_blocs', 'rho_rms_blocs', 'kurtosis', 'asymetrie',
           'sondes' (liste par f₀), 'recommandation', 'seuils'.
    """
    from scipy.stats import kurtosis as _kurt, skew as _skew
    n_sondes = int(STATIONNARITE_N_SONDES if n_sondes is None else n_sondes)
    sig = np.asarray(signal, dtype=float)

    # --- Indicateur 2 : RMS de l'excitation par bloc --------------------------
    taille = _taille_bloc(sig.size, fs, Tb)
    nb = sig.size // taille
    rms_b = (np.sqrt(np.mean(sig[:nb * taille].reshape(nb, taille) ** 2, axis=1))
             if nb >= 2 else np.array([np.nan]))
    mu = float(np.nanmean(rms_b))
    cv_rms = float(np.nanstd(rms_b) / mu) if mu > 0 else np.nan
    iid_rms = quality_gate_iid(rms_b) if nb >= IID_MIN_N else None

    # --- Indicateur 1 : sondes f₀, IID sur les maxima de réponse --------------
    f_lo = max(float(f0_min), 3.0 * float(Q) / (math.pi * float(Tb)))
    f_hi = max(float(f0_max), f_lo)
    sondes = []
    for f0 in np.geomspace(f_lo, f_hi, max(1, n_sondes)):
        rep = reponse_sdof(sig, f0=float(f0), Q=Q, fs=fs)
        _, mx, _, _, _ = extraire_caracteristiques(
            rep, fs, Tb_initial=Tb, feature_flags={}, min_ech=MIN_ECH_PAR_BLOC)
        g = quality_gate_iid(mx)
        sondes.append({'f0': float(f0), 'n': int(g['n']), 'rho': g['rho'],
                       'pvalue': g['pvalue'], 'tested': g['tested'],
                       'fail': g['fail']})
    testees = [s for s in sondes if s['tested']]
    frac_echec = (float(np.mean([s['fail'] for s in testees]))
                  if testees else np.nan)

    # --- Indicateur 3 : gaussianité ------------------------------------------
    kurt = float(_kurt(sig, fisher=False))
    asym = float(_skew(sig))
    gaussien = (abs(kurt - 3.0) <= STATIONNARITE_KURT_TOL
                and abs(asym) <= STATIONNARITE_SKEW_TOL)

    # --- Verdict -------------------------------------------------------------
    ns_iid = bool(np.isfinite(frac_echec) and frac_echec > STATIONNARITE_FRAC_ECHEC)
    ns_rms = bool(np.isfinite(cv_rms) and cv_rms > STATIONNARITE_CV_RMS_MAX)
    if ns_iid and ns_rms:
        verdict = 'NON_STATIONNAIRE'
    elif not ns_iid and not ns_rms:
        verdict = 'STATIONNAIRE'
    else:
        verdict = 'DOUTEUX'

    if verdict == 'STATIONNAIRE' and gaussien:
        reco = ("Signal stationnaire et gaussien : une seule classe suffit "
                "(N_CLUSTERS=1, ou AUTO_SELECT_K=True qui imposera K=1). Le SRE "
                "long terme attendu est régulier et doit rester proche du SRX "
                "analytique — colonnes SRX du rapport, hypothèse bande étroite "
                "gaussienne. Comparer les deux : un écart important signale un "
                "problème d'ajustement, pas une propriété du signal.")
    elif verdict == 'STATIONNAIRE':
        reco = ("Signal stationnaire mais NON gaussien (kurtosis %.1f, asymétrie "
                "%.2f) : une seule classe suffit. Le SRX analytique, gaussien par "
                "construction, sous-estime alors la queue ; c'est le SRE MBD qui "
                "fait foi. Comparer les deux pour quantifier l'écart."
                % (kurt, asym))
    elif verdict == 'NON_STATIONNAIRE':
        reco = ("Signal NON stationnaire (changement de régime, dérive ou "
                "modulation d'énergie). Une loi unique ajustée sur TOUS les blocs "
                "mélange des régimes d'amplitude différents : sa queue est tirée "
                "vers le bas par les blocs faibles, et le long terme est "
                "sous-estimé. Voie conseillée : classes au sens de [1] §C.4 — "
                "AUTO_SELECT_K=True (features rms / kurtosis / crest_factor), "
                "synthèse par produit des répartitions (SYNTHESE_CLASSES="
                "'produit'). Noter que l'échec de la Quality Gate IID vient ici "
                "de l'enveloppe lente : allonger T_b ne le corrigera pas.")
    else:
        reco = ("Indicateurs contradictoires (sondes IID : %s ; CV du RMS par "
                "bloc : %s). Calculer les deux voies — mono-classe et "
                "AUTO_SELECT_K=True — et comparer les SRE projetés : si l'écart "
                "est faible, la question est sans objet ; sinon retenir la voie "
                "classes, plus conservative."
                % ('en échec' if ns_iid else 'OK',
                   'élevé' if ns_rms else 'faible'))

    return {'verdict': verdict, 'gaussien': bool(gaussien),
            'frac_sondes_echec_iid': frac_echec, 'n_sondes_testees': len(testees),
            'cv_rms_blocs': cv_rms,
            'rho_rms_blocs': (iid_rms['rho'] if iid_rms else np.nan),
            'kurtosis': kurt, 'asymetrie': asym,
            'sondes': sondes, 'recommandation': reco,
            'seuils': {'frac_echec': STATIONNARITE_FRAC_ECHEC,
                       'cv_rms_max': STATIONNARITE_CV_RMS_MAX,
                       'kurt_tol': STATIONNARITE_KURT_TOL,
                       'skew_tol': STATIONNARITE_SKEW_TOL}}


def _journaliser_stationnarite(diag):
    """Bloc de synthèse lisible dans le journal d'exécution."""
    logger.info("-" * 60)
    logger.info("  CONTRÔLE DE STATIONNARITÉ (indication, choix utilisateur)")
    logger.info("  Verdict : %s | %s", diag['verdict'],
                'gaussien' if diag['gaussien'] else 'non gaussien')
    logger.info("  Sondes f₀ en échec IID : %.0f %% (%d sondes) | CV RMS/bloc : %.3f"
                " | kurtosis : %.2f | asymétrie : %.2f",
                100.0 * diag['frac_sondes_echec_iid']
                if np.isfinite(diag['frac_sondes_echec_iid']) else float('nan'),
                diag['n_sondes_testees'], diag['cv_rms_blocs'],
                diag['kurtosis'], diag['asymetrie'])
    logger.info("  Recommandation : %s", diag['recommandation'])
    logger.info("-" * 60)


def exporter_csv_stationnarite(filepath, diag, meta):
    """CSV compact : une ligne de synthèse, puis une ligne par sonde f₀."""
    import pandas as pd
    rows = [{'type': 'synthese', 'f0': np.nan,
             'verdict': diag['verdict'], 'gaussien': diag['gaussien'],
             'frac_sondes_echec_iid': diag['frac_sondes_echec_iid'],
             'cv_rms_blocs': diag['cv_rms_blocs'],
             'rho_rms_blocs': diag['rho_rms_blocs'],
             'kurtosis': diag['kurtosis'], 'asymetrie': diag['asymetrie'],
             'rho': np.nan, 'pvalue': np.nan, 'fail': np.nan}]
    for s in diag['sondes']:
        rows.append({'type': 'sonde', 'f0': s['f0'], 'verdict': '',
                     'gaussien': '', 'frac_sondes_echec_iid': np.nan,
                     'cv_rms_blocs': np.nan, 'rho_rms_blocs': np.nan,
                     'kurtosis': np.nan, 'asymetrie': np.nan,
                     'rho': s['rho'], 'pvalue': s['pvalue'], 'fail': s['fail']})
    pd.DataFrame(rows).to_csv(filepath, sep=';', decimal=',', index=False,
                              encoding='utf-8-sig')
    logger.info("CSV stationnarité : %s (%d lignes)", filepath, len(rows))


# =============================================================================
# SECTION 9quater — SYNTHÈSE DES CLASSES PAR PRODUIT DES RÉPARTITIONS (§C.10)
# =============================================================================
#
# [1] §C.10 (p. 86) et [2] fig. 15 : les classes étant indépendantes,
#     P(Z_sup ≤ z) = Π_j F_j(z)^{M_j},  avec M_j = Occ(j)·T_v/T_b.
# Le maximum des quantiles par classe est toujours ≤ au quantile de ce produit
# (chaque classe apporte sa propre chance de dépassement) : il est donc NON
# conservatif. Le quantile du produit est obtenu par résolution scalaire de
#     Σ_j M_j · ln F_j(z) = ln α
# encadrée entre le max des quantiles à α (borne basse) et le max des quantiles
# à α^(1/n_classes) (borne haute élargie si besoin).

def _loi_logcdf(params, x):
    """ln F(x) de la loi ajustée, calculé de façon précise près de F = 1.

    Le logarithme est indispensable : à T_v long, M_j atteint 10⁷ et F_j(z)
    vaut 1 − 10⁻⁸ ; le produit direct des CDF perdrait toute précision.
    Couvre Kappa4 (y compris le cas h = 0, GEV) et Rayleigh généralisée.
    Retourne None pour une loi non prise en charge (l'appelant retombe alors
    sur le maximum des quantiles par classe)."""
    if not (isinstance(params, dict) and params.get('success')):
        return None
    loi = params.get('loi', 'kappa4')
    try:
        if loi == 'rayleigh_gen':
            a = float(params['rg_alpha']); lam = float(params['rg_lambda'])
            u = (lam * float(x)) ** 2
            # log1p(-exp(-u)) via expm1 : stable pour u petit comme pour u grand.
            return float(a * np.log(-np.expm1(-u))) if u > 0 else -np.inf
        if loi not in ('kappa4', 'gev'):
            return None
        xi, al = float(params['xi']), float(params['alpha'])
        k = float(params['k'])
        h = params.get('h', 0.0)
        h = 0.0 if (h is None or not np.isfinite(h)) else float(h)
        y = (float(x) - xi) / al
        if abs(k) < _KAPPA4_K_EPS:
            A = math.exp(-y) if y > -700 else np.inf
        else:
            base = 1.0 - k * y
            if base <= 0.0:
                # k > 0 : borne supérieure atteinte, F = 1 ⇒ ln F = 0.
                # k < 0 : borne inférieure, F = 0 ⇒ ln F = −∞.
                return 0.0 if k > 0 else -np.inf
            A = base ** (1.0 / k)
        if abs(h) < 1e-12:
            return -A                      # Gumbel / GEV : ln F = −A
        t = 1.0 - h * A
        if t <= 0.0:
            return -np.inf
        return float(np.log1p(-h * A) / h)
    except Exception:
        return None


def quantile_produit_classes(classes, total_blocs, Tb, T_proj, alfa):
    """Quantile α de Π_j F_j^{M_j} — synthèse stochastique de [1] §C.10.

    Paramètres
    ----------
    classes     : liste de (params_loi, n_blocs_de_la_classe).
    total_blocs : nombre total de blocs du signal (pour Occ(j) = n_j/total).
    Tb, T_proj  : durée de bloc et durée de projection (s).
    alfa        : probabilité de NON-dépassement visée.

    Retour
    ------
    (sre, ok) — ok=False si une loi n'est pas prise en charge ou si la
    résolution échoue ; l'appelant conserve alors le maximum des quantiles.
    """
    cl = [(p, n) for (p, n) in classes
          if isinstance(p, dict) and p.get('success') and n > 0]
    if not cl or total_blocs <= 0 or Tb <= 0 or not (0.0 < alfa < 1.0):
        return None, False
    if len(cl) == 1:
        # Une seule classe : le produit se réduit à la projection standard.
        v, _M = calculer_projection_lmoments(cl[0][0], None, cl[0][1],
                                             total_blocs, Tb, T_proj, alfa)
        return v, v is not None
    M = [(n / total_blocs) * T_proj / Tb for _, n in cl]
    return _quantile_produit([p for p, _ in cl], M, math.log(alfa))


def _quantile_produit(lois, M, ln_alfa):
    """Racine z de Σ_j M_j · ln F_j(z) = ln α — cœur de la synthèse par produit.

    Sert aux deux horizons : la projection (M_j = Occ(j)·T_proj/T_b, α =
    ALFA_PROJECTION) et la durée du signal (M_j = nombre de blocs mesurés de la
    classe, cf. `traiter_f0`). ln α est passé tel quel, et non α : à la durée
    du signal α = p^N peut être inférieur au plus petit flottant.

    Paramètres
    ----------
    lois    : liste de dicts de loi ajustée (success=True).
    M       : liste des exposants M_j, même ordre.
    ln_alfa : logarithme de la probabilité de non-dépassement visée (< 0).

    Retour : (z, ok) — ok=False si une loi n'est pas prise en charge ou si la
    résolution échoue.
    """
    from scipy.optimize import brentq
    if (not lois or not (ln_alfa < 0.0)
            or any(_loi_logcdf(p, 0.0) is None for p in lois)):
        return None, False

    def q_max(diviseur):
        """Max des quantiles par classe au niveau α^(1/diviseur) (borne
        d'encadrement) : classe j seule, F_j(z)^{M_j} = α^(1/diviseur)."""
        vals = []
        for p, Mi in zip(lois, M):
            v = loi_ppf(p, math.exp(ln_alfa / (diviseur * Mi)))
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return max(vals) if vals else np.nan

    lo, hi = q_max(1.0), q_max(float(len(lois)))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None, False

    def g(x):
        s = 0.0
        for p, Mi in zip(lois, M):
            lf = _loi_logcdf(p, x)
            if lf is None or not np.isfinite(lf):
                return -np.inf
            s += Mi * lf
        return s - ln_alfa

    g_lo = g(lo)
    if not np.isfinite(g_lo) or g_lo >= 0.0:
        return lo, True                    # le max des quantiles suffit déjà
    for _ in range(12):                    # élargissement de la borne haute
        g_hi = g(hi)
        if np.isfinite(g_hi) and g_hi >= 0.0:
            break
        hi = lo + 1.5 * (hi - lo) + 1e-9 * abs(lo)
    else:
        return None, False
    try:
        return float(brentq(g, lo, hi, xtol=1e-9 * max(1.0, abs(hi)),
                            maxiter=200)), True
    except Exception:
        return None, False

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
      G. Synthèse des classes :
            SRE(f₀)  = quantile du produit Π_i F_i^{N_i} ([2] éq. 37) si
                       SYNTHESE_CLASSES='produit', sinon max_i SRE(f₀, classe_i).
                       Les classes sans loi exploitable sont consignées dans
                       'classes_exclues'.
            SDF(f₀) = Σᵢ Σⱼ D_p(i, j)        (= sdf_mbd_empirique)

    Retour
    ------
    dict avec clés :
        'f0', 'success', 'src', 'sre',
        'sre_max_classe', 'sre_synthese' ('produit' | 'max'),
        'classes_exclues', 'occ_exclue'  (classes sans loi exploitable),
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

        # Alias conservé pour les exports CSV et les rapports HTML.
        result['sdf_kappa4_empirical'] = sdf_mbd_empirique
        result['sdf_mbd_empirique']    = sdf_mbd_empirique

        ppf_valides = [p for p in result['valeurs_ppf_list']
                       if p is not None and np.isfinite(p)]
        result['sre'] = max(ppf_valides) if ppf_valides else None

        # Synthèse des classes à la durée du signal ([2] éq. 37) : même règle
        # que pour la projection, avec M_j = nombre de blocs MESURÉS de la
        # classe. Le niveau visé est celui du quantile mono-classe appliqué à
        # l'ensemble des N blocs : Π_j F_j(z)^{N_j} = p^N, ce qui redonne
        # exactement F(z) = p quand il n'y a qu'une classe. Le maximum des
        # quantiles par classe, conservé dans 'sre_max_classe', lui est
        # toujours inférieur ou égal.
        result['sre_max_classe'] = result['sre']
        result['sre_synthese']   = 'max'
        cl_ok = [(p, len(m)) for p, m, s in zip(result['params_list'],
                                                result['maxima_classes'],
                                                result['sre_class_status'])
                 if s == 'ok' and len(m) > 0]
        if SYNTHESE_CLASSES == 'produit' and len(cl_ok) > 1:
            n_inc = sum(n for _, n in cl_ok)
            p_tot = prob_cible
            if option_cunnane:
                denom = n_inc + 1.0 - 2.0 * cunnane_a
                if abs(denom) > 1e-9:
                    p_tot = float(np.clip((n_inc - cunnane_a) / denom,
                                          PROBA_CLIP_EPS, 1.0 - PROBA_CLIP_EPS))
            v_prod, ok_prod = _quantile_produit(
                [p for p, _ in cl_ok], [float(n) for _, n in cl_ok],
                n_inc * math.log(p_tot))
            if ok_prod and v_prod is not None and np.isfinite(v_prod):
                result['sre'] = float(v_prod)
                result['sre_synthese'] = 'produit'

        # Classes absentes de la synthèse : une classe qui contient des blocs
        # mais n'a pas de loi exploitable (trop peu de blocs, ajustement en
        # échec) ne contribue pas au SRE, qui est alors sous-estimé. On le
        # consigne pour que les exports et les rapports le signalent.
        exclues = [i for i, (m, s) in enumerate(zip(result['maxima_classes'],
                                                    result['sre_class_status']))
                   if s != 'ok' and len(m) > 0]
        if not cl_ok:
            exclues = []            # aucune classe valide : trou, pas exclusion
        result['classes_exclues'] = exclues
        result['occ_exclue'] = (
            sum(len(result['maxima_classes'][i]) for i in exclues)
            / float(max(min_len, 1)))

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
        'version':            'v3.6',
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


def _cols_synthese_classes(item, synthese, sre_max_classe, nom_col_max):
    """Colonnes CSV décrivant la synthèse des classes à une fréquence.

    `item` porte 'classes_exclues' (indices des classes sans loi exploitable)
    et 'occ_exclue' (part des blocs du signal qu'elles représentent). Une
    cellule Classes_exclues non vide signale un SRE SOUS-ESTIMÉ : la
    contribution de ces classes au risque n'est pas comptée."""
    return {
        'Synthese_classes':  synthese or 'max',
        nom_col_max:         (sre_max_classe if sre_max_classe is not None
                              else np.nan),
        'Classes_exclues':   ','.join(str(c) for c in
                                      item.get('classes_exclues', [])),
        'Occurrence_exclue': item.get('occ_exclue', 0.0),
    }


def _bilan_classes_exclues(items, quoi):
    """Avertit quand des classes manquent dans la synthèse des classes.

    Une classe sans loi exploitable (moins de MIN_POINTS_KAPPA4 blocs,
    ajustement en échec) est écartée de la synthèse : le spectre est alors
    calculé sur les seules classes restantes, donc sous-estimé, et rien dans
    la courbe ne le montre. Ce bilan le rend visible dans le journal et dans
    le cadre des rapports HTML ; le détail par fréquence est dans la colonne
    Classes_exclues des CSV.

    items : liste de dicts portant 'f0', 'classes_exclues', 'occ_exclue'.
    quoi  : nom du spectre concerné, pour le message.
    Retour : texte pour le cadre HTML, ou None si aucune classe n'est exclue.
    """
    touches = [it for it in items if it.get('classes_exclues')]
    if not touches:
        return None
    f0s     = sorted(float(it['f0']) for it in touches)
    classes = sorted({c for it in touches for c in it['classes_exclues']})
    occ_max = max(float(it.get('occ_exclue', 0.0)) for it in touches)
    txt = (f"{quoi} : à {len(touches)} f₀ sur {len(items)} "
           f"({f0s[0]:g}–{f0s[-1]:g} Hz), classe(s) "
           f"{', '.join(str(c) for c in classes)} sans loi exploitable, soit "
           f"jusqu'à {100.0 * occ_max:.1f} % des blocs")
    logger.warning("Synthèse des classes INCOMPLÈTE — %s. La contribution de "
                   "ces classes au risque n'est pas comptée : le spectre est "
                   "sous-estimé à ces fréquences (colonne Classes_exclues des "
                   "CSV).", txt)
    return txt


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

    # Colonnes de synthèse des classes : seulement s'il y a plusieurs classes
    # (un calcul mono-classe garde exactement les mêmes colonnes qu'avant).
    multi = int(meta.get('K_means', 1)) > 1

    rows = []
    for r in sorted(results, key=lambda x: x['f0']):
        if not r['success']:
            continue
        rho, pval, stat = _iid_cols(r, 'sre')
        row = {
            'Frequence_Hz':                       r['f0'],
            f"SRC_{sx['sdof']}":                  r.get('src', np.nan),
            f"SRE_Kappa4_{sx['sre']}":            r.get('sre') if r.get('sre') is not None else np.nan,
            f"SRE_DSP_NormDef_{sx['sre_dsp']}":   sre_d.get(r['f0'], np.nan),
            f"SRX_alpha_low_{sx['srx_low']}":     xl_d.get(r['f0'],  np.nan),
            f"SRX_alpha_high_{sx['srx_high']}":   xh_d.get(r['f0'],  np.nan),
            f"IID_rho_lag1_{sx['iid']}":          rho,
            f"IID_pvalue_runs_{sx['iid']}":       pval,
            f"IID_status_{sx['iid']}":            stat,
        }
        if multi:
            row.update(_cols_synthese_classes(
                r, r.get('sre_synthese'), r.get('sre_max_classe'),
                f"SRE_Kappa4_max_classe_{sx['sre']}"))
        rows.append(row)
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
    # Colonnes de synthèse des classes (mode retenu, max des quantiles par
    # classe, classes exclues) : seulement s'il y a plusieurs classes.
    multi = int(meta.get('K_means', 1)) > 1
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
        if multi:
            row.update(_cols_synthese_classes(
                pr, pr.get('synthese_classes'), pr.get('sre_proj_max_classe'),
                f"SRE_Projection_max_classe_{sx['proj_sre']}"))
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


def _nom_loi_courte():
    """Libellé compact de la loi active, pour les légendes des rapports."""
    return 'Rayleigh gén.' if LOI_AJUSTEMENT == 'rayleigh_gen' else 'Kappa4'


def _geometrie_rangees(fig, n_rows):
    """Géométrie réelle des sous-graphes : (domaines verticaux, clés d'axes).

    Les domaines sont lus sur les axes construits par make_subplots — seule
    source fiable dès que row_heights et vertical_spacing ne sont pas
    uniformes. Les clés d'axes le sont aussi : avec des axes secondaires
    (secondary_y), la numérotation des axes Y ne suit PAS celle des rangées
    (rangée 4 → 'yaxis5'), et un placement calculé « à la main » viserait le
    mauvais graphe.

    Retour : (liste de (bas, haut), liste de (clé_x, clé_y)) par rangée.
    """
    doms, axes = [], []
    for i in range(1, n_rows + 1):
        try:
            sp = fig.get_subplot(i, 1)
            xk = sp.xaxis.plotly_name
            yk = sp.yaxis.plotly_name
            dom = tuple(sp.yaxis.domain)
        except Exception:                       # repli : partage uniforme
            xk, yk = _axe_key('x', i), _axe_key('y', i)
            step = 1.0 / max(n_rows, 1)
            dom = (1.0 - i * step, 1.0 - (i - 1) * step)
        if len(dom) != 2 or dom[0] is None:
            step = 1.0 / max(n_rows, 1)
            dom = (1.0 - i * step, 1.0 - (i - 1) * step)
        doms.append((float(dom[0]), float(dom[1])))
        axes.append((xk, yk))
    return doms, axes


def _domaines_rangees(fig, n_rows):
    """Domaines verticaux (bas, haut) de chaque sous-graphe (coordonnées paper)."""
    return _geometrie_rangees(fig, n_rows)[0]


def _legendes_par_rangee(fig, n_rows, titres=None, x=1.01):
    """Une légende PAR sous-graphe, ancrée en haut de la rangée correspondante.

    Une figure empilée n'a par défaut qu'UNE légende, en haut à droite, qui
    rassemble les courbes des N graphes : impossible de savoir à quel graphe
    appartient une entrée, et la liste finit par dépasser la hauteur de la
    figure. Plotly accepte plusieurs légendes ('legend', 'legend2', …) : on en
    crée une par rangée, chacune placée au niveau de son graphe.

    Retour : (dict de layout à passer à update_layout, liste des noms de
    légende indexée par rangée — `noms[0]` vaut 'legend' pour la rangée 1).
    """
    doms = _domaines_rangees(fig, n_rows)
    layout, noms = {}, []
    for i in range(1, n_rows + 1):
        nom = 'legend' if i == 1 else f'legend{i}'
        noms.append(nom)
        cfg = dict(x=x, y=doms[i - 1][1], xanchor='left', yanchor='top',
                   font=dict(size=10), bgcolor='rgba(255,255,255,0.88)',
                   bordercolor='#c8c8c8', borderwidth=1,
                   itemsizing='constant', tracegroupgap=4)
        if titres and i - 1 < len(titres) and titres[i - 1]:
            cfg['title'] = dict(text=f"<b>{titres[i - 1]}</b>",
                                font=dict(size=10))
        layout[nom] = cfg
    return layout, noms


def _boutons_echelle_par_rangee(fig, n_rows):
    """Boutons d'échelle (X lin/log, Y lin/log) placés au-dessus et à DROITE
    de chaque sous-graphe, dans la bande libre qui sépare deux graphes.

    Les placer dans la marge de droite les ferait chevaucher les légendes ;
    les placer dans le graphe masquerait les courbes. Ancrés à droite du bord
    du tracé (x = 1, xanchor='right'), sur la même ligne de base que le titre
    du sous-graphe, qui est lui aligné à gauche (cf. _titres_rangees_a_gauche),
    ils occupent la bande libre au-dessus du graphe.
    """
    doms, axes = _geometrie_rangees(fig, n_rows)
    menus = []
    for i in range(1, n_rows + 1):
        xk, yk = axes[i - 1]
        menus.append(dict(
            type='buttons', direction='right', showactive=False,
            x=1.0, y=doms[i - 1][1], xanchor='right',
            yanchor='bottom', pad=dict(r=0, t=0, b=0, l=0),
            font=dict(size=9), bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#c8c8c8',
            buttons=[
                dict(label='X lin', method='relayout',
                     args=[{f'{xk}.type': 'linear'}]),
                dict(label='X log', method='relayout',
                     args=[{f'{xk}.type': 'log'}]),
                dict(label='Y lin', method='relayout',
                     args=[{f'{yk}.type': 'linear'}]),
                dict(label='Y log', method='relayout',
                     args=[{f'{yk}.type': 'log'}]),
            ]))
    return menus


def _titres_rangees_a_gauche(fig, n_rows):
    """Aligne à GAUCHE les titres de sous-graphes créés par make_subplots.

    Centrés (réglage par défaut), ils passent sous les boutons d'échelle dès
    qu'ils sont longs. Alignés à gauche, les deux zones sont disjointes."""
    doms = _domaines_rangees(fig, n_rows)
    tops = [round(d[1], 6) for d in doms]
    for ann in fig.layout.annotations:
        # Les titres de sous-graphes sont les seules annotations en coordonnées
        # paper posées exactement sur le haut d'une rangée, ancrées par le bas.
        if (ann.xref == 'paper' and ann.yref == 'paper'
                and ann.y is not None and ann.yanchor == 'bottom'
                and any(abs(round(float(ann.y), 6) - t) < 5e-3 for t in tops)):
            ann.update(x=0.0, xanchor='left', align='left',
                       font=dict(size=12.5))


def _ecrire_html_avec_cadre(fig, filepath, config_info=None, guide_courbes=None,
                            commentaire_iid=None):
    """Écrit le HTML Plotly précédé de blocs repliables (<details>) : les
    paramètres du calcul, l'explication de la Quality Gate IID, et un guide de
    lecture des courbes.

    Ces blocs sont du HTML statique placé AVANT le graphe : contrairement à une
    annotation Plotly, ils ne recouvrent aucune courbe et restent lisibles quel
    que soit le zoom. plotly.js est embarqué (consultation hors connexion).

    guide_courbes   : dict {nom de courbe: signification}, affiché en tableau.
    commentaire_iid : dict renvoyé par `commenter_quality_gate` ; le bloc porte
                      un liseré à la couleur du verdict (clé '_status').
    """
    import plotly.io as pio
    html = pio.to_html(fig, full_html=True, include_plotlyjs=True)

    def _bloc(titre, dico, ouvert, lisere=None):
        rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0;font-weight:bold;"
            f"white-space:nowrap;vertical-align:top'>{k}</td>"
            f"<td style='padding:2px 0'>{v}</td></tr>"
            for k, v in dico.items() if not str(k).startswith('_'))
        bord = f"border-left:6px solid {lisere};" if lisere else ""
        return (
            f"<details {'open' if ouvert else ''} "
            "style=\"font-family:Segoe UI,Arial,sans-serif;"
            "margin:10px;border:1px solid #c8c8c8;border-radius:6px;"
            f"{bord}"
            "padding:6px 12px;background:#f6f6f6;max-width:1100px\">"
            "<summary style='cursor:pointer;font-weight:bold;font-size:14px'>"
            f"{titre}</summary>"
            f"<table style='font-size:12px;margin-top:6px;"
            f"border-collapse:collapse'>{rows}</table></details>")

    blocs = ""
    if config_info:
        blocs += _bloc("Paramètres du calcul (cliquer pour replier / déplier)",
                       config_info, True)
    if commentaire_iid:
        _coul = {'GO': '#2e7d32', 'WARNING': '#f9a825',
                 'NO-GO': '#c62828'}.get(commentaire_iid.get('_status'))
        blocs += _bloc("Contrat IID (indépendance des blocs) — explication "
                       "des résultats (cliquer pour replier / déplier)",
                       commentaire_iid, True, lisere=_coul)
    if guide_courbes:
        blocs += _bloc("Guide de lecture des courbes (cliquer pour déplier)",
                       guide_courbes, False)
    if blocs:
        low = html.lower()
        idx = low.find('<body>')
        if idx != -1:
            pos = idx + len('<body>')
            html = html[:pos] + blocs + html[pos:]
        else:
            html = blocs + html
    with open(filepath, 'w', encoding='utf-8') as fh:
        fh.write(html)


def generer_html(filepath, results, sre_dsp, srx_low, srx_high, sdf_spectral,
                 f0_grid, proj_results=None, config_info=None,
                 sre_dsp_proj=None, srx_low_proj=None, srx_high_proj=None,
                 alpha_srx_low=None, alpha_srx_high=None, iid_diag=None,
                 commentaire_iid=None):
    """Rapport HTML interactif Plotly — synthèse spectrale (2 à 4 graphes).

    Mise en page : chaque sous-graphe a SA PROPRE légende, placée en face de
    lui, et ses boutons d'échelle en haut à droite ; les titres de sous-graphes
    sont alignés à gauche. Aucun de ces éléments n'en recouvre un autre, quel
    que soit le nombre de courbes.

    En tête de page, sous les paramètres : le bloc « Contrat IID » qui explique
    en clair les résultats de la Quality Gate (`commentaire_iid`, produit par
    `commenter_quality_gate`).

    Contenu :
      - graphe 1 : SRC et SRE sur la durée du signal ;
      - graphe 2 : comparatif SRE / SRX, mesure et projection (axe Y log) —
        SRE DSP (NORMDEF §5.4.2), SRX α_low (dimensionnement) et SRX α_high
        (comparaison au SRC d'un choc), NORMDEF §5.4.3 ;
      - graphe 3 (si SDF actif) : spectres de dommage par fatigue ;
      - graphe 4 (si projection active) : SRE mesuré vs SRE projeté.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly non installé — rapport HTML non généré.")
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def _get(lst, f0, key, default=np.nan):
        for r in lst:
            if r['f0'] == f0 and r['success']:
                v = r.get(key)
                return v if v is not None else default
        return default

    loi_lbl = _nom_loi_courte()

    f0_list = sorted(r['f0'] for r in results if r['success'])
    sre_d_d    = dict(zip(f0_grid, sre_dsp))
    xl_d       = dict(zip(f0_grid, srx_low))
    xh_d       = dict(zip(f0_grid, srx_high))
    sp_d       = dict(zip(f0_grid, sdf_spectral)) if sdf_spectral is not None else {}
    sre_d_proj = dict(zip(f0_grid, sre_dsp_proj))  if sre_dsp_proj  is not None else {}
    xl_proj_d  = dict(zip(f0_grid, srx_low_proj))  if srx_low_proj  is not None else {}
    xh_proj_d  = dict(zip(f0_grid, srx_high_proj)) if srx_high_proj is not None else {}

    sre_k4  = [_get(results, f, 'sre') for f in f0_list]
    src_v   = [_get(results, f, 'src') for f in f0_list]

    # Diagnostic du SRE : raison synthétique + détail par classe, exposés dans
    # l'infobulle de la courbe SRE et des marqueurs d'échec.
    def _sre_diag(f0):
        for r in results:
            if r['f0'] == f0 and r['success']:
                return r.get('sre_diag') or {}
        return {}
    sre_reasons   = [(_sre_diag(f).get('reason') or 'n/a') for f in f0_list]
    sre_per_class = [(_sre_diag(f).get('per_class') or []) for f in f0_list]

    def _per_class_str(pc):
        return ' | '.join(f"cl{i}:{s}" for i, s in enumerate(pc)) if pc else ''
    sre_per_class_str = [_per_class_str(pc) for pc in sre_per_class]
    _iid_conf_map = (iid_diag.get('per_f0_confidence', {})
                     if iid_diag is not None else {})
    iid_conf = [_iid_conf_map.get(float(f), 'n/a') for f in f0_list]
    sre_cd   = np.column_stack([np.asarray(sre_reasons, dtype=object),
                                np.asarray(iid_conf, dtype=object)])

    sdf_t   = [_get(results, f, 'sdf_temporel') for f in f0_list]
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

    _T_lbl = f"{DUREE_PROJECTION:.1e} s"
    titles = [
        '1 — SRC et SRE sur la durée du signal',
        '2 — Comparatif SRE / SRX : MBD, DSP analytique, mesure et projection '
        '(axe Y logarithmique)',
    ]
    if has_sdf:
        titles.append('3 — SDF : spectres de dommage par fatigue '
                      f'(b = {SDF_B:g}, axe Y logarithmique)')
    if has_proj:
        _meth = ('max-stabilité GEV (3 domaines)'
                 if METHODE_PROJECTION == 'gev_domaines' else 'puissance F^M')
        titles.append(f'4 — SRE MBD : durée du signal vs projeté à T = {_T_lbl} '
                      f'(α = {ALFA_PROJECTION}, {_meth})')

    # Le comparatif SRE / SRX (rangée 2) porte le plus de courbes : +30 %.
    _row_units = [1.0] * n_rows
    _row_units[1] = 1.3
    _row_heights = [u / sum(_row_units) for u in _row_units]
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles,
                        vertical_spacing=0.085, row_heights=_row_heights)

    # Une légende par rangée, titrée, placée en face de son graphe.
    titres_leg = ['Mesure', 'SRE / SRX']
    if has_sdf:
        titres_leg.append('SDF')
    if has_proj:
        titres_leg.append('Projection')
    layout_leg, leg = _legendes_par_rangee(fig, n_rows, titres=titres_leg)

    # ------------------------------------------------------------------ row 1
    fig.add_trace(go.Scatter(
        x=f0_list, y=src_v, legend=leg[0],
        name='SRC — max de la réponse (déterministe)',
        line=dict(color='forestgreen', width=1.2),
        hovertemplate='f₀=%{x:.2f} Hz<br>SRC=%{y:.4g} m/s²<extra></extra>'),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sre_k4, legend=leg[0],
        name=f'SRE MBD {loi_lbl} — durée du signal',
        line=dict(color='royalblue', width=2),
        customdata=sre_cd,
        hovertemplate=('f₀=%{x:.2f} Hz<br>SRE=%{y:.4g} m/s²'
                       '<br>statut du fit : %{customdata[0]}'
                       '<br>indépendance IID : %{customdata[1]}'
                       '<extra></extra>')),
        row=1, col=1)

    # Marqueurs aux f₀ sans SRE exploitable — posés sur la courbe SRC pour
    # rester dans l'échelle du graphe.
    fail_x, fail_y, fail_hover = [], [], []
    for f, s, src, reason, pcs in zip(f0_list, sre_k4, src_v,
                                      sre_reasons, sre_per_class_str):
        if s is None or (isinstance(s, float) and not np.isfinite(s)):
            fail_x.append(f)
            fail_y.append(src if (src is not None and np.isfinite(src)) else 0.0)
            fail_hover.append(f"{reason}<br>{pcs}" if pcs else reason)
    if fail_x:
        fig.add_trace(go.Scatter(
            x=fail_x, y=fail_y, mode='markers', legend=leg[0],
            name=f'SRE non calculable ({len(fail_x)} f₀) — survoler pour la raison',
            marker=dict(symbol='x', color='red', size=9, line=dict(width=2)),
            customdata=fail_hover,
            hovertemplate=('f₀=%{x:.2f} Hz<br>SRE MBD : non calculable'
                           '<br>%{customdata}<extra></extra>')),
            row=1, col=1)

    # ------------------------------------------------------------------ row 2
    # Convention de tracé : trait PLEIN = durée du signal, TIRETS = projeté.
    fig.add_trace(go.Scatter(
        x=f0_list, y=src_v, legend=leg[1], name='SRC (référence)',
        line=dict(color='darkgray', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sre_k4, legend=leg[1],
        name=f'SRE MBD {loi_lbl} — signal',
        line=dict(color='royalblue', width=2),
        customdata=sre_cd,
        hovertemplate=('f₀=%{x:.2f} Hz<br>SRE=%{y:.4g} m/s²'
                       '<br>statut du fit : %{customdata[0]}'
                       '<br>indépendance IID : %{customdata[1]}'
                       '<extra></extra>')),
        row=2, col=1)
    if sre_mbd_proj is not None:
        fig.add_trace(go.Scatter(
            x=f0_list, y=sre_mbd_proj, legend=leg[1],
            name=f'SRE MBD {loi_lbl} — projeté {_T_lbl}',
            line=dict(color='royalblue', width=2.5, dash='dash')),
            row=2, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sre_d, legend=leg[1],
        name='SRE DSP analytique — signal',
        line=dict(color='seagreen', width=2)), row=2, col=1)
    if sre_d_p is not None:
        fig.add_trace(go.Scatter(
            x=f0_list, y=sre_d_p, legend=leg[1],
            name=f'SRE DSP analytique — projeté {_T_lbl}',
            line=dict(color='seagreen', width=2, dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=xl_v, legend=leg[1],
        name=f'SRX risque α={aL_lbl} — signal<br>(dimensionnement)',
        line=dict(color='darkorange', width=2)), row=2, col=1)
    if xl_v_p is not None:
        fig.add_trace(go.Scatter(
            x=f0_list, y=xl_v_p, legend=leg[1],
            name=f'SRX risque α={aL_lbl} — projeté {_T_lbl}',
            line=dict(color='darkorange', width=2, dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=xh_v, legend=leg[1],
        name=f'SRX risque α={aH_lbl} — signal<br>(comparaison au choc)',
        line=dict(color='firebrick', width=2)), row=2, col=1)
    if xh_v_p is not None:
        fig.add_trace(go.Scatter(
            x=f0_list, y=xh_v_p, legend=leg[1],
            name=f'SRX risque α={aH_lbl} — projeté {_T_lbl}',
            line=dict(color='firebrick', width=2, dash='dash')), row=2, col=1)
    fig.update_yaxes(type='log', row=2, col=1)

    # ------------------------------------------------------------------ row 3
    row = 3
    if has_sdf:
        def _clean_pos_basic(arr):
            a = np.asarray(arr, dtype=float)
            return np.where((a > 0) & np.isfinite(a), a, np.nan)
        fig.add_trace(go.Scatter(
            x=f0_list, y=_clean_pos_basic(sdf_t), legend=leg[row - 1],
            name='SDF rainflow sur tout le signal',
            line=dict(color='crimson', width=2)), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=f0_list, y=_clean_pos_basic(sdf_k), legend=leg[row - 1],
            name='SDF MBD — Σ dommages par bloc<br>([1] §C.10)',
            line=dict(color='orange', width=2)), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=f0_list, y=_clean_pos_basic(sdf_sp), legend=leg[row - 1],
            name='SDF analytique DSP (bande étroite, Bendat)',
            line=dict(color='purple', width=2, dash='dash')), row=row, col=1)
        fig.update_yaxes(type='log', row=row, col=1)
        row += 1

    # ------------------------------------------------------------------ row 4
    if has_proj:
        pd_ = {pr['f0']: pr.get('sre_proj_max') for pr in proj_results}
        sre_proj = [pd_.get(f) for f in f0_list]
        fig.add_trace(go.Scatter(
            x=f0_list, y=sre_k4, legend=leg[row - 1],
            name='SRE MBD — durée du signal',
            line=dict(color='steelblue', width=1.2)), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=f0_list, y=sre_proj, legend=leg[row - 1],
            name=f'SRE MBD — projeté à T = {_T_lbl}',
            line=dict(color='crimson', width=2)), row=row, col=1)
        row += 1

    # --- Axes : titres explicites, unités incluses ---------------------------
    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Fréquence propre f₀ (Hz)",
                         title_standoff=6, row=i, col=1)
    fig.update_yaxes(title_text="Accélération (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Accélération (m/s², log)", row=2, col=1)
    _r = 3
    if has_sdf:
        fig.update_yaxes(title_text="Dommage (u.a., log)", row=_r, col=1)
        _r += 1
    if has_proj:
        fig.update_yaxes(title_text="Accélération (m/s²)", row=_r, col=1)

    _nom_fic = (config_info or {}).get('Fichier', '')
    fig.update_layout(
        title=dict(
            text=(f"<b>MBD V3.6 — synthèse spectrale</b> — {_nom_fic}"
                  f"<br><span style='font-size:11px'>Loi {loi_lbl} · "
                  f"T_b = {TB:g} s · Q = {Q:g} · "
                  f"trait plein = durée du signal, tirets = projeté · "
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>"),
            x=0.0, xanchor='left', y=0.985, yanchor='top'),
        height=int(330 * sum(_row_units)) + 120,
        margin=dict(l=85, r=330, t=115, b=60),
        hovermode='closest',
        updatemenus=_boutons_echelle_par_rangee(fig, n_rows),
        **layout_leg,
    )
    _titres_rangees_a_gauche(fig, n_rows)

    guide = {
        'SRC': "Spectre de Réponse au Choc : maximum de la réponse de "
               "l'oscillateur 1-DDL sur le signal. Grandeur déterministe, sans "
               "extrapolation — borne inférieure de référence.",
        f'SRE MBD {loi_lbl}': "Spectre de Réponse Extrême par la méthode des "
               "blocs de la norme : la loi est ajustée sur les maxima par bloc "
               "T_b, puis le quantile demandé est extrait. « Durée du signal » "
               "= quantile atteignable sur la population mesurée ; « projeté » "
               "= extrapolé à la durée de vie visée.",
        'SRE DSP analytique': "SRE calculé depuis la densité spectrale de "
               "puissance du signal (NORMDEF §5.4.2). Hypothèse bande étroite "
               "gaussienne : référence valable pour un signal stationnaire "
               "gaussien, optimiste sinon.",
        'SRX': "Spectre de Réponse à risque de dépassement (NORMDEF §5.4.3). "
               "α faible = enveloppe de DIMENSIONNEMENT ; α élevé (0,99) sert à "
               "comparer la vibration aléatoire à la sévérité d'un choc.",
        'SDF': "Spectre de Dommage par Fatigue (modèle de Basquin N·σ_a^b = C, "
               "cumul de Miner). Trois estimations : rainflow sur tout le "
               "signal, somme des dommages par bloc (voie MBD de la norme), et "
               "approximation analytique bande étroite depuis la DSP.",
        'Marqueurs ✕ rouges': "Fréquences où l'ajustement n'a pas abouti : "
               "survoler le marqueur affiche la raison et l'état par classe.",
        'Indépendance IID (infobulle)': "Mention affichée au survol des "
               "courbes SRE : « ok » si les blocs passent les deux tests "
               "d'indépendance à cette fréquence, « reduced » sinon. "
               "L'interprétation d'ensemble est donnée dans le bloc "
               "« Contrat IID » en tête de page.",
        'Boutons X/Y lin-log': "En haut à droite de chaque graphe : basculent "
               "l'échelle de l'axe correspondant, graphe par graphe.",
    }
    _ecrire_html_avec_cadre(fig, filepath, config_info, guide_courbes=guide,
                            commentaire_iid=commentaire_iid)
    logger.info("Rapport HTML : %s", filepath)


def generer_html_details(filepath, results, sre_dsp, srx_low, srx_high,
                          sdf_spectral, f0_grid, config_info=None,
                          proj_results=None, t_mesure=None, t_proj=None,
                          alpha_srx_low=None, alpha_srx_high=None,
                          iid_diag=None, commentaire_iid=None):
    """Rapport HTML de diagnostic de l'ajustement (4 à 5 graphes).

    En tête de page, sous les paramètres : le bloc « Contrat IID » qui explique
    en clair les résultats de la Quality Gate tracés au graphe 5
    (`commentaire_iid`, produit par `commenter_quality_gate`).

    Contenu :
      - graphe 1 : diagramme des L-moments (τ3, τ4) — un point par (f₀, classe),
        avec les frontières du domaine de la loi et la mise en évidence de la
        fréquence sélectionnée ;
      - graphe 2 : fonctions de répartition par classe — répartition empirique
        vs loi ajustée, pour la fréquence sélectionnée (menu en haut à gauche) ;
      - graphe 3 : comparatif SRE / SRX avec l'indicateur d'écart MSDI ;
      - graphe 4 : comparatif SDF projeté ;
      - graphe 5 (si Quality Gate active) : indépendance des blocs vs f₀.

    Mise en page : une légende par graphe, en face de lui ; boutons d'échelle en
    haut à droite de chaque graphe ; sélecteur de fréquence dans la bande de
    titre. Les contrôles et les légendes n'occupent jamais la même zone.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly non installé — rapport détails non généré.")
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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
    loi_lbl = _nom_loi_courte()

    ratio_rf = (t_proj / t_mesure
                if (t_mesure and t_proj and t_mesure > 0) else None)

    proj_d = {}
    if proj_results:
        for pr in proj_results:
            proj_d[pr['f0']] = pr

    def _safe_mean(vals):
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                          dtype=float)
        return float(np.mean(arr)) if arr.size else np.nan

    msdi_mean = [_safe_mean(r.get('msdi_list', [])) for r in results_ok]

    has_iid = (iid_diag is not None
               and len(np.asarray(iid_diag.get('f0', []))) > 0)

    subplot_titles = [
        "1 — Diagramme des L-moments (τ3, τ4) : un point par (f₀, classe)",
        f"2 — Répartition par classe à la fréquence sélectionnée : empirique "
        f"vs {loi_lbl} ajustée",
        "3 — Comparatif SRE / SRX sur la durée du signal, avec écart MSDI "
        "(axe de droite)",
        "4 — Comparatif SDF projeté (axe Y logarithmique), avec écart MSDI "
        "(axe de droite)",
    ]
    specs = [
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
    ]
    if has_iid:
        subplot_titles.append(
            "5 — Indépendance des blocs (Quality Gate IID) : ρ de Spearman "
            "lag-1 et p-value du test des suites — statut %s"
            % iid_diag.get('status', ''))
        specs.append([{"secondary_y": False}])
    n_rows_det = len(subplot_titles)
    # Le comparatif SRE (rangée 3) porte le plus de courbes : +30 %.
    _ru_det = [1.0] * n_rows_det
    _ru_det[2] = 1.3
    _rh_det = [u / sum(_ru_det) for u in _ru_det]
    fig = make_subplots(
        rows=n_rows_det, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.075,
        specs=specs,
        row_heights=_rh_det,
    )

    titres_leg = ['Domaine (τ3, τ4)', 'Répartitions', 'SRE / SRX', 'SDF projeté']
    if has_iid:
        titres_leg.append('Indépendance')
    layout_leg, leg = _legendes_par_rangee(fig, n_rows_det, titres=titres_leg)

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # --- Rangée 1 : diagramme (τ3, τ4) ---------------------------------------
    # Courbes théoriques Kappa4 à h fixé (k variable) : elles délimitent les
    # sous-familles classiques et situent visuellement chaque ajustement.
    for h_val, color, extra in [
            (-1.0, '#888888', '<br>logistique généralisée — LIMITE HAUTE'),
            (0.0, '#4444aa', '<br>valeurs extrêmes généralisée (GEV)'),
            (1.0, '#44aa44', '<br>Pareto généralisée (GPD)'),
            (5.0, '#aa4444', '')]:
        xs_th, ys_th = _kappa4_tau_curve_for_h(h_val)
        if xs_th.size:
            fig.add_trace(go.Scatter(
                x=xs_th, y=ys_th, legend=leg[0],
                mode='lines', line=dict(color=color, width=1.2, dash='dot'),
                name=f'Kappa4 h = {h_val:g}{extra}',
                legendgroup='tau_theo',
                hovertemplate='τ3=%{x:.3f}<br>τ4=%{y:.3f}<extra></extra>',
            ), row=1, col=1)

    # Limite BASSE valable pour toute loi : τ4 = (5τ3² − 1)/4 ([2] eq. 15).
    tau3_grid = np.linspace(-0.999, 0.999, 500)
    fig.add_trace(go.Scatter(
        x=tau3_grid, y=(5 * tau3_grid ** 2 - 1) / 4, legend=leg[0],
        mode='lines', line=dict(color='black', width=1.5),
        name='LIMITE BASSE toutes lois<br>τ4 = (5τ3² − 1)/4',
        legendgroup='tau_theo',
        hovertemplate='τ3=%{x:.3f}<br>τ4=%{y:.3f}<extra></extra>',
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
            txt.append(f"f₀ = {r['f0']:.1f} Hz | classe {ci} | N = {n_pts} blocs"
                       f"<br>ajustement : "
                       f"{'réussi' if p.get('success') else 'échec'}"
                       f" ({p.get('fail_reason', 'n/a')})"
                       f"<br>k = {p.get('k', float('nan')):.3f} | "
                       f"h = {p.get('h', float('nan')):.3f}")
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='markers', legend=leg[0],
                marker=dict(size=5, color=palette[ci % len(palette)], opacity=0.7),
                name=f'(τ3, τ4) mesurés — classe {ci}', text=txt, hoverinfo='text',
                legendgroup=f'class_{ci}',
            ), row=1, col=1)

    # Surbrillance de la fréquence sélectionnée (une trace par f₀, une seule
    # visible à la fois, pilotée par le menu de sélection).
    tau_hl_idx = []
    for r in results_ok:
        xs_h, ys_h, txt_h = [], [], []
        for ci, p in enumerate(r.get('params_list', [])):
            t3, t4 = p.get('t3'), p.get('t4')
            if t3 is None or t4 is None or not (np.isfinite(t3) and np.isfinite(t4)):
                continue
            xs_h.append(t3); ys_h.append(t4)
            txt_h.append(f"f₀ = {r['f0']:.1f} Hz | classe {ci}")
        fig.add_trace(go.Scatter(
            x=xs_h, y=ys_h, mode='markers', legend=leg[0],
            marker=dict(size=16, color='rgba(0,0,0,0)',
                        line=dict(color='black', width=2)),
            name=f'f₀ sélectionnée = {r["f0"]:.1f} Hz',
            text=txt_h, hoverinfo='text',
            legendgroup='tau_highlight', showlegend=False,
            visible=False,
        ), row=1, col=1)
        tau_hl_idx.append(len(fig.data) - 1)

    fig.update_xaxes(title_text='τ3 — L-asymétrie', row=1, col=1)
    fig.update_yaxes(title_text='τ4 — L-aplatissement', row=1, col=1)

    # --- Rangée 2 : répartitions par classe, pilotées par le menu f₀ ---------
    # showlegend=True sur TOUTES les traces : Plotly masque les entrées des
    # traces invisibles, donc la légende affiche les RMSE/MSDI de la SEULE
    # fréquence sélectionnée (et non ceux de la première fréquence).
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
                x=maxima_arr, y=ecdf, mode='markers', legend=leg[1],
                marker=dict(size=4, color=color),
                # Nom sur deux lignes : une entrée de légende ne se renvoie
                # pas à la ligne toute seule et déborderait de la marge.
                name=f'Classe {ci} — répartition empirique<br>'
                     f'N = {n} blocs | RMSE = {rmse_s} | MSDI = {msdi_s}',
                legendgroup=f'cdf_class_{ci}',
                hovertemplate='max = %{x:.4g} m/s²<br>F = %{y:.4f}<extra></extra>',
                visible=False, showlegend=True,
            ), row=2, col=1)
            group_idx.append(len(fig.data) - 1)

            if params_i.get('success'):
                try:
                    x_grid = np.linspace(maxima_arr[0], maxima_arr[-1], 200)
                    if params_i.get('loi') == 'rayleigh_gen':
                        a   = float(params_i['rg_alpha'])
                        lam = float(params_i['rg_lambda'])
                        z   = np.clip(lam * x_grid, 0.0, None)
                        cdf_theo = np.power(1.0 - np.exp(-(z * z)), a)
                        nom_loi  = (f"Rayleigh généralisée ajustée<br>"
                                    f"α = {a:.3g} | λ = {lam:.3g}")
                    elif _kappa4_use_exact(params_i['k'], params_i['h']):
                        cdf_theo = _kappa4_cdf_exact(
                            x_grid, params_i['xi'], params_i['alpha'],
                            params_i['k'], params_i['h'])
                        nom_loi  = (f"Kappa4 ajustée<br>k = {params_i['k']:.3f} | "
                                    f"h = {params_i['h']:.3f}")
                    else:
                        cdf_theo = scipy_kappa4(
                            h=params_i['h'], k=params_i['k'],
                            loc=params_i['xi'], scale=params_i['alpha']
                        ).cdf(x_grid)
                        nom_loi  = (f"Kappa4 ajustée<br>k = {params_i['k']:.3f} | "
                                    f"h = {params_i['h']:.3f}")
                    fig.add_trace(go.Scatter(
                        x=x_grid, y=cdf_theo, mode='lines', legend=leg[1],
                        line=dict(color=color, width=2),
                        name=f'Classe {ci} — {nom_loi}',
                        legendgroup=f'cdf_class_{ci}',
                        hovertemplate='x = %{x:.4g} m/s²<br>F = %{y:.4f}'
                                      '<extra></extra>',
                        visible=False, showlegend=True,
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

    fig.update_xaxes(title_text='Maximum par bloc (m/s²)', row=2, col=1)
    fig.update_yaxes(title_text='Probabilité cumulée F', row=2, col=1)

    # --- Rangée 3 : comparatif SRE + MSDI moyen ------------------------------
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

    fig.add_trace(go.Scatter(
        x=f0_list, y=src_v, legend=leg[2],
        name='SRC — max de la réponse (déterministe)',
        line=dict(color='forestgreen', width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sre_k4, legend=leg[2],
        name=f'SRE MBD {loi_lbl} — durée du signal',
        line=dict(color='royalblue', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sre_d, legend=leg[2],
        name='SRE DSP analytique',
        line=dict(color='seagreen', width=2, dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=xl_v, legend=leg[2],
        name=f'SRX risque α={aL_lbl} (dimensionnement)',
        line=dict(color='darkorange', width=2, dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=xh_v, legend=leg[2],
        name=f'SRX risque α={aH_lbl} (comparaison au choc)',
        line=dict(color='firebrick', width=2, dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=msdi_mean, legend=leg[2],
        name='MSDI moyen des classes (axe de droite)',
        mode='lines+markers',
        line=dict(color='mediumorchid', width=1.2, dash='dashdot'),
        marker=dict(size=3),
        hovertemplate='f₀=%{x:.2f} Hz<br>MSDI=%{y:.4g}<extra></extra>'),
        row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text='Fréquence propre f₀ (Hz)', row=3, col=1)
    fig.update_yaxes(title_text='Accélération (m/s²)', row=3, col=1,
                     secondary_y=False)
    fig.update_yaxes(title_text='MSDI (écart loi / mesure)', row=3, col=1,
                     secondary_y=True, showgrid=False)

    # --- Rangée 4 : comparatif SDF projeté + MSDI moyen ---------------------
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
        sdf_mbd_proj    = [proj_d.get(f, {}).get('sdf_proj_tcl', np.nan) for f in f0_list]
        sdf_k4logn_proj = [proj_d.get(f, {}).get('sdf_proj_k4_logn', np.nan) for f in f0_list]
        sdf_dsp_proj    = [proj_d.get(f, {}).get('sdf_proj_dsp', np.nan) for f in f0_list]
        _suffixe_proj   = 'projeté'
    else:
        logger.info("Rapport détails : pas de projection — SDF de la rangée 4 "
                    "affiché en valeurs non projetées.")
        sdf_mbd_proj    = [_get(results_ok, f, 'sdf_kappa4_empirical') for f in f0_list]
        sdf_k4logn_proj = [np.nan] * len(f0_list)
        sdf_dsp_proj    = [sp_d.get(f, np.nan) for f in f0_list]
        _suffixe_proj   = 'NON projeté'

    sdf_mbd_proj    = _clean_pos(sdf_mbd_proj)
    sdf_k4logn_proj = _clean_pos(sdf_k4logn_proj)
    sdf_dsp_proj    = _clean_pos(sdf_dsp_proj)
    sdf_rf_brut     = _clean_pos(sdf_rf_brut)
    sdf_rf_proj     = _clean_pos(sdf_rf_proj)

    fig.add_trace(go.Scatter(
        x=f0_list, y=sdf_mbd_proj, legend=leg[3],
        name=f'SDF MBD {_suffixe_proj}<br>somme des dommages par bloc',
        line=dict(color='orange', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sdf_k4logn_proj, legend=leg[3],
        name=f'SDF MBD {_suffixe_proj}<br>loi sur D_bloc + log-normale',
        line=dict(color='goldenrod', width=2, dash='dot')), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sdf_dsp_proj, legend=leg[3],
        name=f'SDF analytique DSP {_suffixe_proj} (bande étroite)',
        line=dict(color='purple', width=2, dash='dash')), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sdf_rf_brut, legend=leg[3],
        name='SDF rainflow sur le signal (non projeté)',
        line=dict(color='crimson', width=1.2, dash='dot')), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=sdf_rf_proj, legend=leg[3],
        name='SDF rainflow × T_proj/T_signal',
        line=dict(color='crimson', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=f0_list, y=msdi_mean, legend=leg[3],
        name='MSDI moyen des classes (axe de droite)',
        mode='lines+markers',
        line=dict(color='mediumorchid', width=1.2, dash='dashdot'),
        marker=dict(size=3),
        hovertemplate='f₀=%{x:.2f} Hz<br>MSDI=%{y:.4g}<extra></extra>'),
        row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text='Fréquence propre f₀ (Hz)', row=4, col=1)
    fig.update_yaxes(title_text='Dommage (u.a., log)', type='log',
                     row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text='MSDI (écart loi / mesure)', row=4, col=1,
                     secondary_y=True, showgrid=False)

    # --- Rangée 5 : Quality Gate IID ----------------------------------------
    if has_iid:
        if0   = np.asarray(iid_diag['f0'], dtype=float)
        order = np.argsort(if0)
        if0   = if0[order]

        def _ord(branch, key):
            arr = np.asarray(iid_diag.get(branch, {}).get(key, []), dtype=float)
            return arr[order] if arr.size == order.size else None

        for branch, dash, cset, lbl in (
                ('sre', 'solid', ('#1f77b4', '#7fb3d5'), 'maxima (SRE)'),
                ('sdf', 'dot',   ('#d62728', '#e8888a'), 'dommages (SDF)')):
            rho_b  = _ord(branch, 'rho')
            pval_b = _ord(branch, 'pvalue')
            if rho_b is not None:
                fig.add_trace(go.Scatter(
                    x=if0, y=rho_b, legend=leg[4],
                    name=f'ρ Spearman lag-1 — {lbl}',
                    mode='lines', line=dict(color=cset[0], width=1.5, dash=dash)),
                    row=5, col=1)
            if pval_b is not None:
                fig.add_trace(go.Scatter(
                    x=if0, y=pval_b, legend=leg[4],
                    name=f'p-value test des suites — {lbl}',
                    mode='lines', line=dict(color=cset[1], width=1.5, dash=dash)),
                    row=5, col=1)

        # Seuils : annotés à GAUCHE dans le graphe (à droite, ils sortiraient
        # sous la légende de la rangée).
        for yv, txt in ((IID_RHO_MAX,   f'|ρ| max = {IID_RHO_MAX}'),
                        (-IID_RHO_MAX,  None),
                        (IID_PVALUE_MIN, f'p-value min = {IID_PVALUE_MIN}')):
            fig.add_hline(y=yv, line=dict(color='gray', width=1, dash='dash'),
                          annotation_text=txt,
                          annotation_position='top left',
                          annotation_font=dict(size=10, color='gray'),
                          row=5, col=1)
        fig.update_xaxes(title_text='Fréquence propre f₀ (Hz)', row=5, col=1)
        fig.update_yaxes(title_text='ρ lag-1  /  p-value', row=5, col=1)

    # --- Sélecteur de fréquence (rangées 1 et 2) -----------------------------
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
        buttons.append(dict(label=f"f₀ = {f0:.1f} Hz", method='update',
                            args=[{'visible': vis}]))

    updatemenus = []
    if buttons:
        # Placé dans la bande de titre, à gauche : hors de toute zone de tracé
        # et hors de la colonne des légendes.
        updatemenus.append(dict(
            buttons=buttons, direction='down', showactive=True,
            x=0.0, y=1.035, xanchor='left', yanchor='bottom',
            bgcolor='white', bordercolor='#8c8c8c', font=dict(size=11),
        ))
        fig.add_annotation(
            x=0.0, y=1.085, xref='paper', yref='paper', xanchor='left',
            yanchor='bottom', showarrow=False, font=dict(size=11),
            text="<b>Fréquence f₀ affichée</b> dans les graphes 1 et 2 :")

    updatemenus += _boutons_echelle_par_rangee(fig, n_rows_det)
    _nom_fic = (config_info or {}).get('Fichier', '')
    fig.update_layout(
        title=dict(
            text=(f"<b>MBD V3.6 — diagnostic de l'ajustement</b> — {_nom_fic}"
                  f"<br><span style='font-size:11px'>Loi {loi_lbl} · "
                  f"T_b = {TB:g} s · Q = {Q:g} · "
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>"),
            x=0.0, xanchor='left', y=0.985, yanchor='top'),
        height=int(360 * sum(_ru_det)) + 220,
        margin=dict(l=90, r=370, t=220, b=60),
        hovermode='closest',
        updatemenus=updatemenus,
        **layout_leg,
    )
    _titres_rangees_a_gauche(fig, n_rows_det)

    guide = {
        'Diagramme (τ3, τ4)': "Chaque point est un ajustement (une fréquence, "
            "une classe) placé selon l'asymétrie et l'aplatissement de ses "
            "L-moments. Les courbes en pointillés sont les lois Kappa4 à h "
            "fixé ; le trait noir est la limite basse valable pour toute loi. "
            "Un nuage compact signale une forme stable le long du spectre ; un "
            "nuage étalé, une forme qui change d'une fréquence à l'autre.",
        'Limite haute h = −1': "Frontière du domaine d'unicité. Un point mesuré "
            "au-dessus n'est atteignable par aucune Kappa4 : l'ajustement "
            "retenu est alors celui du bord du domaine (h = −1), au point de "
            "la courbe le plus proche du point mesuré ([2] éq. 28).",
        'Répartitions (graphe 2)': "Points = répartition empirique des maxima "
            "par bloc de la classe ; trait = loi ajustée. L'écart se lit dans "
            "la queue droite, celle qui pilote le SRE.",
        'RMSE et MSDI': "Deux mesures d'écart entre loi et mesure, affichées "
            "dans la légende du graphe 2 et en courbe sur les graphes 3 et 4. "
            "Le MSDI ne porte que sur la QUEUE DROITE (points au-dessus de la "
            "moyenne, positions de Cunnane) : c'est l'indicateur pertinent "
            "pour un spectre extrême.",
        'Graphe 5 — indépendance': "ρ de Spearman lag-1 et p-value du test des "
            "suites, fréquence par fréquence. Sortir des seuils en pointillés "
            "signale des blocs corrélés, donc une inférence par L-moments "
            "moins fiable à cette fréquence. ρ doit rester ENTRE les deux "
            "seuils ±|ρ| max ; la p-value doit rester AU-DESSUS de son seuil. "
            "Le bloc « Contrat IID » en tête de page donne l'interprétation : "
            "test en cause, bandes touchées, part du hasard, cause probable "
            "et action à mener.",
        'Sélecteur de fréquence': "En haut à gauche : change la fréquence "
            "affichée dans les graphes 1 et 2. La légende du graphe 2 se met à "
            "jour avec les RMSE / MSDI de la fréquence choisie.",
        'Boutons X/Y lin-log': "En haut à droite de chaque graphe : basculent "
            "l'échelle de l'axe correspondant, graphe par graphe.",
    }
    _ecrire_html_avec_cadre(fig, filepath, config_info, guide_courbes=guide,
                            commentaire_iid=commentaire_iid)
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
    logger.info("  MBD V3.6 — SRC / SRE & SRX / SDF selon NF X50-144-3 Annexe C")
    logger.info("  Loi : %s | classes : %s | projection : %s",
                LOI_AJUSTEMENT,
                ('auto (stationnarité)' if (AUTO_SELECT_K
                                            and AUTO_K_SELON_STATIONNARITE)
                 else ('auto (silhouette)' if AUTO_SELECT_K
                       else f'K={N_CLUSTERS} imposé')),
                METHODE_PROJECTION if ENABLE_PROJECTION else 'désactivée')
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
    # Retrait éventuel du début de signal (cf. TRIM_DEBUT_S, SECTION 1).
    if TRIM_DEBUT_S and TRIM_DEBUT_S > 0:
        n_trim = int(round(float(TRIM_DEBUT_S) * fs))
        n_trim = min(max(n_trim, 0), max(len(signal) - 16, 0))
        if n_trim > 0:
            logger.info("   TRIM_DEBUT_S=%g s → %d échantillons retirés en début "
                        "de signal.", TRIM_DEBUT_S, n_trim)
            t = t[n_trim:] - t[n_trim]
            signal = signal[n_trim:]
    logger.info("   fs=%.1f Hz | %d points | durée=%.2f s", fs, len(signal), t[-1] - t[0])

    # 1bis. Contrôle de stationnarité (SECTION 9ter) — avant tout calcul lourd.
    stat_diag = None
    if STATIONNARITE_ENABLED:
        logger.info("1bis. Contrôle de stationnarité (%d sondes f₀)...",
                    STATIONNARITE_N_SONDES)
        try:
            stat_diag = detecter_non_stationnarite(signal, fs, TB, Q,
                                                   F0_MIN, F0_MAX)
            _journaliser_stationnarite(stat_diag)
        except Exception as _exc_stat:
            logger.warning("Contrôle de stationnarité : erreur ignorée (%s)",
                           _exc_stat)
            stat_diag = None

    # 2. Extraction des features de l'excitation
    # AUTO_SELECT_K sans aucune feature activée : on retient rms (énergie),
    # kurtosis (impulsivité) et crest_factor (forme), qui séparent les régimes
    # d'amplitude sans supposer de contenu spectral particulier.
    _flags = dict(FEATURE_FLAGS)
    if AUTO_SELECT_K and not any(_flags.values()):
        _flags.update({'rms': True, 'kurtosis': True, 'crest_factor': True})
        logger.info("   AUTO_SELECT_K sans feature active : rms, kurtosis et "
                    "crest_factor utilisées pour K-Means.")
    logger.info("2. Extraction des features de l'excitation (Tb=%.3f s)...", TB)
    (features_exc, maxima_exc, used_names, n_blocs_exc,
     taille_bloc_exc) = extraire_caracteristiques(
        signal, fs, Tb_initial=TB, feature_flags=_flags, min_ech=MIN_ECH_PAR_BLOC)
    # Durée de bloc EFFECTIVE (taille entière en échantillons / fs) : c'est
    # elle, et non la consigne TB, qui entre dans M = T_proj / T_b.
    tb_eff = taille_bloc_exc / fs
    logger.info("   %d blocs de %d échantillons (T_b effectif = %.6g s) | "
                "%d échantillons de fin de signal ignorés",
                n_blocs_exc, taille_bloc_exc, tb_eff,
                len(signal) - n_blocs_exc * taille_bloc_exc)
    if abs(tb_eff / TB - 1.0) > 0.01:
        logger.warning("   T_b effectif (%.6g s) ≠ T_b demandé (%g s) : la "
                       "taille de bloc est bornée par MIN_ECH_PAR_BLOC ou par "
                       "la longueur du signal. M est calculé avec la durée "
                       "effective.", tb_eff, TB)
    logger.info("   %d features actives : %s", len(used_names), used_names)

    # 3. Classification K-Means sur l'excitation
    # K = 1 est imposé si le signal est jugé STATIONNAIRE : la silhouette n'est
    # pas définie pour K = 1 et découperait sinon un signal homogène.
    logger.info("3. Classification des blocs...")
    k1_stationnaire = (AUTO_SELECT_K and AUTO_K_SELON_STATIONNARITE
                       and stat_diag is not None
                       and stat_diag['verdict'] == 'STATIONNAIRE')
    if (k1_stationnaire or (N_CLUSTERS == 1 and not AUTO_SELECT_K)
            or features_exc.shape[0] < 2 or len(used_names) == 0):
        clusters = np.zeros(n_blocs_exc, dtype=int)
        n_k_final = 1
        if k1_stationnaire:
            logger.info("   K=1 : signal jugé STATIONNAIRE (AUTO_SELECT_K + "
                        "AUTO_K_SELON_STATIONNARITE).")
        else:
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
    # DELTA_F0 = pas en Hz, pas un nombre de points : le nombre de fréquences
    # se déduit de l'étendue DIVISÉE par le pas.
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
    # blocs avant de faire confiance à l'inférence par L-moments.
    iid_diag = None
    if IID_GATE_ENABLED:
        logger.info("4bis. Quality Gate IID (indépendance statistique)...")
        iid_diag = agreger_quality_gate(all_results, f0_spectrum, TB,
                                        SDF_ENABLED and HAS_RAINFLOW)
        # Croisement avec le contrôle de stationnarité : sur un signal non
        # stationnaire, la dépendance entre blocs vient de l'enveloppe lente,
        # et l'action corrective proposée par la Quality Gate (allonger T_b)
        # serait sans effet.
        if (iid_diag is not None and iid_diag.get('status') != 'GO'
                and stat_diag is not None
                and stat_diag['verdict'] == 'NON_STATIONNAIRE'):
            logger.warning("   Quality Gate IID : le signal est jugé NON "
                           "STATIONNAIRE — la dépendance entre blocs vient de "
                           "l'enveloppe lente d'énergie, pas d'un T_b trop "
                           "court. Préférer les classes (AUTO_SELECT_K=True) à "
                           "un allongement de T_b.")

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
            exclues_proj = []     # classes non vides sans projection valide

            for i, (params, maxima) in enumerate(zip(params_list, maxima_classes)):
                n_blocs_i = len(maxima)
                if isinstance(params, dict) and params.get('success'):
                    n_classes_ok += 1
                if METHODE_PROJECTION == 'gev_domaines':
                    # Projection par max-stabilité GEV (3 domaines, [2] §4.1).
                    p_gev = (params_gev_list[i]
                             if i < len(params_gev_list) else None)
                    sre_p, M, dom = calculer_projection_gev_domaines(
                        p_gev, n_blocs_i, total_blocs, tb_eff,
                        DUREE_PROJECTION, ALFA_PROJECTION)
                else:
                    # Méthode 'puissance' : F^M sur la loi LOI_AJUSTEMENT.
                    sre_p, M = calculer_projection_lmoments(
                        params, maxima, n_blocs_i, total_blocs, tb_eff,
                        DUREE_PROJECTION, ALFA_PROJECTION)
                    dom = None
                if sre_p is not None and np.isfinite(sre_p):
                    n_classes_proj_ok += 1
                    if sre_proj_max is None or sre_p > sre_proj_max:
                        sre_proj_max = sre_p
                        M_best       = M
                        dom_best     = dom
                elif n_blocs_i > 0:
                    exclues_proj.append(i)
            if n_classes_proj_ok == 0:
                exclues_proj = []   # aucune classe valide : trou, pas exclusion

            # Synthèse des classes ([1] §C.10, SECTION 9quater) : quantile du
            # produit Π_j F_j^{M_j}, plus conservatif que le max des quantiles.
            sre_proj_max_classe = sre_proj_max
            synthese = 'max'
            if SYNTHESE_CLASSES == 'produit' and n_classes_proj_ok > 1:
                lois_cl = (params_gev_list if METHODE_PROJECTION == 'gev_domaines'
                           else params_list)
                classes_cl = [(lois_cl[i] if i < len(lois_cl) else None,
                               len(maxima_classes[i]))
                              for i in range(len(maxima_classes))]
                v_prod, ok_prod = quantile_produit_classes(
                    classes_cl, total_blocs, tb_eff, DUREE_PROJECTION,
                    ALFA_PROJECTION)
                if ok_prod and v_prod is not None and np.isfinite(v_prod):
                    sre_proj_max = float(v_prod)
                    M_best = DUREE_PROJECTION / tb_eff
                    synthese = 'produit'

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
                        sdf_blocs, cl_trunc, n_k_final, total_blocs, tb_eff,
                        DUREE_PROJECTION, ALFA_PROJECTION)

                # Projection K4 sur D_bloc (NF X50 144-3 Annexe C).
                params_dmg_list  = r.get('params_dmg_list', [])
                d_blocs_classes  = r.get('d_blocs_classes', [])
                if params_dmg_list and d_blocs_classes:
                    sdf_proj_k4_logn = calculer_projection_dmg_kappa4(
                        params_dmg_list, d_blocs_classes,
                        n_k_final, total_blocs, tb_eff,
                        DUREE_PROJECTION, ALFA_PROJECTION)

            sdf_proj_dsp = sdf_sp_dict.get(f0, np.nan)
            if np.isfinite(sdf_proj_dsp):
                sdf_proj_dsp *= ratio_dsp

            proj_results.append({
                'f0':              f0,
                'sre_proj_max':    sre_proj_max,
                'sre_proj_max_classe': sre_proj_max_classe,
                'synthese_classes': synthese,
                'classes_exclues': exclues_proj,
                'occ_exclue':      (sum(len(maxima_classes[i])
                                        for i in exclues_proj)
                                    / float(max(total_blocs, 1))),
                'M':               M_best,
                'gev_domaine':     dom_best,
                'sdf_proj_tcl':    sdf_proj_tcl,
                'sdf_proj_k4_logn': sdf_proj_k4_logn,
                'sdf_proj_dsp':    sdf_proj_dsp,
            })

    # Classes absentes de la synthèse (ajustement impossible) : avertissement
    # au journal et texte repris dans le cadre des rapports HTML.
    bilans_exclusion = [b for b in (
        _bilan_classes_exclues(successful, "SRE à la durée du signal"),
        _bilan_classes_exclues(proj_results, "SRE projeté")) if b]

    # 7. Exports CSV
    logger.info("7. Export des résultats CSV...")
    duree_mesure = len(signal) / fs if fs else float('nan')
    meta_run = build_run_meta(fs, duree_mesure, n_k_final, num_f0, ts)
    meta_run['synthese_classes'] = SYNTHESE_CLASSES
    meta_run['sdof_amorcage']    = bool(SDOF_AMORCAGE)
    meta_run['kappa4_h_min']     = KAPPA4_H_MIN
    meta_run['trim_debut_s']     = float(TRIM_DEBUT_S)
    meta_run['Tb_effectif_s']    = float(tb_eff)
    meta_run['n_blocs']          = int(n_blocs_exc)
    if stat_diag is not None:
        meta_run['stationnarite'] = {
            k: v for k, v in stat_diag.items() if k not in ('sondes',)}

    tag = (f"v3p5pub_f{F0_MIN:.0f}-{F0_MAX:.0f}"
           f"_Q{Q}_Tb{_fmt_compact(TB)}"
           f"_b{_fmt_compact(SDF_B)}"
           f"_T{_fmt_duration_compact(DUREE_PROJECTION)}"
           f"_a{_fmt_compact(ALFA_PROJECTION)}")

    meta_json = os.path.join(run_dir, f"params_{tag}_{ts}.json")
    exporter_run_meta_json(meta_json, meta_run)

    if stat_diag is not None:
        exporter_csv_stationnarite(
            os.path.join(run_dir, f"Stationnarite_{tag}_{ts}.csv"),
            stat_diag, meta_run)

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
        'Tb':            (f"{TB} s — {n_blocs_exc} blocs de {taille_bloc_exc} "
                          "échantillons"),
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
        'Amorçage 1-DDL': ('actif (%.0f τ)' % SDOF_AMORCAGE_N_TAU
                           if SDOF_AMORCAGE else 'inactif'),
    }
    if n_k_final > 1:
        config_info['Synthèse classes'] = (
            "produit des répartitions Π F_j^{M_j} ([1] §C.10, [2] éq. 37), "
            "à la durée du signal comme en projection"
            if SYNTHESE_CLASSES == 'produit'
            else "max des quantiles par classe (non conservatif)")
    if bilans_exclusion:
        config_info['⚠ Classes exclues'] = (
            "<b>Synthèse des classes incomplète — spectre sous-estimé aux "
            "fréquences concernées.</b> " + " ; ".join(bilans_exclusion)
            + ". Détail par fréquence : colonne Classes_exclues des CSV.")
    if stat_diag is not None:
        config_info['Stationnarité'] = (
            f"{stat_diag['verdict']} "
            f"({'gaussien' if stat_diag['gaussien'] else 'non gaussien'}) — "
            f"sondes IID en échec {100.0 * stat_diag['frac_sondes_echec_iid']:.0f} %, "
            f"CV RMS/bloc {stat_diag['cv_rms_blocs']:.3f}, "
            f"kurtosis {stat_diag['kurtosis']:.2f}")
        config_info['Recommandation'] = stat_diag['recommandation']
    if iid_diag is not None:
        _ico = {'GO': '🟢', 'WARNING': '🟡', 'NO-GO': '🔴'}.get(
            iid_diag['status'], '')
        config_info['Quality Gate IID'] = (
            f"{_ico} {iid_diag['status']} "
            f"({100.0 * iid_diag['frac_fail']:.1f}% f₀ hors-tolérance) — "
            "voir le bloc « Contrat IID » ci-dessous")

    # Explication en clair de la Quality Gate IID, commune aux deux rapports.
    commentaire_iid = None
    if iid_diag is not None:
        try:
            commentaire_iid = commenter_quality_gate(
                iid_diag, all_results, TB, Q, duree_mesure=duree_mesure,
                stat_diag=stat_diag, tb_effectif=tb_eff)
        except Exception as _exc_iid:
            logger.warning("Commentaire Quality Gate IID : erreur ignorée (%s)",
                           _exc_iid)

    generer_html(html_path, all_results, sre_dsp, srx_low, srx_high, sdf_spectral,
                 f0_spectrum, proj_results=proj_results or None,
                 config_info=config_info,
                 sre_dsp_proj=sre_dsp_proj,
                 srx_low_proj=srx_low_proj,
                 srx_high_proj=srx_high_proj,
                 alpha_srx_low=ALPHA_SRX_LOW,
                 alpha_srx_high=ALPHA_SRX_HIGH,
                 iid_diag=iid_diag,
                 commentaire_iid=commentaire_iid)

    html_details = os.path.join(run_dir, f"Rapport_Details_{tag}_{ts}.html")
    _t_mesure = len(signal) / fs if fs else None
    generer_html_details(html_details, all_results, sre_dsp, srx_low, srx_high,
                         sdf_spectral, f0_spectrum, config_info=config_info,
                         proj_results=proj_results or None,
                         alpha_srx_low=ALPHA_SRX_LOW,
                         alpha_srx_high=ALPHA_SRX_HIGH,
                         t_mesure=_t_mesure,
                         t_proj=DUREE_PROJECTION if ENABLE_PROJECTION else None,
                         iid_diag=iid_diag,
                         commentaire_iid=commentaire_iid)

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
