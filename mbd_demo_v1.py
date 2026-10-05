# -*- coding: utf-8 -*-
"""
mbd_demo_v1.py — Démonstrateur MBD complet (gaussien / non-gaussien / instationnaire)
=====================================================================================

Programme COURT qui réutilise au maximum `mbd_simple-multi-process_v3_6.py` :
il génère un signal d'excitation (stationnaire ou non-stationnaire) puis lance
le calcul MBD complet via la fonction `main()` du module principal — donc avec
EXACTEMENT les mêmes visuels et la même sortie (rapports HTML interactifs, CSV
suffixés, sidecar JSON). À chaque exécution, toute la sortie atterrit dans un
sous-dossier horodaté indépendant créé par `main()` ; la démo y range en plus le
signal généré et un fichier `RESUME_demo.txt`.

Les signaux générés ayant une nature CONNUE (stationnaire ou non, gaussien ou
non), la démo sert aussi à voir réagir les contrôles de la V3.5pub : verdict de
stationnarité, choix automatique du nombre de classes, contrat IID commenté
dans les rapports HTML, synthèse des classes par produit des répartitions.

Trois familles de signaux (paramètre MODE_SIGNAL) :
  - 'stationnaire' : PSD plate → phases aléatoires → iFFT → variance unité →
        transformation non-linéaire ZMNL (Hermite) pilotée par skewness S et
        kurtosis K cibles (article CFM 2025 — Clou/Lelan §4.3) :
            y = h·[ z + a·(z² − 1) + b·(z³ − 3z) ]
            a = S / (4 + 2·√(1 + 1.5·(K − 3)))
            b = (√(1 + 1.5·(K − 3)) − 1) / 18
            h = 1 / √(1 + 2a² + 6b²)
        K = 3 → gaussien ; K > 3 → non-gaussien (queues lourdes).
  - 'phases'      : concaténation de segments stationnaires aux statistiques
        distinctes (RMS / kurtosis / skewness) → instationnarité PAR BLOCS,
        que la classification K-Means doit retrouver comme classes localement
        stationnaires.
  - 'enveloppe'   : signal de base de variance unité multiplié par une enveloppe
        RMS(t) lentement variable (sinus ou marche aléatoire) → instationnarité
        CONTINUE.

Usage :
    python mbd_demo_v1.py
Modifier les paramètres dans les blocs CONFIG ci-dessous (sections A à D).

Note multiprocess : la démo applique ses surcharges de configuration AU NIVEAU
MODULE (et enregistre le module principal dans sys.modules) afin que les workers
créés par « spawn » (Windows) héritent bien des mêmes réglages, y compris
LOI_AJUSTEMENT et les paramètres experts qui ne transitent pas par les arguments
de tâche. On peut donc régler USE_MULTIPROCESS=True en toute sécurité.
"""

# --- Limitation BLAS à 1 thread AVANT import numpy (cf. module principal §4.9) ---
# Évite la sur-souscription quand USE_MULTIPROCESS=True (chaque worker tenterait
# sinon d'utiliser tous les cœurs). Posé ici car la démo importe numpy en premier.
import os
for _ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_ev, "1")

import sys

# Console en UTF-8 : les sorties contiennent des symboles non-ASCII (≈, m/s², f₀,
# verdicts 🟢/🟡/🔴 émis par le module principal). Sans cela, un terminal Windows
# en cp1252 lèverait UnicodeEncodeError. reconfigure() modifie le flux en place,
# donc le logging déjà configuré du module principal en bénéficie aussi.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import shutil
import glob
import importlib.util
from datetime import datetime

import numpy as np


# =============================================================================
# SECTION A — GÉNÉRATION DU SIGNAL
# =============================================================================
# MODE_SIGNAL sélectionne la famille de signal généré :
#   'stationnaire' | 'phases' | 'enveloppe'  (voir docstring).
MODE_SIGNAL = 'phases'

# --- Paramètres communs à tous les modes ---
FS          = 5000.0     # fréquence d'échantillonnage (Hz)
PSD_F_MIN   = 5.0        # bande de la PSD plate (Hz)
PSD_F_MAX   = 800.0
SEED        = 12345      # graine du générateur de signal (reproductibilité)

# --- Mode 'stationnaire' (sert aussi de signal de base au mode 'enveloppe') ---
DUREE       = 300.0    # durée du signal généré (s) — réduire (ex. 600) pour un essai rapide
RMS_CIBLE   = 30.0       # écart-type visé de l'excitation (m/s²)
KURTOSIS    = 3.0        # 3.0 = gaussien ; > 3 = non-gaussien (queues lourdes)
SKEWNESS    = 0.0        # 0 = symétrique

# --- Mode 'phases' (instationnarité PAR BLOCS) ---
# Liste de segments mis bout à bout, chacun stationnaire mais de statistiques
# propres. Tuple = (durée_s, rms_cible, kurtosis, skewness). La durée totale est
# la somme des durées de phase (DUREE ci-dessus est ignorée en mode 'phases').
PHASES = [
    # (durée_s,  rms,   kurtosis, skewness)
    (600.0,    7.0,  3.0,      0.0),   # ex. roulage doux (gaussien, RMS faible)
    (600.0,    10.0,  4.0,      0.0),   # ex. piste sévère (gaussien, RMS fort)
    (600.0,    3.0,  2.5,      0.0),   # ex. chocs/cailloux (non-gaussien)
]

# --- Mode 'enveloppe' (instationnarité CONTINUE) ---
# Signal de base de variance unité (mêmes KURTOSIS/SKEWNESS que le mode
# stationnaire, sur DUREE) modulé par une enveloppe RMS(t) ∈ [RMS_MIN, RMS_MAX].
ENVELOPPE   = 'sinus'    # 'sinus' (période PERIODE_ENV) | 'marche_alea' (lissée)
RMS_MIN     = 15.0       # RMS minimal de l'enveloppe (m/s²)
RMS_MAX     = 45.0       # RMS maximal de l'enveloppe (m/s²)
PERIODE_ENV = 6000.0     # période (sinus) / échelle de temps (marche aléatoire), en s


# =============================================================================
# SECTION B — CLASSIFICATION K-MEANS  ⚠ POUR SIGNAUX NON-STATIONNAIRES
# =============================================================================
# Ce bloc N'A D'INTÉRÊT QUE pour les signaux NON-STATIONNAIRES
# (MODE_SIGNAL='phases' ou 'enveloppe'). Il partitionne les blocs Tb en classes
# localement stationnaires sur lesquelles le fit Kappa4/Rayleigh est mené
# séparément, puis synthétisé (SRE = synthèse SYNTHESE_CLASSES des classes,
# SDF = somme des classes).
#
# >>> EN MODE 'stationnaire', GARDER N_CLUSTERS = 1 (pas de partition), ou
#     laisser faire AUTO_SELECT_K=True : le contrôle de stationnarité (SECTION
#     B bis) impose alors K = 1 de lui-même.
#
# Pour un signal non-stationnaire :
#   - mettre N_CLUSTERS ≈ nombre de régimes attendus (ex. len(PHASES)),
#     OU AUTO_SELECT_K = True pour le choisir automatiquement : K = 1 si le
#     signal est jugé STATIONNAIRE (AUTO_K_SELON_STATIONNARITE), sinon K
#     maximisant le score Silhouette dans K_RANGE ;
#   - activer dans FEATURE_FLAGS les descripteurs qui DISCRIMINENT les régimes :
#     'variance'/'rms' (changement de niveau), 'kurtosis'/'crest_factor'
#     (impulsivité), 'dominant_freq'/'spectral_centroid' (contenu fréquentiel).
#     Avec AUTO_SELECT_K=True et AUCUNE feature activée, le module principal
#     utilise d'office rms + kurtosis + crest_factor ;
#   - veiller à conserver assez de blocs par classe : il faut
#     n_blocs ≳ N_CLUSTERS × MIN_SAMPLES_PER_CLUSTER (≥ 40 par classe pour Kappa4).
#     Une classe trop petite pour être ajustée est EXCLUE de la synthèse : le
#     spectre est alors sous-estimé, et le cas est signalé (journal, cadre des
#     rapports HTML, colonne Classes_exclues des CSV).
#
# SYNTHESE_CLASSES : 'produit' (défaut) = quantile du produit des répartitions
#   Π_j F_j^{M_j} (NF X50-144-3 §C.10, Colin 2023 éq. 37), à la durée du signal
#   comme en projection ; 'max' = maximum des quantiles par classe (toujours
#   inférieur ou égal, donc non conservatif).
N_CLUSTERS                 = 3
AUTO_SELECT_K              = False
AUTO_K_SELON_STATIONNARITE = True         # K = 1 si verdict STATIONNAIRE
K_RANGE                    = range(3, 9)  # plage testée si AUTO_SELECT_K=True
MIN_SAMPLES_PER_CLUSTER    = 40
SYNTHESE_CLASSES           = 'produit'    # 'produit' | 'max'

# Activer/désactiver chaque feature (centrée-réduite avant K-Means). Les 12 clés
# doivent être présentes. En mode 'stationnaire' elles sont sans effet (K=1). True
FEATURE_FLAGS = {
    'mean': True,            'variance': True,        'skewness': True,
    'kurtosis': True,        'rms': True,             'mav': False,
    'crest_factor': False,    'autocorr_lag1': False,   'zcr': False,
    'dominant_freq': False,   'spectral_centroid': False, 'spectral_spread': False,
}


# =============================================================================
# SECTION B bis — CONTRÔLE DE STATIONNARITÉ  (calculé AVANT la boucle f₀)
# =============================================================================
# Verdict STATIONNAIRE / NON_STATIONNAIRE / DOUTEUX, à partir de deux
# indicateurs : (a) fraction de fréquences sondes dont les maxima par bloc
# échouent au contrat IID, (b) coefficient de variation du RMS par bloc. Le
# signal est jugé NON stationnaire si les DEUX dépassent leur seuil. Le verdict
# est journalisé, exporté (Stationnarite_*.csv, params_*.json) et repris dans
# les rapports HTML avec une recommandation.
#   - MODE_SIGNAL='stationnaire' doit donner STATIONNAIRE (gaussien si
#     KURTOSIS = 3 et SKEWNESS = 0) ;
#   - 'phases' et 'enveloppe' doivent donner NON_STATIONNAIRE dès que les
#     niveaux RMS diffèrent nettement.
STATIONNARITE_ENABLED     = True
STATIONNARITE_N_SONDES    = 8      # fréquences sondes (4–20)
STATIONNARITE_FRAC_ECHEC  = 0.3    # seuil de l'indicateur (a) (0,1–0,5)
STATIONNARITE_CV_RMS_MAX  = 0.15   # seuil de l'indicateur (b) (0,05–0,40)
STATIONNARITE_KURT_TOL    = 0.5    # |kurtosis − 3| toléré pour « gaussien »
STATIONNARITE_SKEW_TOL    = 0.3    # |asymétrie| tolérée pour « gaussien »


# =============================================================================
# SECTION C — OPTIONS DE CALCUL MBD  (miroir de la SECTION 1 du module principal)
# =============================================================================
# Loi d'ajustement statistique : 'kappa4' (Hosking 4 paramètres) ou
# 'rayleigh_gen' (Rayleigh généralisée, Kundu & Raqab — PPF analytique exacte).
LOI_AJUSTEMENT      = 'kappa4'

# SDOF étalon
Q                   = 10        # coefficient de surtension Q = 1/(2ξ) ; 5–50
TB                  = 1.0       # durée de bloc T_b (s) ; 0.05–10. Un bloc compte
                                # round(T_b·fs) échantillons ; viser
                                # T_b ≥ 3·Q/(π·F0_MIN) pour le contrat IID.
TRIM_DEBUT_S        = 0.0       # durée (s) retirée en début de signal (0 = rien)
SDOF_AMORCAGE       = True      # (expert) amorce l'oscillateur sur le début du
                                # signal renversé : pas d'oscillation libre
                                # parasite dans le premier bloc
SDOF_AMORCAGE_N_TAU = 8.0       # (expert) durée d'amorçage, en constantes de temps

# Spectre de fréquences propres f₀
F0_MIN              = 10        # ≥ 1/T_b recommandé
F0_MAX              = 400       # < fs/4 (anti-repliement)
DELTA_F0            = 1.0       # pas en Hz

# Quantile cible (PPF) — Cunnane recommandé par la norme
PROBABILITE_CIBLE   = 0.9       # pris si OPTION_CUNNANE=False ; P de non-dépassement
OPTION_CUNNANE      = True      # p_eff = (N-a)/(N+1-2a)
CUNNANE_A           = 0.4

# SRX analytique DSP — risque α élevé (comparaison vs SRC d'un choc).
# (α_low est DÉRIVÉ de ALFA_PROJECTION ci-dessous, comme dans le module principal.)
ALPHA_SRX_HIGH      = 0.99

# SDF (Spectre de Dommage par Fatigue) — Basquin/Miner + rainflow (Numba requis)
SDF_ENABLED         = True      # passe à False automatiquement si Numba absent
SDF_B               = 8.0       # pente de Basquin b (3–14 selon matériau)
SDF_C               = 1.0       # constante de Basquin C (analyse relative)

# Projection longue durée
ENABLE_PROJECTION   = True
DUREE_PROJECTION    = 36_000_000   # T_v en s (≈ 1 an d'usage)
ALFA_PROJECTION     = 0.90         # P de non-dépassement projetée (risque α = 1 − valeur)
METHODE_PROJECTION  = 'puissance'  # 'puissance' | 'gev_domaines'
GEV_GUMBEL_K_TOL    = 0.01         # (expert) seuil d'étiquetage du domaine Gumbel

# Quality Gate IID (indépendance des blocs)
IID_GATE_ENABLED    = True
IID_RHO_MAX         = 0.2
IID_PVALUE_MIN      = 0.05
IID_FAIL_FRAC_MAX   = 0.05
IID_NOGO_FRAC       = 0.30
IID_MIN_N           = 20

# Divers / numérique
MIN_POINTS_KAPPA4   = 40
RANDOM_SEED         = 53        # graine RNG du module principal (KMeans, etc.)
KAPPA4_DEBUG_ANALYTIC = False   # True → exporte Kappa4_Debug_*.csv
KAPPA4_H_MIN        = -1.0      # (expert) borne basse de h : domaine d'unicité de
                                # la Kappa4. Un point (τ3, τ4) hors domaine est
                                # ramené à la loi du bord (fit « ok_h_borne »).


# =============================================================================
# SECTION D — RUNTIME / SORTIE
# =============================================================================
MODULE_PATH      = "mbd_simple-multi-process_v3_6.py"  # module principal réutilisé
OUTPUT_FOLDER    = "mbd_demo_output"                      # dossier racine des runs démo
USE_MULTIPROCESS = True                                   # boucle f₀ parallélisée
N_WORKERS        = 10                                     # workers (None = auto)
# =============================================================================


def _charger_module(path):
    """Charge le module principal (nom à tirets → non importable directement).

    Enregistre le module dans ``sys.modules`` AVANT exécution pour que les
    workers « spawn » (Windows) puissent le retrouver lors du dé-picklage des
    fonctions de travail (`_mp_worker_traiter`, `traiter_f0`)."""
    here = os.path.dirname(os.path.abspath(__file__))
    abs_path = path if os.path.isabs(path) else os.path.join(here, path)
    spec = importlib.util.spec_from_file_location("mbd_main", abs_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # spawn-safe
    spec.loader.exec_module(mod)
    return mod


def _appliquer_config(mod):
    """Pousse les réglages des sections B/C/D dans les globals du module
    principal. Appelé AU NIVEAU MODULE (donc ré-exécuté dans chaque worker
    spawn) pour que tous les réglages — y compris LOI_AJUSTEMENT — soient
    honorés en multiprocess."""
    # --- SECTION B : classification ---
    mod.N_CLUSTERS                 = N_CLUSTERS
    mod.AUTO_SELECT_K              = AUTO_SELECT_K
    mod.AUTO_K_SELON_STATIONNARITE = AUTO_K_SELON_STATIONNARITE
    mod.K_RANGE                    = K_RANGE
    mod.MIN_SAMPLES_PER_CLUSTER    = MIN_SAMPLES_PER_CLUSTER
    mod.SYNTHESE_CLASSES           = SYNTHESE_CLASSES
    mod.FEATURE_FLAGS              = dict(FEATURE_FLAGS)

    # --- SECTION B bis : contrôle de stationnarité ---
    mod.STATIONNARITE_ENABLED    = STATIONNARITE_ENABLED
    mod.STATIONNARITE_N_SONDES   = STATIONNARITE_N_SONDES
    mod.STATIONNARITE_FRAC_ECHEC = STATIONNARITE_FRAC_ECHEC
    mod.STATIONNARITE_CV_RMS_MAX = STATIONNARITE_CV_RMS_MAX
    mod.STATIONNARITE_KURT_TOL   = STATIONNARITE_KURT_TOL
    mod.STATIONNARITE_SKEW_TOL   = STATIONNARITE_SKEW_TOL

    # --- SECTION C : options de calcul MBD ---
    mod.LOI_AJUSTEMENT    = LOI_AJUSTEMENT
    mod.Q                 = Q
    mod.TB                = TB
    mod.TRIM_DEBUT_S      = TRIM_DEBUT_S
    mod.SDOF_AMORCAGE     = SDOF_AMORCAGE
    mod.SDOF_AMORCAGE_N_TAU = SDOF_AMORCAGE_N_TAU
    mod.KAPPA4_H_MIN      = KAPPA4_H_MIN
    mod.F0_MIN            = F0_MIN
    mod.F0_MAX            = F0_MAX
    mod.DELTA_F0          = DELTA_F0
    mod.PROBABILITE_CIBLE = PROBABILITE_CIBLE
    mod.OPTION_CUNNANE    = OPTION_CUNNANE
    mod.CUNNANE_A         = CUNNANE_A
    mod.ALPHA_SRX_HIGH    = ALPHA_SRX_HIGH
    mod.SDF_ENABLED       = SDF_ENABLED
    mod.SDF_B             = SDF_B
    mod.SDF_C             = SDF_C
    mod.ENABLE_PROJECTION = ENABLE_PROJECTION
    mod.DUREE_PROJECTION  = DUREE_PROJECTION
    mod.ALFA_PROJECTION   = ALFA_PROJECTION
    # IMPORTANT : ALPHA_SRX_LOW est calculé UNE FOIS à l'import du module
    # principal (= 1 − ALFA_PROJECTION). Comme on surcharge ALFA_PROJECTION ici,
    # il faut le recalculer, sinon le SRX α_low ne suivrait pas.
    mod.ALPHA_SRX_LOW     = 1.0 - ALFA_PROJECTION
    mod.METHODE_PROJECTION = METHODE_PROJECTION
    mod.GEV_GUMBEL_K_TOL  = GEV_GUMBEL_K_TOL
    mod.IID_GATE_ENABLED  = IID_GATE_ENABLED
    mod.IID_RHO_MAX       = IID_RHO_MAX
    mod.IID_PVALUE_MIN    = IID_PVALUE_MIN
    mod.IID_FAIL_FRAC_MAX = IID_FAIL_FRAC_MAX
    mod.IID_NOGO_FRAC     = IID_NOGO_FRAC
    mod.IID_MIN_N         = IID_MIN_N
    mod.MIN_POINTS_KAPPA4 = MIN_POINTS_KAPPA4
    mod.RANDOM_SEED       = RANDOM_SEED
    mod.KAPPA4_DEBUG_ANALYTIC = KAPPA4_DEBUG_ANALYTIC

    # --- SECTION D : runtime / sortie ---
    mod.OUTPUT_FOLDER     = OUTPUT_FOLDER
    mod.USE_MULTIPROCESS  = USE_MULTIPROCESS
    mod.N_WORKERS         = N_WORKERS


# --- Chargement + application de la config AU NIVEAU MODULE (avant tout fork) ---
m = _charger_module(MODULE_PATH)
_appliquer_config(m)


# =============================================================================
# Générateurs de signal
# =============================================================================
def _gen_segment(n, fs, f_min, f_max, rms_cible, kurtosis, skewness, rng):
    """Un segment stationnaire (PSD plate + ZMNL Hermite). Renvoie y (m/s²).

    Cœur partagé par les trois modes (article CFM 2025 §4.3)."""
    n = int(n)
    n += n % 2  # n pair pour rfft/irfft

    # 1. PSD plate → spectre à phases aléatoires → iFFT → signal gaussien.
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= f_min) & (freqs <= f_max)
    spec = np.zeros(len(freqs), dtype=complex)
    spec[mask] = (rng.standard_normal(mask.sum())
                  + 1j * rng.standard_normal(mask.sum()))
    z = np.fft.irfft(spec, n=n)

    # 2. Normalisation à variance unité (gaussien standard).
    z = (z - z.mean()) / z.std()

    # 3. Transformation ZMNL (Hermite) : impose skewness/kurtosis cibles.
    K, S = float(kurtosis), float(skewness)
    racine = np.sqrt(max(1.0 + 1.5 * (K - 3.0), 0.0))
    a = S / (4.0 + 2.0 * racine) if (4.0 + 2.0 * racine) != 0 else 0.0
    b = (racine - 1.0) / 18.0
    h = 1.0 / np.sqrt(1.0 + 2.0 * a * a + 6.0 * b * b)
    y = h * (z + a * (z * z - 1.0) + b * (z**3 - 3.0 * z))

    # 4. Correction de variance → RMS physique visé.
    y = (y - y.mean()) / y.std() * rms_cible
    return y


def _stats_lignes(prefix, y):
    """Ligne de diagnostic « RMS / skew / kurt » pour le RESUME."""
    from scipy.stats import kurtosis as _k, skew as _s
    return (f"{prefix}RMS={float(np.std(y)):.3f} m/s²  "
            f"skew≈{float(_s(y)):.3f}  kurt≈{float(_k(y, fisher=False)):.3f}")


def generer_signal_stationnaire(fs, duree, f_min, f_max, rms_cible,
                                 kurtosis, skewness, seed=0):
    """Signal stationnaire gaussien/non-gaussien. Renvoie (t, y, lignes)."""
    rng = np.random.default_rng(seed)
    n = int(round(fs * duree))
    y = _gen_segment(n, fs, f_min, f_max, rms_cible, kurtosis, skewness, rng)
    t = np.arange(len(y)) / fs
    lignes = [
        f"Mode               : stationnaire",
        f"Durée              : {len(y) / fs:.1f} s ({len(y)} pts à {fs:.0f} Hz)",
        f"Cibles             : RMS={rms_cible} m/s²  K={kurtosis}  S={skewness}",
        _stats_lignes("Atteint            : ", y),
    ]
    return t, y, lignes


def generer_signal_phases(fs, phases, f_min, f_max, seed=0):
    """Signal non-stationnaire par concaténation de phases stationnaires.
    Renvoie (t, y, lignes)."""
    segments, lignes = [], [f"Mode               : phases ({len(phases)} segments concaténés)"]
    for i, (duree_i, rms_i, kurt_i, skew_i) in enumerate(phases):
        rng = np.random.default_rng(seed + 1 + i)
        n_i = int(round(fs * duree_i))
        seg = _gen_segment(n_i, fs, f_min, f_max, rms_i, kurt_i, skew_i, rng)
        segments.append(seg)
        lignes.append(f"  phase {i + 1:>2} : {duree_i:>8.0f} s | "
                      f"cibles RMS={rms_i} K={kurt_i} S={skew_i} | "
                      + _stats_lignes("atteint ", seg))
    y = np.concatenate(segments)
    t = np.arange(len(y)) / fs
    lignes.append(f"Durée totale       : {len(y) / fs:.1f} s ({len(y)} pts à {fs:.0f} Hz)")
    return t, y, lignes


def generer_signal_enveloppe(fs, duree, f_min, f_max, kurtosis, skewness,
                             enveloppe, rms_min, rms_max, periode_env, seed=0):
    """Signal non-stationnaire continu = base unité × enveloppe RMS(t).
    Renvoie (t, y, lignes)."""
    rng = np.random.default_rng(seed)
    n = int(round(fs * duree))
    base = _gen_segment(n, fs, f_min, f_max, 1.0, kurtosis, skewness, rng)
    t = np.arange(len(base)) / fs

    if enveloppe == 'sinus':
        # Oscillation lente RMS_MIN → RMS_MAX, période PERIODE_ENV (départ à RMS_MIN).
        env = rms_min + (rms_max - rms_min) * 0.5 * (1.0 - np.cos(2.0 * np.pi * t / periode_env))
    elif enveloppe == 'marche_alea':
        # Marche aléatoire à pas ~ PERIODE_ENV, interpolée puis normalisée dans la plage.
        n_knots = max(2, int(round((t[-1] if len(t) else 0.0) / periode_env)) + 1)
        knots = np.cumsum(rng.standard_normal(n_knots))
        knots = (knots - knots.min()) / (np.ptp(knots) + 1e-12)
        tk = np.linspace(0.0, t[-1] if len(t) else 0.0, n_knots)
        env = rms_min + (rms_max - rms_min) * np.interp(t, tk, knots)
    else:
        raise ValueError(f"ENVELOPPE inconnue : {enveloppe!r} (attendu 'sinus' ou 'marche_alea')")

    y = base * env
    lignes = [
        f"Mode               : enveloppe ({enveloppe})",
        f"Durée              : {len(y) / fs:.1f} s ({len(y)} pts à {fs:.0f} Hz)",
        f"Enveloppe RMS(t)   : [{rms_min}, {rms_max}] m/s²  période/échelle={periode_env} s",
        f"Base               : K={kurtosis}  S={skewness}",
        _stats_lignes("Atteint (global)   : ", y),
    ]
    return t, y, lignes


def construire_signal():
    """Dispatch selon MODE_SIGNAL. Renvoie (t, y, lignes)."""
    if MODE_SIGNAL == 'stationnaire':
        return generer_signal_stationnaire(FS, DUREE, PSD_F_MIN, PSD_F_MAX,
                                           RMS_CIBLE, KURTOSIS, SKEWNESS, seed=SEED)
    if MODE_SIGNAL == 'phases':
        return generer_signal_phases(FS, PHASES, PSD_F_MIN, PSD_F_MAX, seed=SEED)
    if MODE_SIGNAL == 'enveloppe':
        return generer_signal_enveloppe(FS, DUREE, PSD_F_MIN, PSD_F_MAX,
                                        KURTOSIS, SKEWNESS, ENVELOPPE,
                                        RMS_MIN, RMS_MAX, PERIODE_ENV, seed=SEED)
    raise ValueError(f"MODE_SIGNAL inconnu : {MODE_SIGNAL!r} "
                     f"(attendu 'stationnaire', 'phases' ou 'enveloppe')")


def ecrire_csv(path, t, y, skip_rows=10, delimiter=";"):
    """Écrit le signal au format attendu par importer_signal_csv :
    `skip_rows` lignes d'en-tête puis 2 colonnes (temps;accel)."""
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(skip_rows):
            fh.write(f"# entete demo ligne {i + 1}\n")
        # savetxt : même format qu'une écriture ligne à ligne, bien plus rapide
        # sur les signaux de plusieurs millions de points.
        np.savetxt(fh, np.column_stack([t, y]), fmt="%.8f", delimiter=delimiter)


# =============================================================================
# Aides : avertissements de cohérence, dossier de run, RESUME, synthèse
# =============================================================================
def _sous_dossiers(folder):
    if not os.path.isdir(folder):
        return set()
    return {d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))}


def _avertissements_sanity(fs, n_pts):
    """Signale (sans bloquer) les réglages physiquement douteux."""
    msgs = []
    if F0_MAX > fs / 4.0:
        msgs.append(f"F0_MAX={F0_MAX} > fs/4={fs / 4.0:.0f} Hz → risque de repliement.")
    # Critère du contrat IID : sous f_mem, la constante de temps de l'oscillateur
    # τ = Q/(π·f₀) dépasse T_b/3 et les blocs successifs ne sont plus indépendants.
    f_mem = 3.0 * Q / (np.pi * TB) if TB > 0 else float('inf')
    if F0_MIN > 0 and F0_MIN < f_mem:
        msgs.append(f"TB={TB}s court pour F0_MIN={F0_MIN} Hz : sous {f_mem:.1f} Hz "
                    f"les blocs se recouvrent (contrat IID). Viser "
                    f"T_b ≥ 3·Q/(π·F0_MIN) = {3.0 * Q / (np.pi * F0_MIN):.2f} s.")
    # Un bloc compte round(T_b·fs) échantillons ; la fin incomplète est ignorée.
    taille_bloc = max(int(round(TB * fs)), 1)
    n_blocs = n_pts // taille_bloc
    if MODE_SIGNAL != 'stationnaire' and N_CLUSTERS == 1 and not AUTO_SELECT_K:
        msgs.append("signal non-stationnaire mais N_CLUSTERS=1 → pas de classification "
                    "(voir SECTION B).")
    if AUTO_SELECT_K and AUTO_K_SELON_STATIONNARITE and not STATIONNARITE_ENABLED:
        msgs.append("AUTO_K_SELON_STATIONNARITE sans contrôle de stationnarité "
                    "(STATIONNARITE_ENABLED=False) : K sera choisi par la "
                    "silhouette seule, qui découpe même un signal homogène.")
    if (not AUTO_SELECT_K and N_CLUSTERS > 1
            and n_blocs < N_CLUSTERS * MIN_SAMPLES_PER_CLUSTER):
        msgs.append(f"~{n_blocs} blocs pour {N_CLUSTERS} classes × "
                    f"{MIN_SAMPLES_PER_CLUSTER} mini : K sera décrémenté automatiquement.")
    if MIN_SAMPLES_PER_CLUSTER < MIN_POINTS_KAPPA4:
        msgs.append(f"MIN_SAMPLES_PER_CLUSTER={MIN_SAMPLES_PER_CLUSTER} < "
                    f"MIN_POINTS_KAPPA4={MIN_POINTS_KAPPA4} : une classe peut être "
                    f"trop petite pour être ajustée, donc exclue de la synthèse.")
    for msg in msgs:
        print(f"  [avertissement] {msg}")


def _lire_meta(run_dir):
    """Paramètres et diagnostics du run, relus dans le sidecar params_*.json
    écrit par le module principal. Dict vide si le fichier est absent."""
    import json
    fichiers = glob.glob(os.path.join(run_dir, "params_*.json"))
    if not fichiers:
        return {}
    try:
        with open(fichiers[0], encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _lignes_controles(meta):
    """Lignes « ce que le calcul a constaté » : découpage, nombre de classes
    retenu, verdict de stationnarité. Communes au RESUME et à la console."""
    if not meta:
        return []
    lignes = []
    if 'n_blocs' in meta:
        lignes.append(f"Découpage          : {meta['n_blocs']} blocs de "
                      f"{meta.get('Tb_effectif_s', float('nan')):.6g} s")
    if 'K_means' in meta:
        lignes.append(f"Classes retenues   : K = {meta['K_means']} "
                      f"(synthèse : {meta.get('synthese_classes', '?')})")
    stat = meta.get('stationnarite')
    if stat:
        lignes.append(
            f"Stationnarité      : {stat.get('verdict')} "
            f"({'gaussien' if stat.get('gaussien') else 'non gaussien'}) — "
            f"sondes en échec {100.0 * stat.get('frac_sondes_echec_iid', float('nan')):.0f} %, "
            f"CV RMS/bloc {stat.get('cv_rms_blocs', float('nan')):.3f}, "
            f"kurtosis {stat.get('kurtosis', float('nan')):.2f}")
        if stat.get('recommandation'):
            lignes.append(f"Recommandation     : {stat['recommandation']}")
    return lignes


def _ecrire_resume(run_dir, lignes_signal, csv_path):
    """Écrit RESUME_demo.txt : config démo + stats du signal + contrôles
    constatés par le calcul + pointeurs."""
    chemin = os.path.join(run_dir, "RESUME_demo.txt")
    cfg = [
        f"Loi d'ajustement   : {LOI_AJUSTEMENT}",
        f"SDOF               : Q={Q}  T_b={TB} s  amorçage={'oui' if SDOF_AMORCAGE else 'non'}",
        f"Fréquences f₀      : {F0_MIN}–{F0_MAX} Hz (pas {DELTA_F0} Hz)",
        f"Quantile cible     : P={PROBABILITE_CIBLE}  Cunnane={'oui' if OPTION_CUNNANE else 'non'}",
        f"SDF                : {'b=' + str(SDF_B) if SDF_ENABLED else 'désactivé'}",
        f"Projection         : {'T=%.2e s, α=%s, %s' % (DUREE_PROJECTION, ALFA_PROJECTION, METHODE_PROJECTION) if ENABLE_PROJECTION else 'désactivée'}",
        f"Classification     : N_CLUSTERS={N_CLUSTERS}  AUTO_SELECT_K={AUTO_SELECT_K}  "
        f"AUTO_K_SELON_STATIONNARITE={AUTO_K_SELON_STATIONNARITE}",
        f"Synthèse classes   : {SYNTHESE_CLASSES}",
        f"Features actives   : {[k for k, v in FEATURE_FLAGS.items() if v] or 'aucune'}",
        f"Stationnarité      : {'contrôle actif' if STATIONNARITE_ENABLED else 'contrôle désactivé'}"
        f"  (seuils : sondes {STATIONNARITE_FRAC_ECHEC}, CV RMS {STATIONNARITE_CV_RMS_MAX})",
        f"Contrat IID        : {'actif' if IID_GATE_ENABLED else 'désactivé'}"
        f"  (|ρ| ≤ {IID_RHO_MAX}, p-value ≥ {IID_PVALUE_MIN})",
        f"Multiprocess       : {USE_MULTIPROCESS}  (N_WORKERS={N_WORKERS})",
    ]
    controles = _lignes_controles(_lire_meta(run_dir))
    with open(chemin, "w", encoding="utf-8") as fh:
        fh.write("RESUME — mbd_demo_v1.py\n")
        fh.write(f"Date               : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Module principal   : {MODULE_PATH}\n")
        fh.write("\n[Signal généré]\n")
        fh.write("\n".join("  " + l for l in lignes_signal) + "\n")
        fh.write(f"  CSV                : {os.path.basename(csv_path)}\n")
        fh.write("\n[Configuration du calcul MBD]\n")
        fh.write("\n".join("  " + l for l in cfg) + "\n")
        if controles:
            fh.write("\n[Ce que le calcul a constaté]\n")
            fh.write("\n".join("  " + l for l in controles) + "\n")
            fh.write("  Le détail du contrat IID (test en cause, bandes, cause "
                     "probable, action) est\n  dans le bloc « Contrat IID » en "
                     "tête des deux rapports HTML.\n")
        fh.write("\n[Sorties du run] — sous-dossier courant\n")
        for f in sorted(os.listdir(run_dir)):
            fh.write(f"  {f}\n")
    return chemin


def _afficher_synthese(run_dir):
    """Affiche une synthèse courte lue dans le CSV SRE du run (best effort).
    Ne doit JAMAIS faire échouer la démo : toute la sortie est déjà écrite."""
    try:
        import pandas as pd
        sre_files = [f for f in glob.glob(os.path.join(run_dir, "SRE_*.csv"))
                     if "Projection" not in os.path.basename(f)]
        if not sre_files:
            return
        df = pd.read_csv(sre_files[0], sep=None, engine='python', dtype=str)

        def _peak(prefix):
            cols = [c for c in df.columns if c.startswith(prefix)]
            if not cols:
                return None
            # CSV français possible (décimale ','). Coercition robuste.
            s = pd.to_numeric(df[cols[0]].str.replace(',', '.', regex=False),
                              errors='coerce')
            return float(s.max()) if s.notna().any() else None

        print("\n  --- Synthèse ---")
        for ligne in _lignes_controles(_lire_meta(run_dir)):
            if not ligne.startswith("Recommandation"):
                print("  " + ligne)
        for label, prefix in (("SRC max  ", "SRC"),
                              ("SRE max  ", "SRE_Kappa4"),
                              ("SRX α_hi ", "SRX_alpha_high")):
            val = _peak(prefix)
            if val is not None:
                print(f"  {label}: {val:.4g} m/s²")
        # SRE projeté : lu dans le CSV de projection, s'il existe.
        proj_files = glob.glob(os.path.join(run_dir, "SRE_Projection_*.csv"))
        if proj_files:
            dp = pd.read_csv(proj_files[0], sep=None, engine='python', dtype=str)
            cols = [c for c in dp.columns if c.startswith("SRE_Projection_")]
            if cols:
                s = pd.to_numeric(dp[cols[0]].str.replace(',', '.', regex=False),
                                  errors='coerce')
                if s.notna().any():
                    print(f"  SRE proj.: {float(s.max()):.4g} m/s²")
        # Contrat IID : nombre de fréquences en échec (colonne IID_status).
        stat_cols = [c for c in df.columns if c.startswith("IID_status")]
        if stat_cols:
            statuts = df[stat_cols[0]]
            n_test = int((statuts != "n_a").sum())
            n_fail = int((statuts == "FAIL").sum())
            print(f"  Contrat IID : {n_fail} f₀ en échec sur {n_test} testées "
                  f"(explication dans les rapports HTML)")
        # Classes exclues de la synthèse (colonne présente si K > 1).
        if "Classes_exclues" in df.columns:
            n_excl = int((df["Classes_exclues"].fillna("") != "").sum())
            if n_excl:
                print(f"  ⚠ Synthèse des classes incomplète à {n_excl} f₀ : "
                      f"spectre sous-estimé (colonne Classes_exclues).")
    except Exception as exc:
        print(f"  (synthèse console indisponible : {exc})")


# =============================================================================
# Point d'entrée (PARENT uniquement — sous le garde __main__)
# =============================================================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Signal de démonstration
    t, y, lignes_signal = construire_signal()
    print(f"Signal généré ({MODE_SIGNAL}) : {len(y)} pts | fs={FS:.0f} Hz "
          f"| durée={len(y) / FS:.0f} s")
    for l in lignes_signal:
        print("  " + l)

    # 2. CSV unique par run (le nom devient aussi le préfixe du dossier de run)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"signal_demo_{MODE_SIGNAL}_{ts}.csv"
    csv_path = os.path.join(OUTPUT_FOLDER, csv_name)
    ecrire_csv(csv_path, t, y, skip_rows=10, delimiter=";")
    print(f"  CSV : {csv_path}")

    # 3. Brancher le module principal sur ce CSV (le reste de la config a déjà
    #    été appliqué au niveau module par _appliquer_config).
    m.CSV_FILEPATH  = csv_path
    m.CSV_SKIP_ROWS = 10
    m.CSV_DELIMITER = ";"
    m.OUTPUT_FOLDER = OUTPUT_FOLDER

    # 4. Avertissements de cohérence (non bloquants)
    _avertissements_sanity(FS, len(y))

    # 5. Lancement du pipeline MBD complet du module principal. main() crée
    #    lui-même un sous-dossier horodaté dédié et y écrit toute la sortie.
    print(f"Lancement du calcul MBD (loi={LOI_AJUSTEMENT}, "
          f"multiprocess={USE_MULTIPROCESS})…")
    avant = _sous_dossiers(OUTPUT_FOLDER)
    m.main(use_mp=USE_MULTIPROCESS)
    nouveaux = sorted(_sous_dossiers(OUTPUT_FOLDER) - avant)
    run_dir = os.path.join(OUTPUT_FOLDER, nouveaux[-1]) if nouveaux else OUTPUT_FOLDER

    # 6. Rendre le dossier de run auto-suffisant : y ranger le signal + un RESUME.
    if nouveaux:
        try:
            shutil.move(csv_path, os.path.join(run_dir, csv_name))
            csv_path = os.path.join(run_dir, csv_name)
        except Exception:
            pass
        _ecrire_resume(run_dir, lignes_signal, csv_path)

    # 7. Synthèse console
    _afficher_synthese(run_dir)
    print(f"\nTerminé. Résultats dans : {run_dir}/")


if __name__ == "__main__":
    main()
