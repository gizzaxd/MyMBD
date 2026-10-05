# -*- coding: utf-8 -*-
"""Helpers communs aux tests unitaires visuels MBD.

Objectif PÉDAGOGIQUE : chaque test doit être compréhensible SANS lire le code.
Chaque figure contient donc, de gauche à droite :

    1. un PANNEAU TEXTE expliquant en français simple : ce qu'on vérifie,
       comment la donnée de test est fabriquée (et pourquoi elle est fiable),
       et le critère de réussite ;
    2. un PANNEAU « DONNÉE D'INFÉRENCE » montrant la donnée réellement
       fournie à la fonction testée ;
    3. un ou plusieurs PANNEAUX DE VÉRIFICATION (fonction vs référence).

`run_all.py` agrège le tout dans un index HTML qui reprend la même explication
en prose pour un lecteur non technique.
"""

import os
import sys
import tempfile
import textwrap
import importlib.util

# Cache Numba NEUF par exécution : le module principal est chargé via importlib
# sous un nom non importable ; un cache Numba périmé tente alors de réimporter
# un module '<dynamic>' et lève ModuleNotFoundError (cf. test_rainflow). Un
# répertoire de cache vide force une recompilation propre, sans dépickling.
os.environ.setdefault("NUMBA_CACHE_DIR",
                       tempfile.mkdtemp(prefix="numba_cache_mbd_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

_MODULE_FILE = "mbd_simple-multi-process_v3_4.py"


def racine_projet():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def charger_module():
    """Charge mbd_simple-multi-process_vx_x.py (nom à tirets)."""
    if "mbd_main" in sys.modules:
        return sys.modules["mbd_main"]
    path = os.path.join(racine_projet(), _MODULE_FILE)
    spec = importlib.util.spec_from_file_location("mbd_main", path)
    mod = importlib.util.module_from_spec(spec)
    # Enregistré AVANT exec pour que Numba puisse réimporter le module
    # définissant les fonctions @njit (sinon nom non résoluble).
    sys.modules["mbd_main"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
#  Bloc explicatif (prose pour non-spécialiste)                                #
# --------------------------------------------------------------------------- #
#
# Une « explication » est un dict :
#     {
#       "fonction":  "nom de la/des fonction(s) testée(s)",
#       "but":       "ce que la fonction calcule, en une phrase simple",
#       "verifie":   "ce que le test prouve concrètement",
#       "donnee":    "comment la donnée de test est fabriquée et pourquoi on "
#                    "connaît la bonne réponse à l'avance",
#       "critere":   "à quelle condition le test est déclaré PASS",
#       "lecture":   "comment lire les graphiques (PASS attendu = ...)",
#     }


_SECTIONS = [
    ("À quoi sert la fonction", "but"),
    ("Ce que ce test prouve", "verifie"),
    ("Donnée de test (et pourquoi elle est fiable)", "donnee"),
    ("Comment lire les graphiques", "lecture"),
    ("Critère de réussite", "critere"),
]


def _ecrire_explication(ax, expl, largeur_car=46):
    ax.axis("off")
    y = 0.99
    ax.text(0.0, y, expl.get("fonction", ""), transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", color="#1a237e")
    y -= 0.055
    for titre, cle in _SECTIONS:
        txt = expl.get(cle)
        if not txt:
            continue
        ax.text(0.0, y, titre, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", va="top", color="#37474f")
        y -= 0.038
        for ligne in textwrap.wrap(txt, largeur_car):
            ax.text(0.0, y, ligne, transform=ax.transAxes, fontsize=9,
                    va="top", color="#212121")
            y -= 0.034
        y -= 0.022


def figure_test(n_verif, explication, largeur_verif=4.9, hauteur=5.0,
                largeur_texte=4.7, largeur_data=4.9):
    """Construit la figure pédagogique standard.

    Retourne (fig, ax_data, axes_verif) :
      - ax_data : panneau où le test DOIT tracer la donnée d'inférence ;
      - axes_verif : liste de n_verif panneaux de vérification.
    """
    ncol = 2 + n_verif
    width = largeur_texte + largeur_data + n_verif * largeur_verif
    fig = plt.figure(figsize=(width, hauteur))
    gs = gridspec.GridSpec(
        1, ncol, figure=fig,
        width_ratios=[largeur_texte, largeur_data] + [largeur_verif] * n_verif)
    ax_txt = fig.add_subplot(gs[0, 0])
    _ecrire_explication(ax_txt, explication)
    ax_data = fig.add_subplot(gs[0, 1])
    axes_v = [fig.add_subplot(gs[0, 2 + i]) for i in range(n_verif)]
    return fig, ax_data, axes_v


def legende_data(ax, texte):
    """Légende encadrée 'comment cette donnée est produite' sous un panneau."""
    ax.text(0.5, -0.22, texte, transform=ax.transAxes, fontsize=8.2,
            ha="center", va="top", style="italic", color="#37474f",
            bbox=dict(facecolor="#eceff1", edgecolor="#b0bec5",
                      boxstyle="round,pad=0.4"))


def bandeau(fig, passed, msg):
    """Bandeau coloré PASS/FAIL en haut de la figure."""
    color = "#2e7d32" if passed else "#c62828"
    txt = ("PASS — " if passed else "FAIL — ") + msg
    fig.suptitle(txt, color="white", fontsize=13, fontweight="bold",
                 bbox=dict(facecolor=color, edgecolor="none",
                           boxstyle="round,pad=0.4"), y=0.995)


def sauver(fig, outdir, nom):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, nom + ".png")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def resultat(nom, passed, msg, png, explication=None):
    return {"nom": nom, "passed": bool(passed), "msg": msg, "png": png,
            "explication": explication or {}}
