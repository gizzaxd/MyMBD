# -*- coding: utf-8 -*-
"""Test `calculer_sdf_rainflow` — comptage de cycles + cumul de dommage.

Le test est conçu pour être LU sans connaître le code : on prend un signal
dont on sait compter les cycles à la main (un sinus parfait), et on montre
visuellement le comptage puis l'accumulation linéaire du dommage (Miner).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import _th

EXPL = {
    "fonction": "calculer_sdf_rainflow",
    "but": ("Estime la 'fatigue' subie par une pièce : elle découpe le "
            "signal en cycles (méthode rainflow ASTM E1049) et additionne "
            "le dommage de chacun (règle de Miner : D = Σ amplitude^b / C)."),
    "verifie": ("1) le comptage : un sinus de Ncyc oscillations d'amplitude "
                "X est bien vu comme Ncyc cycles d'amplitude X ; "
                "2) le cumul : le dommage s'accumule de façon LINÉAIRE et "
                "atteint exactement Ncyc·X^b/C."),
    "donnee": ("Un sinus pur : Ncyc oscillations identiques d'amplitude X. "
               "On SAIT compter à la main : Ncyc cycles, chacun d'étendue "
               "2X (donc amplitude X). La bonne réponse Ncyc·X^b/C est donc "
               "connue exactement, sans aucune approximation."),
    "lecture": ("Centre = le signal et un cycle annoté. Droite, panneau 1 = "
                "nb de cycles comptés (doit valoir Ncyc) ; panneau 2 = le "
                "dommage cumulé monte en ligne droite jusqu'à la valeur "
                "théorique ; panneau 3 = vérif sur 4 amplitudes × 4 pentes b."),
    "critere": ("Erreur relative sur le dommage total < 5 % pour toutes les "
                "combinaisons (amplitude, b)."),
}


def run(outdir):
    m = _th.charger_module()

    if not getattr(m, "HAS_RAINFLOW", False):
        fig, ax, _ = _th.figure_test(1, EXPL)
        ax.text(0.5, 0.5, "Numba indisponible — test rainflow ignoré",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        msg = "rainflow indisponible (numba absent) — test ignoré"
        _th.bandeau(fig, True, msg)
        png = _th.sauver(fig, outdir, "test_rainflow")
        return _th.resultat("calculer_sdf_rainflow", True, msg, png, EXPL)

    fs = 2000.0

    # --- 1) Comptage + cumul sur un cas pédagogique simple ------------------
    Xd, bd, Cd, Ncyc_d = 2.0, 4.0, 1.0, 24
    t_d = np.arange(0, Ncyc_d, 1.0 / fs)
    sig_d = Xd * np.sin(2 * np.pi * 1.0 * t_d)

    # Comptage : à b=1, C=1, D = Σ amplitude = (nb de cycles) × X.
    D_b1 = m.calculer_sdf_rainflow(Xd * np.sin(2 * np.pi * 1.0 * t_d), 1.0, 1.0)
    n_cycles_comptes = D_b1 / Xd

    # Cumul : dommage sur les j premiers cycles → doit valoir j · X^b / C.
    js = np.arange(0, Ncyc_d + 1, 2)
    D_cum, D_cum_th = [], []
    for j in js:
        if j == 0:
            D_cum.append(0.0)
        else:
            tj = np.arange(0, j, 1.0 / fs)
            D_cum.append(m.calculer_sdf_rainflow(
                Xd * np.sin(2 * np.pi * 1.0 * tj), bd, Cd))
        D_cum_th.append(j * (Xd ** bd) / Cd)

    # --- 2) Balayage de validation (amplitudes × pentes b) -----------------
    Ncyc = 200
    t = np.arange(0, Ncyc, 1.0 / fs)
    amplitudes = [0.5, 1.0, 2.0, 5.0]
    b_list = [3.0, 5.0, 8.0, 12.0]
    C = 1.0
    err_max = 0.0
    sweep = {}
    for b in b_list:
        obt, att = [], []
        for X in amplitudes:
            D = m.calculer_sdf_rainflow(X * np.sin(2 * np.pi * 1.0 * t), b, C)
            D_att = Ncyc * (X ** b) / C
            obt.append(D); att.append(D_att)
            err_max = max(err_max, abs(D - D_att) / D_att)
        sweep[b] = (obt, att)

    err_cnt = abs(n_cycles_comptes - Ncyc_d) / Ncyc_d
    passed = (err_max < 0.05) and (err_cnt < 0.02)

    # ------------------------------ Visuels --------------------------------
    fig, ax_data, (a1, a2, a3) = _th.figure_test(3, EXPL, largeur_verif=4.7)

    # Panneau donnée : le signal + un cycle annoté (étendue 2X, amplitude X)
    show = t_d < 3.0
    ax_data.plot(t_d[show], sig_d[show], "b-", lw=1.3)
    ax_data.axhline(Xd, color="g", ls=":", lw=1)
    ax_data.axhline(-Xd, color="g", ls=":", lw=1)
    ax_data.annotate("", xy=(0.75, Xd), xytext=(0.75, -Xd),
                     arrowprops=dict(arrowstyle="<->", color="r"))
    ax_data.text(0.80, 0.0, f"étendue 2X = {2*Xd:g}\namplitude X = {Xd:g}",
                 color="r", fontsize=8, va="center")
    ax_data.set_title("Donnée d'inférence : sinus pur")
    ax_data.set_xlabel("temps (s)"); ax_data.set_ylabel("signal")
    ax_data.grid(alpha=0.3)
    _th.legende_data(ax_data, f"{Ncyc_d} oscillations identiques d'amplitude "
                              f"X={Xd:g}. On compte Ncyc={Ncyc_d} cycles\n"
                              f"'à la main' → réponse exacte connue.")

    # Panneau 1 : comptage de cycles
    a1.bar(["compté", "attendu"], [n_cycles_comptes, Ncyc_d],
           color=["#1e88e5", "#43a047"], width=0.5)
    a1.set_title("Comptage rainflow\n(nb de cycles, via D à b=1)")
    a1.set_ylabel("nombre de cycles")
    a1.text(0, n_cycles_comptes, f"{n_cycles_comptes:.2f}",
            ha="center", va="bottom", fontsize=9)
    a1.grid(alpha=0.3, axis="y")

    # Panneau 2 : cumul de dommage (Miner) — accumulation linéaire
    a2.plot(js, D_cum_th, "k--", lw=1.5, label="théorie  j·X^b/C")
    a2.plot(js, D_cum, "ro", ms=6, label="rainflow (cumul)")
    a2.set_title("Cumul du dommage (règle de Miner)")
    a2.set_xlabel("nb de cycles inclus")
    a2.set_ylabel("dommage cumulé D")
    a2.grid(alpha=0.3); a2.legend(fontsize=8)

    # Panneau 3 : balayage amplitudes × b
    for b in b_list:
        obt, att = sweep[b]
        a3.loglog(amplitudes, att, "k--")
        a3.loglog(amplitudes, obt, "o", label=f"b={b:.0f}")
    a3.set_title("Validation : D vs Ncyc·X^b/C")
    a3.set_xlabel("amplitude X"); a3.set_ylabel("dommage")
    a3.grid(alpha=0.3, which="both"); a3.legend(fontsize=8)

    msg = (f"cycles comptés = {n_cycles_comptes:.2f}/{Ncyc_d} "
           f"(err {err_cnt*100:.2f}%) ; dommage err. rel. max "
           f"= {err_max*100:.2f}% (seuil 5%)")
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_rainflow")
    return _th.resultat("calculer_sdf_rainflow", passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
