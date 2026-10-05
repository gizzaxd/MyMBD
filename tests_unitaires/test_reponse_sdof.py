# -*- coding: utf-8 -*-
"""Test `reponse_sdof` — réponse d'un oscillateur 1 degré de liberté."""
import os
import numpy as np
import matplotlib.pyplot as plt
import _th

EXPL = {
    "fonction": "reponse_sdof",
    "but": ("Simule comment une pièce (modélisée par un oscillateur de "
            "fréquence propre f0 et de facteur de qualité Q) répond à une "
            "vibration imposée. Brique de base de tous les spectres MBD."),
    "verifie": ("Le résultat classique de mécanique vibratoire : excité "
                "exactement à sa résonance f0, l'oscillateur amplifie le "
                "mouvement d'un facteur Q (résultat exact, sans convention "
                "discutable)."),
    "donnee": ("Une vibration sinusoïdale d'amplitude connue A, à la "
               "fréquence propre f0. La physique dit que l'amplification "
               "doit valoir Q : c'est notre étalon, balayé sur 6 f0 × 4 Q."),
    "lecture": ("Centre : la vibration imposée et la réponse amplifiée. "
                "Droite : l'amplification mesurée doit coïncider avec les "
                "lignes Q attendues, quelle que soit f0."),
    "critere": "Erreur sur le facteur d'amplification < 12 %.",
}


def run(outdir):
    m = _th.charger_module()
    fs = 8000.0
    A = 5.0
    f0_list = np.array([10, 25, 50, 100, 200, 400], dtype=float)
    Q_list = np.array([5, 10, 20, 50], dtype=float)

    fig, ax_data, (ax_v,) = _th.figure_test(1, EXPL, largeur_verif=5.6)

    err_max = 0.0
    for Q in Q_list:
        ratios = []
        for f0 in f0_list:
            dur = max(2.0, 60.0 / f0)
            t = np.arange(0, dur, 1.0 / fs)
            exc = A * np.sin(2 * np.pi * f0 * t)
            z = m.reponse_sdof(exc, f0=f0, Q=Q, fs=fs)
            pseudo = (2 * np.pi * f0) ** 2 * z
            n0 = int(0.3 * len(z))
            ratio = np.max(np.abs(pseudo[n0:])) / A
            ratios.append(ratio)
            err_max = max(err_max, abs(ratio - Q) / Q)
            if Q == 10 and f0 == 50:                  # cas illustratif
                show = t < 0.6
                ax_data.plot(t[show], exc[show], "b-", lw=1,
                             label="vibration imposée (A=5)")
                ax_data.plot(t[show], pseudo[show], "r-", lw=1,
                             label="réponse (≈ Q×A à résonance)")
        ax_v.plot(f0_list, ratios, "o-", label=f"mesuré Q={Q:.0f}")
        ax_v.hlines(Q, f0_list[0], f0_list[-1], colors="gray",
                    linestyles="--", linewidth=1)

    ax_data.set_title("Donnée d'inférence : f0=50 Hz, Q=10")
    ax_data.set_xlabel("temps (s)"); ax_data.set_ylabel("accélération")
    ax_data.legend(fontsize=8); ax_data.grid(alpha=0.3)
    _th.legende_data(ax_data, "Sinus à la fréquence propre f0. La physique\n"
                              "impose une amplification = Q (étalon exact).")

    ax_v.set_xlabel("f0 (Hz)")
    ax_v.set_ylabel("amplification mesurée  (≈ Q attendu, lignes --)")
    ax_v.set_title("Amplification à résonance vs Q attendu")
    ax_v.legend(fontsize=8); ax_v.grid(alpha=0.3)

    passed = err_max < 0.12
    msg = f"erreur max sur l'amplification = {err_max*100:.1f}% (seuil 12%)"
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_reponse_sdof")
    return _th.resultat("reponse_sdof", passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
