# -*- coding: utf-8 -*-
"""Test `calculer_sre_analytique` — SRE/SRX analytiques depuis la DSP."""
import os
import numpy as np
import matplotlib.pyplot as plt
import _th

EXPL = {
    "fonction": "calculer_sre_analytique",
    "but": ("Calcule, directement depuis la densité spectrale du signal, "
            "le SRE (extrême attendu) et le SRX (extrême à risque de "
            "dépassement) — la branche analytique de comparaison du MBD."),
    "verifie": ("Une INVARIANCE exacte, indépendante des conventions : si "
                "on double l'amplitude du signal d'entrée, SRE et SRX "
                "doivent exactement doubler (linéarité). Plus : résultats "
                "finis, positifs, et SRX(α faible) ≥ SRE."),
    "donnee": ("Un bruit gaussien (signal d'excitation réaliste) et le "
               "MÊME signal multiplié par 2. Le rapport attendu est connu "
               "exactement = 2, sur toute la grille de fréquences f0."),
    "lecture": ("Panneau 1 : les spectres SRE/SRX (×1 et ×2). Panneau 2 : "
                "le rapport ×2/×1 doit être plat à 2.0 partout."),
    "critere": ("Écart de linéarité < 1e-3, spectres positifs/finis, "
                "SRX(α=0.01) ≥ SRE."),
}


def run(outdir):
    m = _th.charger_module()
    rng = np.random.default_rng(5)
    fs = 6000.0
    t = np.arange(0, 30.0, 1.0 / fs)
    sig = rng.standard_normal(len(t)) * 4.0
    f0_grid = np.array([20, 40, 80, 160, 320, 600], dtype=float)
    Q = 10.0

    r1 = m.calculer_sre_analytique(sig, fs, Q, f0_grid,
                                   alpha_srx_low=0.01, alpha_srx_high=0.99)
    r2 = m.calculer_sre_analytique(2.0 * sig, fs, Q, f0_grid,
                                   alpha_srx_low=0.01, alpha_srx_high=0.99)
    sre1, xl1 = np.array(r1[0]), np.array(r1[1])
    sre2 = np.array(r2[0])

    ratio = sre2 / np.maximum(sre1, 1e-12)
    err_scale = np.max(np.abs(ratio - 2.0))
    fini = np.all(np.isfinite(sre1)) and np.all(sre1 > 0)
    ordre = np.all(xl1 >= sre1 * 0.999)

    fig, ax_data, (a1, a2) = _th.figure_test(2, EXPL, largeur_verif=5.0)

    show = t < 0.3
    ax_data.plot(t[show], sig[show], lw=0.8, label="signal ×1")
    ax_data.plot(t[show], 2.0 * sig[show], lw=0.8, alpha=0.7,
                 label="signal ×2")
    ax_data.set_title("Donnée d'inférence : bruit gaussien")
    ax_data.set_xlabel("temps (s)"); ax_data.set_ylabel("accélération")
    ax_data.legend(fontsize=8); ax_data.grid(alpha=0.3)
    _th.legende_data(ax_data, "Bruit gaussien et son double exact.\n"
                              "Réponse attendue : tout doit doubler.")

    a1.plot(f0_grid, sre1, "o-", label="SRE (×1)")
    a1.plot(f0_grid, sre2, "s-", label="SRE (×2)")
    a1.plot(f0_grid, xl1, "^--", label="SRX α=0.01 (×1)")
    a1.set_xlabel("f0 (Hz)"); a1.set_ylabel("amplitude")
    a1.set_title("Spectres SRE / SRX")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)

    a2.plot(f0_grid, ratio, "o-")
    a2.axhline(2.0, color="k", ls="--", label="attendu = 2")
    a2.set_title("Rapport SRE(×2)/SRE(×1)")
    a2.set_xlabel("f0 (Hz)"); a2.grid(alpha=0.3); a2.legend(fontsize=8)

    passed = (err_scale < 1e-3) and fini and ordre
    msg = (f"écart linéarité max = {err_scale:.2e} (seuil 1e-3) ; "
           f"positif={fini} ; SRX_low≥SRE={ordre}")
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_sre_analytique")
    return _th.resultat("calculer_sre_analytique", passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
