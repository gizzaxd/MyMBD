# -*- coding: utf-8 -*-
"""Test `calculer_lmoments` — L-moments (résumés robustes d'une distribution)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import _th

REF = {
    "Uniforme(0,1)":    dict(L1=0.5, L2=1.0 / 6.0, t3=0.0, t4=0.0),
    "Exponentielle(1)": dict(L1=1.0, L2=0.5, t3=1.0 / 3.0, t4=1.0 / 6.0),
    "Normale(0,1)":     dict(L1=0.0, L2=1.0 / np.sqrt(np.pi), t3=0.0,
                             t4=0.1226017),
}

EXPL = {
    "fonction": "calculer_lmoments",
    "but": ("Calcule les L-moments (L2 = dispersion, τ3 = asymétrie, "
            "τ4 = aplatissement). Ce sont des résumés de forme plus "
            "robustes que les moments classiques ; ils alimentent "
            "l'ajustement Kappa4."),
    "verifie": ("Que les L-moments calculés convergent vers les valeurs "
                "THÉORIQUES EXACTES de trois lois connues quand on augmente "
                "le nombre d'échantillons."),
    "donnee": ("Tirages de trois lois aux L-moments connus analytiquement : "
               "Uniforme (τ3=0, τ4=0), Exponentielle (τ3=1/3, τ4=1/6), "
               "Normale (τ4≈0.1226). Ces valeurs exactes sont l'étalon."),
    "lecture": ("À droite, un panneau par loi : les points (L2, τ3, τ4) "
                "doivent rejoindre les lignes pointillées (valeurs vraies) "
                "quand n grandit. PASS = écart final négligeable."),
    "critere": "Écart max aux valeurs théoriques à n=200 000 < 0.02.",
}


def run(outdir):
    m = _th.charger_module()
    rng = np.random.default_rng(7)
    ns = [50, 200, 1000, 5000, 20000, 200000]

    fig, ax_data, axes = _th.figure_test(3, EXPL, largeur_verif=4.4)

    # Panneau donnée : histogrammes des trois lois de référence (n modéré)
    ax_data.hist(rng.random(4000), bins=40, density=True, alpha=0.6,
                 label="Uniforme")
    ax_data.hist(rng.exponential(1.0, 4000), bins=40, density=True,
                 alpha=0.6, label="Exponentielle")
    ax_data.hist(rng.standard_normal(4000), bins=40, density=True,
                 alpha=0.6, label="Normale")
    ax_data.set_title("Donnée d'inférence : 3 lois étalons")
    ax_data.set_xlabel("valeur"); ax_data.set_ylabel("densité")
    ax_data.legend(fontsize=8); ax_data.grid(alpha=0.3)
    _th.legende_data(ax_data, "Tirages de lois dont les L-moments sont\n"
                              "connus EXACTEMENT par le calcul → étalon.")

    err_max = 0.0
    for ax, (nom, ref) in zip(axes, REF.items()):
        l2s, t3s, t4s = [], [], []
        for n in ns:
            if nom.startswith("Uniforme"):
                d = rng.random(n)
            elif nom.startswith("Expo"):
                d = rng.exponential(1.0, n)
            else:
                d = rng.standard_normal(n)
            _, l2, t3, t4 = m.calculer_lmoments(d)
            l2s.append(l2); t3s.append(t3); t4s.append(t4)
        ax.axhline(ref["L2"], color="C0", ls="--", lw=1)
        ax.axhline(ref["t3"], color="C1", ls="--", lw=1)
        ax.axhline(ref["t4"], color="C2", ls="--", lw=1)
        ax.semilogx(ns, l2s, "C0o-", label="L2")
        ax.semilogx(ns, t3s, "C1s-", label="τ3")
        ax.semilogx(ns, t4s, "C2^-", label="τ4")
        ax.set_title(f"{nom}\n(-- = valeurs vraies)")
        ax.set_xlabel("n (échantillons)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        err_max = max(err_max,
                      abs(l2s[-1] - ref["L2"]),
                      abs(t3s[-1] - ref["t3"]),
                      abs(t4s[-1] - ref["t4"]))

    passed = err_max < 0.02
    msg = f"erreur max @ n=200000 (vs théorie) = {err_max:.4f} (seuil 0.02)"
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_lmoments")
    return _th.resultat("calculer_lmoments", passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
