# -*- coding: utf-8 -*-
"""Test `calculer_projection_lmoments` — extrapolation TVE de l'extrême."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kappa4 as sp_k4
import _th

EXPL = {
    "fonction": "calculer_projection_lmoments",
    "but": ("Extrapole l'extrême attendu sur une durée d'usage T_proj "
            "(ex. toute la vie d'un véhicule) à partir d'une loi ajustée "
            "sur un essai court — c'est le cœur de la méthode MBD."),
    "verifie": ("Que la projection respecte EXACTEMENT son identité "
                "mathématique : SRE_proj = quantile(loi, α^(1/M)) avec "
                "M = (n_classe/total)·T_proj/Tb, et qu'elle croît bien "
                "avec la durée projetée."),
    "donnee": ("On ajuste une Kappa4 sur 30 000 tirages de paramètres "
               "connus, puis on recalcule M et le quantile de façon "
               "INDÉPENDANTE de la fonction : c'est la référence."),
    "lecture": ("À droite : les croix (fonction) doivent tomber pile sur "
                "la courbe de référence, et monter quand T_proj augmente "
                "(plus on roule longtemps, plus l'extrême est grand)."),
    "critere": ("Écart à l'identité < 1e-9 ET SRE projeté strictement "
                "croissant avec T_proj."),
}


def run(outdir):
    m = _th.charger_module()
    rng = np.random.default_rng(9)
    data = sp_k4(h=0.1, k=0.05, loc=20.0, scale=4.0).rvs(30000,
                                                          random_state=rng)
    params = m.ajuster_kappa4(data)
    assert params.get("success"), "fit Kappa4 de référence échoué"

    Tb = 1.0
    n_classe = total = 500
    alfa = 0.90
    T_projs = np.array([6e2, 6e3, 6e4, 6e6, 6e8])

    sre_obt, sre_att, errs = [], [], []
    for Tp in T_projs:
        sre, M = m.calculer_projection_lmoments(params, data, n_classe,
                                                total, Tb, Tp, alfa)
        M_att = (n_classe / total) * Tp / Tb
        sre_ref = m.loi_ppf(params, alfa ** (1.0 / M_att))
        sre_obt.append(sre); sre_att.append(sre_ref)
        e = max(abs(sre - sre_ref) / max(abs(sre_ref), 1e-9),
                abs(M - M_att) / M_att)
        errs.append(e)

    fig, ax_data, (ax_v,) = _th.figure_test(1, EXPL, largeur_verif=6.0)

    ax_data.hist(data, bins=70, density=True, color="#80cbc4", alpha=0.85)
    ax_data.set_title("Donnée d'inférence : échantillon de blocs")
    ax_data.set_xlabel("extrême par bloc"); ax_data.set_ylabel("densité")
    ax_data.grid(alpha=0.3)
    _th.legende_data(ax_data, "30 000 tirages Kappa4 (params connus) → on "
                              "ajuste la loi,\npuis on extrapole à très "
                              "longue durée.")

    ax_v.semilogx(T_projs, sre_att, "k--o", lw=1.5,
                  label="référence indépendante  quantile(α^(1/M))")
    ax_v.semilogx(T_projs, sre_obt, "rx", ms=12, mew=2, label="fonction")
    ax_v.set_xlabel("T_proj (s) — durée d'usage projetée")
    ax_v.set_ylabel("SRE projeté (extrême attendu)")
    ax_v.set_title("Extrapolation TVE : extrême vs durée projetée")
    ax_v.grid(alpha=0.3, which="both"); ax_v.legend(fontsize=9)

    err_max = max(errs)
    croissant = all(x <= y + 1e-9 for x, y in zip(sre_obt, sre_obt[1:]))
    passed = (err_max < 1e-9) and croissant
    msg = (f"écart max vs identité = {err_max:.2e} (seuil 1e-9) ; "
           f"monotone croissant={croissant}")
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_projection")
    return _th.resultat("calculer_projection_lmoments", passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
