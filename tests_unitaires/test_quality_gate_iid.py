# -*- coding: utf-8 -*-
"""Test `quality_gate_iid` / `_iid_verdict` — contrôle d'indépendance des
blocs (Quality Gate IID, cahier des charges §3-5)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import _th

EXPL = {
    "fonction": "quality_gate_iid / _iid_verdict",
    "but": ("Avant d'ajuster les lois, vérifie que les valeurs extrêmes "
            "des blocs sont bien INDÉPENDANTES entre elles (hypothèse "
            "requise). Sinon le résultat MBD est biaisé."),
    "verifie": ("Que le module détecte correctement : des colonnes "
                "indépendantes (à laisser passer), une tendance et des "
                "séries corrélées (à rejeter) ; et que le verdict global "
                "GO / WARNING / NO-GO suit bien la fraction de fréquences "
                "en échec."),
    "donnee": ("Une matrice [400 blocs × colonnes] mélangeant : 60 "
               "colonnes INDÉPENDANTES (réponse connue : à valider), 1 "
               "colonne en tendance pure et 12 colonnes fortement "
               "corrélées AR(1) φ=0.85 (réponse connue : à rejeter)."),
    "lecture": ("Centre : un exemple de série indépendante (verte) vs "
                "corrélée (rouge). Droite : ρ et p-value par colonne — les "
                "vertes restent dans les seuils, les rouges les franchissent."),
    "critere": ("Faux positifs IID ≤ 15 %, tendance + AR(1) tous détectés, "
                "API 1D/2D correcte, verdict GO/WARNING/NO-GO exact."),
}


def _ar1(rng, n, phi, sigma=1.0):
    e = np.empty(n)
    e[0] = rng.standard_normal()
    w = rng.standard_normal(n) * sigma
    for t in range(1, n):
        e[t] = phi * e[t - 1] + w[t]
    return e


def run(outdir):
    m = _th.charger_module()
    rng = np.random.default_rng(2026)
    N, M_iid, M_ar = 400, 60, 12

    iid_block = rng.standard_normal((N, M_iid))
    trend_col = np.arange(N, dtype=float)[:, None]
    ar_block = np.column_stack([_ar1(rng, N, 0.85) for _ in range(M_ar)])
    X = np.hstack([iid_block, trend_col, ar_block])

    d = m.quality_gate_iid(X)
    rho, pval, fail = d['rho'], d['pvalue'], d['fail']
    sl_iid = slice(0, M_iid)
    i_trend = M_iid
    sl_ar = slice(M_iid + 1, M_iid + 1 + M_ar)

    frac_fp = float(np.mean(fail[sl_iid]))
    ok_iid = frac_fp <= 0.15
    ok_trend = (bool(fail[i_trend]) and rho[i_trend] > 0.999
                and pval[i_trend] < 1e-6)
    ok_ar = bool(np.all(fail[sl_ar])) and bool(np.all(rho[sl_ar] > 0.2))
    s = m.quality_gate_iid(trend_col.ravel())
    ok_api = (isinstance(s['rho'], float) and isinstance(s['fail'], bool)
              and bool(s['fail']) and rho.shape == (X.shape[1],))
    ff, fn = m.IID_FAIL_FRAC_MAX, m.IID_NOGO_FRAC
    ok_verdict = (m._iid_verdict(0.0) == 'GO'
                  and m._iid_verdict(ff) == 'GO'
                  and m._iid_verdict(0.5 * (ff + fn)) == 'WARNING'
                  and m._iid_verdict(min(1.0, fn + 0.05)) == 'NO-GO')
    passed = ok_iid and ok_trend and ok_ar and ok_api and ok_verdict

    fig, ax_data, (a1, a2) = _th.figure_test(2, EXPL, largeur_verif=5.0)

    ax_data.plot(iid_block[:, 0], color="#2e7d32", lw=0.9,
                 label="colonne indépendante (OK)")
    ax_data.plot(ar_block[:, 0], color="#c62828", lw=0.9,
                 label="colonne corrélée AR(1) (à rejeter)")
    ax_data.set_title("Donnée d'inférence : 2 colonnes types")
    ax_data.set_xlabel("indice de bloc"); ax_data.set_ylabel("valeur")
    ax_data.legend(fontsize=8); ax_data.grid(alpha=0.3)
    _th.legende_data(ax_data, "Matrice 400×73 : 60 indépendantes + 1 "
                              "tendance\n+ 12 AR(1) φ=0.85 (réponses "
                              "connues d'avance).")

    idx = np.arange(X.shape[1])
    cats = (["IID"] * M_iid) + ["dep"] + ["dep"] * M_ar
    cvec = ["#2e7d32" if c == "IID" else "#c62828" for c in cats]

    a1.scatter(idx, rho, c=cvec, s=26)
    a1.axhline(m.IID_RHO_MAX, color="k", ls="--", lw=1)
    a1.axhline(-m.IID_RHO_MAX, color="k", ls="--", lw=1,
               label=f"seuil |ρ|={m.IID_RHO_MAX}")
    a1.set_title("Autocorrélation lag-1 par colonne")
    a1.set_xlabel("colonne (f0)"); a1.set_ylabel("ρ (rangs)")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)

    a2.scatter(idx, np.clip(pval, 1e-12, 1.0), c=cvec, s=26)
    a2.axhline(m.IID_PVALUE_MIN, color="k", ls="--", lw=1,
               label=f"seuil p={m.IID_PVALUE_MIN}")
    a2.set_yscale("log")
    a2.set_title("p-value test des suites")
    a2.set_xlabel("colonne (f0)"); a2.set_ylabel("p-value (log)")
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=8)

    msg = (f"IID faux-positifs={frac_fp*100:.1f}% (≤15%) ; "
           f"tendance={ok_trend} ; AR(1) tous={ok_ar} ; "
           f"API={ok_api} ; verdict={ok_verdict}")
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_quality_gate_iid")
    return _th.resultat("quality_gate_iid / _iid_verdict",
                        passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
