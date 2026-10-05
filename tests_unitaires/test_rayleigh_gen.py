# -*- coding: utf-8 -*-
"""Test `ajuster_rayleigh_gen` / `rayleigh_gen_ppf` — loi de Rayleigh
généralisée (Kundu & Raqab), alternative à Kappa4 sur signaux gaussiens."""
import os
import numpy as np
import matplotlib.pyplot as plt
import _th

EXPL = {
    "fonction": "ajuster_rayleigh_gen / rayleigh_gen_ppf",
    "but": ("Ajuste la loi de Rayleigh généralisée, recommandée pour les "
            "extrêmes de réponses à vibrations gaussiennes (souvent plus "
            "stable que Kappa4 dans ce cas)."),
    "verifie": ("Que les deux paramètres (forme α, échelle λ) imposés sont "
                "bien retrouvés, et que les quantiles ajustés collent à la "
                "formule analytique exacte."),
    "donnee": ("On fabrique l'échantillon par INVERSION EXACTE de la loi : "
               "x = √(−ln(1−u^(1/α)))/λ avec u uniforme. Les (α,λ) sont "
               "donc connus exactement. Balayage α∈{0.5,1,2,5,10} (α=1 = "
               "Rayleigh pure), λ∈{0.2,1,5}."),
    "lecture": ("Droite : α ajusté vs α vrai pour chaque λ — les points "
                "doivent suivre la diagonale. Centre : l'échantillon et la "
                "loi vraie superposée."),
    "critere": ("Tous les ajustements réussis, erreur paramètres < 10 % et "
                "erreur quantiles < 6 %."),
}


def ppf_exacte(p, a, lam):
    return np.sqrt(-np.log(1.0 - p ** (1.0 / a))) / lam


def run(outdir):
    m = _th.charger_module()
    rng = np.random.default_rng(11)
    alphas = [0.5, 1.0, 2.0, 5.0, 10.0]
    lambdas = [0.2, 1.0, 5.0]
    probs = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])

    fig, ax_data, (ax_v,) = _th.figure_test(1, EXPL, largeur_verif=5.6)

    err_param_max = err_ppf_max = 0.0
    n_ok = n_tot = 0
    for lam in lambdas:
        a_obt = []
        for a in alphas:
            u = rng.random(40000)
            x = ppf_exacte(u, a, lam)
            p = m.ajuster_rayleigh_gen(x)
            n_tot += 1
            if p.get("success"):
                n_ok += 1
                a_hat, l_hat = p["rg_alpha"], p["rg_lambda"]
                a_obt.append(a_hat)
                err_param_max = max(err_param_max, abs(a_hat - a) / a,
                                    abs(l_hat - lam) / lam)
                ppf_fit = np.array([m.rayleigh_gen_ppf(p, pr) for pr in probs])
                ppf_th = ppf_exacte(probs, a, lam)
                err_ppf_max = max(err_ppf_max,
                                  np.max(np.abs(ppf_fit - ppf_th)
                                         / np.maximum(ppf_th, 1e-9)))
                if abs(lam - 1.0) < 1e-9 and abs(a - 2.0) < 1e-9:
                    ax_data.hist(x, bins=70, density=True, color="#b39ddb",
                                 alpha=0.85, label="échantillon")
                    xs = np.linspace(x.min(), np.percentile(x, 99.5), 300)
                    pdf = (2 * a * lam**2 * xs
                           * np.exp(-(lam*xs)**2)
                           * (1 - np.exp(-(lam*xs)**2))**(a-1))
                    ax_data.plot(xs, pdf, "r-", lw=2, label="loi vraie")
            else:
                a_obt.append(np.nan)
        ax_v.plot(alphas, a_obt, "o-", label=f"λ = {lam}")

    ax_v.plot(alphas, alphas, "k--", label="α ajusté = α vrai (idéal)")
    ax_v.set_xlabel("α vrai (imposé)"); ax_v.set_ylabel("α ajusté")
    ax_v.set_title("Forme α retrouvée, pour chaque échelle λ")
    ax_v.grid(alpha=0.3); ax_v.legend(fontsize=8)

    ax_data.set_title("Donnée d'inférence : α=2, λ=1")
    ax_data.set_xlabel("valeur"); ax_data.set_ylabel("densité")
    ax_data.legend(fontsize=8); ax_data.grid(alpha=0.3)
    _th.legende_data(ax_data, "Échantillon par inversion EXACTE de la CDF\n"
                              "→ (α,λ) parfaitement connus.")

    passed = (n_ok == n_tot) and err_param_max < 0.10 and err_ppf_max < 0.06
    msg = (f"{n_ok}/{n_tot} fits OK ; err. param max={err_param_max*100:.1f}% "
           f"(seuil 10%) ; err. PPF max={err_ppf_max*100:.1f}% (seuil 6%)")
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_rayleigh_gen")
    return _th.resultat("ajuster_rayleigh_gen / rayleigh_gen_ppf",
                        passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
