# -*- coding: utf-8 -*-
"""Test `ajuster_kappa4` / `kappa4_ppf` — ajustement de la loi Kappa4.

Couverture renforcée des points DÉGÉNÉRÉS de la ligne k=0 (logistique h=-1,
Gumbel h=0, exponentielle h=1) traités par le développement limité k→0.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import _th


def kappa4_q(F, xi, alpha, k, h):
    """Fonction quantile Kappa4 EXACTE (Hosking), avec limites h→0 et k→0.

    Référence INDÉPENDANTE du module testé — et de scipy.stats.kappa4 qui
    calcule np.log(h) et renvoie NaN pour la branche k=0, h≤0 (logistique,
    Gumbel). Convention identique à scipy : x_std(k=0) = −ln((1−F^h)/h).
    """
    F = np.asarray(F, dtype=float)
    if abs(h) < 1e-12:
        inner = -np.log(F)                       # limite h→0
    else:
        inner = (1.0 - F ** h) / h
    if abs(k) < 1e-12:
        return xi - alpha * np.log(inner)        # limite k→0
    return xi + (alpha / k) * (1.0 - inner ** k)


def kappa4_pdf(x, xi, alpha, k, h, F):
    """Densité approchée par différences finies de la quantile (panneau visuel)."""
    dF = np.gradient(F)
    dx = np.gradient(x)
    return dF / np.maximum(dx, 1e-12)

CAS = [
    dict(h=0.0, k=0.0,  nom="Gumbel (k=0,h=0)"),
    dict(h=-1.0, k=0.0, nom="Logistique (k=0,h=-1)"),
    dict(h=1.0, k=0.0,  nom="Exponentielle (k=0,h=1)"),
    dict(h=-0.5, k=1e-3, nom="Transition k≈0"),
    dict(h=0.2, k=0.1,  nom="h=0.2, k=0.1"),
    dict(h=-0.3, k=0.2, nom="h=-0.3, k=0.2"),
    dict(h=0.5, k=-0.2, nom="h=0.5, k=-0.2"),
]

EXPL = {
    "fonction": "ajuster_kappa4 / kappa4_ppf",
    "but": ("La loi Kappa4 (4 paramètres) sert à modéliser les extrêmes du "
            "spectre MBD. ajuster_kappa4 retrouve ses paramètres à partir "
            "d'un échantillon ; kappa4_ppf en déduit les quantiles."),
    "verifie": ("Que les paramètres sont correctement retrouvés, y compris "
                "sur les cas limites k=0 (logistique, Gumbel, exponentielle) "
                "qui faisaient échouer l'ancien code."),
    "donnee": ("On tire 40 000 nombres par INVERSION de la quantile Kappa4 "
               "analytique exacte (kappa4_q), à paramètres imposés. Cette "
               "quantile est indépendante du module ET de scipy (buggé en "
               "k=0, h≤0) : c'est notre étalon de référence."),
    "lecture": ("À droite : chaque point = un quantile ajusté vs sa valeur "
                "vraie. PASS = tous les points collés à la diagonale "
                "(ajusté = vrai) pour les 7 cas."),
    "critere": ("Les 7 cas ajustés avec succès ET erreur relative max sur "
                "les quantiles < 8 %."),
}


def run(outdir):
    m = _th.charger_module()
    rng = np.random.default_rng(3)
    probs = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
    loc, scale = 10.0, 3.0

    fig, ax_data, (ax_v,) = _th.figure_test(1, EXPL, largeur_verif=5.6)

    err_rel_max = 0.0
    n_ok = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(CAS)))
    for cas, col in zip(CAS, colors):
        k, h = cas["k"], cas["h"]
        u = np.clip(rng.random(40000), 1e-9, 1.0 - 1e-9)
        data = kappa4_q(u, loc, scale, k, h)
        ppf_th = kappa4_q(probs, loc, scale, k, h)

        p = m.ajuster_kappa4(data)
        ok = p.get("success", False)
        n_ok += int(ok)
        ppf_fit = np.array([m.kappa4_ppf(p, pr) if ok else np.nan
                            for pr in probs], dtype=float)
        if ok and np.all(np.isfinite(ppf_fit)):
            denom = np.maximum(np.abs(ppf_th), 1e-9)
            err_rel_max = max(err_rel_max,
                              np.max(np.abs(ppf_fit - ppf_th) / denom))
        ax_v.plot(ppf_th, ppf_fit, "o", ms=6, color=col,
                  label=f"{cas['nom']} {'OK' if ok else 'KO'}")

        if cas["nom"].startswith("Logistique"):     # panneau donnée d'inférence
            Fg = np.linspace(1e-4, 1.0 - 1e-4, 600)
            xg = kappa4_q(Fg, loc, scale, k, h)
            pdf_g = kappa4_pdf(xg, loc, scale, k, h, Fg)
            ax_data.hist(data, bins=80, density=True, range=(
                np.percentile(data, 0.5), np.percentile(data, 99.5)),
                color="#90caf9", alpha=0.8,
                label="échantillon (40 000 tirages)")
            ax_data.plot(xg, pdf_g, "r-", lw=2,
                         label="densité Kappa4 vraie")
            ax_data.set_xlim(np.percentile(data, 0.5),
                             np.percentile(data, 99.5))
            ax_data.set_title("Donnée d'inférence — cas logistique")
            ax_data.set_xlabel("valeur"); ax_data.set_ylabel("densité")
            ax_data.legend(fontsize=8)
            ax_data.grid(alpha=0.3)

    lim = [min(ax_v.get_xlim()[0], ax_v.get_ylim()[0]),
           max(ax_v.get_xlim()[1], ax_v.get_ylim()[1])]
    ax_v.plot(lim, lim, "k--", lw=1, label="ajusté = vrai (idéal)")
    ax_v.set_title("Quantile ajusté vs quantile vrai")
    ax_v.set_xlabel("quantile vrai (étalon)")
    ax_v.set_ylabel("quantile ajusté (fonction)")
    ax_v.grid(alpha=0.3)
    ax_v.legend(fontsize=7.5, loc="upper left")
    _th.legende_data(ax_data, "40 000 tirages d'une Kappa4 à paramètres\n"
                              "imposés (scipy) → la densité rouge est la "
                              "vérité connue d'avance.")

    passed = (n_ok == len(CAS)) and (err_rel_max < 0.08)
    msg = (f"{n_ok}/{len(CAS)} fits OK (strict) ; erreur rel. max PPF "
           f"= {err_rel_max*100:.2f}% (seuil 8%)")
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_kappa4")
    return _th.resultat("ajuster_kappa4 / kappa4_ppf", passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
