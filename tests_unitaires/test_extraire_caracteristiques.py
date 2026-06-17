# -*- coding: utf-8 -*-
"""Test `extraire_caracteristiques` — découpage du signal en blocs."""
import os
import numpy as np
import matplotlib.pyplot as plt
import _th

EXPL = {
    "fonction": "extraire_caracteristiques",
    "but": ("Découpe le signal mesuré en blocs temporels et calcule pour "
            "chaque bloc des indicateurs (moyenne, maximum, …) qui servent "
            "ensuite à classer les blocs et à ajuster les lois."),
    "verifie": ("Que la moyenne et le maximum restitués pour chaque bloc "
                "valent EXACTEMENT les valeurs qu'on a imposées en "
                "construisant le signal."),
    "donnee": ("On fabrique le signal bloc par bloc : chaque bloc est une "
               "rampe dont on IMPOSE la moyenne (5 → 25) et donc le maximum "
               "(moyenne+1). On connaît la bonne réponse, par construction, "
               "pour plusieurs tailles de bloc (Tb = 0.05, 0.2, 1.0 s)."),
    "lecture": ("Centre : le signal construit (les paliers = moyennes "
                "imposées). Droite : moyenne et maximum extraits doivent "
                "tomber pile sur les valeurs imposées (lignes --)."),
    "critere": "Erreur max (moyenne ET maximum) < 1e-6 (quasi exact).",
}


def run(outdir):
    m = _th.charger_module()
    fs = 1000.0
    erreurs = []

    fig, ax_data, (a1, a2) = _th.figure_test(2, EXPL, largeur_verif=4.8)

    for Tb in (0.05, 0.2, 1.0):
        taille = int(round(Tb * fs))
        n_blocs = 25
        means_imposes = np.linspace(5.0, 25.0, n_blocs)
        sig = []
        for mu in means_imposes:
            rampe = np.linspace(0.0, 2.0, taille)      # moyenne = 1.0
            sig.append(mu - 1.0 + rampe)               # moyenne bloc = mu
        sig = np.concatenate(sig)
        max_attendus = means_imposes + 1.0

        feats, maxima, noms, nb, _ = m.extraire_caracteristiques(
            sig, fs, Tb_initial=Tb, feature_flags={"mean": True}, min_ech=5)
        k = min(nb, n_blocs)
        idx_mean = list(noms).index("mean") if "mean" in noms else 0
        mean_obt = np.asarray(feats)[:k, idx_mean]
        max_obt = np.asarray(maxima)[:k]

        erreurs += [np.max(np.abs(mean_obt - means_imposes[:k])),
                    np.max(np.abs(max_obt - max_attendus[:k]))]

        a1.plot(means_imposes[:k], "k--",
                label="imposé" if Tb == 0.05 else None)
        a1.plot(mean_obt, ".", label=f"obtenu Tb={Tb}")
        a2.plot(max_attendus[:k], "k--",
                label="attendu" if Tb == 0.05 else None)
        a2.plot(max_obt, ".", label=f"obtenu Tb={Tb}")
        if Tb == 0.2:                                  # panneau donnée
            ax_data.plot(sig[:taille * 6], lw=0.9, color="#5c6bc0")
            for j in range(6):
                ax_data.axhline(means_imposes[j], color="r", ls=":", lw=0.8)

    ax_data.set_title("Donnée d'inférence : signal construit (Tb=0.2)")
    ax_data.set_xlabel("échantillon"); ax_data.set_ylabel("signal")
    ax_data.grid(alpha=0.3)
    _th.legende_data(ax_data, "Rampes à moyenne imposée (lignes rouges).\n"
                              "La moyenne/max de chaque bloc est connue.")

    a1.set_title("Moyenne par bloc"); a2.set_title("Maximum par bloc")
    for a in (a1, a2):
        a.set_xlabel("indice de bloc")
        a.legend(fontsize=8); a.grid(alpha=0.3)

    err = max(erreurs)
    passed = err < 1e-6
    msg = f"erreur max (mean & maxima) = {err:.2e} (seuil 1e-6)"
    _th.bandeau(fig, passed, msg)
    png = _th.sauver(fig, outdir, "test_extraire_caracteristiques")
    return _th.resultat("extraire_caracteristiques", passed, msg, png, EXPL)


if __name__ == "__main__":
    print(run(os.path.dirname(os.path.abspath(__file__))))
