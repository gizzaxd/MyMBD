# -*- coding: utf-8 -*-
"""Lance tous les tests unitaires visuels et génère un index HTML récapitulatif.

Usage :
    python tests_unitaires/run_all.py
Sortie : tests_unitaires/_resultats/index.html (+ PNG par test).
"""
import os
import sys
import glob
import html
import importlib.util
import traceback

ICI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ICI, "_resultats")
sys.path.insert(0, ICI)  # pour `import _th` dans les tests


def _charger_test(path):
    nom = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(nom, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    os.makedirs(OUT, exist_ok=True)
    fichiers = sorted(glob.glob(os.path.join(ICI, "test_*.py")))
    resultats = []
    for f in fichiers:
        nom = os.path.basename(f)
        try:
            mod = _charger_test(f)
            res = mod.run(OUT)
        except Exception:
            res = {"nom": nom, "passed": False,
                   "msg": "EXCEPTION:\n" + traceback.format_exc(),
                   "png": None}
        resultats.append(res)
        etat = "PASS" if res["passed"] else "FAIL"
        print(f"[{etat}] {res['nom']} — {res['msg'].splitlines()[0]}")

    n_ok = sum(1 for r in resultats if r["passed"])
    n = len(resultats)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Tests unitaires des fonctions du programme Python MyMBD</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;"
        "color:#212121;background:#fafafa}"
        ".card{border:1px solid #ccc;border-radius:8px;margin:18px 0;"
        "padding:16px;background:#fff}.pass{border-left:10px solid #2e7d32}"
        ".fail{border-left:10px solid #c62828}"
        "img{max-width:100%;border:1px solid #eee;margin-top:10px}"
        "pre{white-space:pre-wrap;background:#f6f6f6;padding:8px}"
        ".expl{background:#f3f6fb;border:1px solid #d0d9e6;border-radius:6px;"
        "padding:10px 14px;margin:8px 0;line-height:1.5}"
        ".expl h3{margin:0 0 6px 0;color:#1a237e;font-size:1.05em}"
        ".expl dt{font-weight:bold;color:#37474f;margin-top:8px}"
        ".expl dd{margin:2px 0 0 0}"
        ".intro{background:#fff8e1;border:1px solid #ffe082;border-radius:6px;"
        "padding:10px 14px;margin:10px 0}</style></head><body>",
        f"<h1>Tests unitaires des fonctions du programme Python MyMBD — {n_ok}/{n} PASS</h1>",
        "<div class='intro'>Chaque test fabrique une donnée dont la "
        "<b>bonne réponse est connue d'avance</b> (théorie exacte ou "
        "construction maîtrisée), appelle la fonction, et compare. "
        "Le bloc bleu explique en clair ce qui est vérifié ; l'image "
        "montre à gauche cette explication, au centre la donnée de test, "
        "à droite la fonction comparée à la référence.</div>",
    ]
    _SEC = [("À quoi sert la fonction", "but"),
            ("Ce que ce test prouve", "verifie"),
            ("Donnée de test (et pourquoi elle est fiable)", "donnee"),
            ("Comment lire les graphiques", "lecture"),
            ("Critère de réussite", "critere")]
    for r in resultats:
        cls = "pass" if r["passed"] else "fail"
        etat = "PASS" if r["passed"] else "FAIL"
        parts.append(f"<div class='card {cls}'><h2>{etat} — "
                     f"{html.escape(str(r['nom']))}</h2>")
        expl = r.get("explication") or {}
        if expl:
            block = ["<div class='expl'>"]
            if expl.get("fonction"):
                block.append(f"<h3>{html.escape(str(expl['fonction']))}</h3>")
            block.append("<dl>")
            for titre, cle in _SEC:
                if expl.get(cle):
                    block.append(f"<dt>{titre}</dt>"
                                 f"<dd>{html.escape(str(expl[cle]))}</dd>")
            block.append("</dl></div>")
            parts.append("".join(block))
        parts.append(f"<pre>{html.escape(str(r['msg']))}</pre>")
        if r["png"]:
            rel = os.path.basename(r["png"])
            parts.append(f"<img src='{rel}'>")
        parts.append("</div>")
    parts.append("</body></html>")

    index = os.path.join(OUT, "index.html")
    with open(index, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"\nIndex : {index}  ({n_ok}/{n} PASS)")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
