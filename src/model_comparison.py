"""
model_comparison.py
Compare Naive Bayes (ecrit a la main) et Random Forest sur 4 jeux de variables, par
validation croisee IMBRIQUEE, pour la classification a 3 classes.

Jeux de variables : baseline17 | A (factuelles) | A+B (toutes) | A+B sans variables derivees
Protocole (aucune information du pli de TEST ne sert a choisir quoi que ce soit) :

  BOUCLE EXTERNE  5 plis x n_repeats repetitions (stratifies)  -> estimation de la performance
    BOUCLE INTERNE  3 plis sur le jeu d'apprentissage de ce pli seulement
        - classement des variables (chi2 + MI, rang combine) refait sur l'apprentissage interne
        - evalue chaque configuration : k (nombre de variables) x hyperparametres du modele
        - garde la meilleure (macro-F1 sur les predictions interieures)
    la configuration choisie est ensuite entrainee sur TOUT l'apprentissage externe
    (classement refait) et evaluee sur le pli de test.

Metrique principale : macro-F1 (moyenne des F1 des 3 classes), plus F1 par classe et
balanced accuracy. Les predictions des 5 plis externes d'une repetition forment une
prediction "hors echantillon" pour chaque patient ; on en tire une valeur par repetition.
Reference : classe majoritaire et hasard stratifie. Cible fixee : macro-F1 >= 0.75.

Sorties (../data/processed/) : model_comparison_results.csv, _configs.csv, _tests.csv,
model_comparison_summary.txt.   Usage :  python model_comparison.py [--repeats 3]
"""

import argparse
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import diagnostic_mapping as dm
import feature_selection as fs
from naive_bayes import NaiveBayes

CLASSES = sorted(dm.MODEL_CLASSES)
N_SPLITS = 5
TARGET = 0.75
K_GRID = [5, 8, 10, 15, 20, None]          # None = toutes les variables du jeu
NB_GRID = [dict(alpha=a, shrink_k=s, prior=p)
           for a in (0.5, 2.0, 5.0) for s in (2.0, 15.0) for p in ("empirical", "uniform")]
# Random Forest : seul k est regle (grille identique a celle du Naive Bayes) ; le reste est fixe
# (100 arbres, classes equilibrees par sous-echantillon, feuilles >= 2) : la forêt est peu sensible
# a ces reglages, et chaque entrainement coute ~0,3 s sur une machine a un coeur.
RF_GRID = [dict(min_samples_leaf=2)]
RF_TREES = 100


def make_model(name, params):
    if name == "NB":
        return NaiveBayes(**params)
    return RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced_subsample",
                                  random_state=0, n_jobs=1, **params)


def make_configs(model, n_cols, k_grid=K_GRID, nb_grid=None, rf_grid=None):
    ks = []
    for k in k_grid:
        k_eff = n_cols if (k is None or k >= n_cols) else k
        if k_eff not in ks:
            ks.append(k_eff)
    grid = (nb_grid or NB_GRID) if model == "NB" else (rf_grid or RF_GRID)
    return [dict(k=k, **p) for k in ks for p in grid]


def fit_predict(model, cfg, ranking, Xtr, ytr, Xte):
    cols = ranking[:cfg["k"]]
    params = {a: b for a, b in cfg.items() if a != "k"}
    return make_model(model, params).fit(Xtr[cols], ytr).predict(Xte[cols])


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)


def summarize(pred, y):
    """pred : matrice (repetitions, n) de predictions hors echantillon -> moyennes sur les repetitions."""
    yv = np.asarray(y)
    rows = []
    for r in range(pred.shape[0]):
        rows.append({
            "macro_f1": macro_f1(yv, pred[r]),
            "balanced_acc": balanced_accuracy_score(yv, pred[r]),
            **{f"f1_{c}": f for c, f in zip(CLASSES, f1_score(yv, pred[r], average=None, labels=CLASSES, zero_division=0))},
        })
    df = pd.DataFrame(rows)
    return df.mean().to_dict() | {"macro_f1_sd": df["macro_f1"].std(ddof=1) if len(df) > 1 else np.nan}


# ------------------------------------------------------------
# Un jeu de variables : validation croisee imbriquee
# ------------------------------------------------------------
def run_set(X, y, cols, mask_all, models=("NB", "RF"), k_grid=K_GRID, nb_grid=None, rf_grid=None,
            n_repeats=3, inner_splits=3, seed=42, log=None):
    cols = list(cols)
    Xs = X[cols]
    mask = mask_all[[X.columns.get_loc(c) for c in cols]]
    n = len(y)
    configs = {m: make_configs(m, len(cols), k_grid, nb_grid, rf_grid) for m in models}
    pooled = {(m, i): np.empty((n_repeats, n), dtype=object) for m in models for i in range(len(configs[m]))}
    nested = {m: np.empty((n_repeats, n), dtype=object) for m in models}
    dummy = {s: np.empty((n_repeats, n), dtype=object) for s in ("most_frequent", "stratified")}
    chosen = {m: [] for m in models}
    fold_scores = {m: [] for m in models}

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=seed)
    t0 = time.time()
    for f, (tr, te) in enumerate(cv.split(Xs, y)):
        r = f // N_SPLITS
        Xtr, ytr, Xte, yte = Xs.iloc[tr], y.iloc[tr], Xs.iloc[te], y.iloc[te]

        # --- boucle interne : choix de la configuration sur l'apprentissage seul
        inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed + f)
        oof = {(m, i): np.empty(len(tr), dtype=object) for m in models for i in range(len(configs[m]))}
        for itr, iva in inner.split(Xtr, ytr):
            rank_in = fs.rank_columns(Xtr.iloc[itr], ytr.iloc[itr], "combined", mask)
            for m in models:
                for i, cfg in enumerate(configs[m]):
                    oof[(m, i)][iva] = fit_predict(m, cfg, rank_in, Xtr.iloc[itr], ytr.iloc[itr], Xtr.iloc[iva])
        best = {m: int(np.argmax([macro_f1(ytr, oof[(m, i)]) for i in range(len(configs[m]))])) for m in models}

        # --- boucle externe : toutes les configs sur le test (courbe) ; la choisie = estimation imbriquee
        rank_out = fs.rank_columns(Xtr, ytr, "combined", mask)
        for m in models:
            for i, cfg in enumerate(configs[m]):
                pred = fit_predict(m, cfg, rank_out, Xtr, ytr, Xte)
                pooled[(m, i)][r, te] = pred
                if i == best[m]:
                    nested[m][r, te] = pred
                    chosen[m].append(cfg)
                    fold_scores[m].append(macro_f1(yte, pred))
        for s in dummy:
            dummy[s][r, te] = DummyClassifier(strategy=s, random_state=f).fit(Xtr, ytr).predict(Xte)
        if log and (f + 1) % N_SPLITS == 0:
            log(f"      repetition {r + 1}/{n_repeats} terminee ({time.time() - t0:.0f} s)")

    out = {"nested": {m: summarize(nested[m], y) for m in models},
           "baselines": {s: summarize(dummy[s], y) for s in dummy},
           "chosen": chosen, "fold_scores": fold_scores, "configs": []}
    for m in models:
        for i, cfg in enumerate(configs[m]):
            out["configs"].append({"modele": m, **cfg, **summarize(pooled[(m, i)], y)})
    return out


def corrected_ttest(a, b, test_over_train=1 / (N_SPLITS - 1)):
    """Test t apparie corrige (Nadeau & Bengio) : les plis se recouvrent, donc la variance
    est majoree. a, b : scores par pli (memes plis). Retourne (diff moyenne, p-valeur)."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:   # differences toutes identiques
        return float(d.mean()), (1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / np.sqrt((1 / J + test_over_train) * var)
    return float(d.mean()), float(2 * stats.t.sf(abs(t), df=J - 1))


# ------------------------------------------------------------
# Experience complete
# ------------------------------------------------------------
def get_sets(D):
    sets = fs.get_feature_sets(D)
    sets["A+B sans derivees"] = D.loc[D.source_column != "derivee", "feature"].tolist()
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    X, y, D, ids = fs.load_dataset()
    mask_all = fs.discrete_mask(X)
    sets = get_sets(D)
    log = lambda s: (print(s), sys.stdout.flush())
    log(f"{len(X)} patients | classes : {y.value_counts().to_dict()}")
    log(f"Validation croisee imbriquee : externe 5 x {args.repeats}, interne 3 | cible macro-F1 >= {TARGET}\n")

    results, configs, tests, lines = [], [], [], []
    for name, cols in sets.items():
        t0 = time.time()
        log(f"=== Jeu {name} ({len(cols)} variables) ===")
        out = run_set(X, y, cols, mask_all, n_repeats=args.repeats, log=log)
        for m, s in out["nested"].items():
            freq = Counter((c["k"], tuple(sorted((a, b) for a, b in c.items() if a != "k"))) for c in out["chosen"][m])
            (k_top, p_top), n_top = freq.most_common(1)[0]
            results.append({"jeu": name, "modele": {"NB": "Naive Bayes", "RF": "Random Forest"}[m],
                            "macro_f1": s["macro_f1"], "macro_f1_sd": s["macro_f1_sd"],
                            **{f"f1_{c}": s[f"f1_{c}"] for c in CLASSES}, "balanced_acc": s["balanced_acc"],
                            "macro_f1_plis_moy": float(np.mean(out["fold_scores"][m])),
                            "config_la_plus_choisie": f"k={k_top} {dict(p_top)} ({n_top}/{len(out['chosen'][m])} plis)"})
        for b, s in out["baselines"].items():
            results.append({"jeu": name, "modele": {"most_frequent": "Classe majoritaire", "stratified": "Hasard stratifie"}[b],
                            "macro_f1": s["macro_f1"], "macro_f1_sd": s["macro_f1_sd"],
                            **{f"f1_{c}": s[f"f1_{c}"] for c in CLASSES}, "balanced_acc": s["balanced_acc"]})
        configs += [{"jeu": name, **c} for c in out["configs"]]
        diff, p = corrected_ttest(out["fold_scores"]["NB"], out["fold_scores"]["RF"])
        tests.append({"jeu": name, "diff_macro_f1_NB_moins_RF": diff, "p_valeur_corrigee": p})
        log(f"    NB = {out['nested']['NB']['macro_f1']:.3f} | RF = {out['nested']['RF']['macro_f1']:.3f} "
            f"| NB-RF = {diff:+.3f} (p corrige = {p:.2f}) | {time.time() - t0:.0f} s\n")

    res = pd.DataFrame(results)
    outdir = fs.OUTPUT_DIR
    res.round(4).to_csv(outdir / "model_comparison_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(configs).round(4).to_csv(outdir / "model_comparison_configs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tests).round(4).to_csv(outdir / "model_comparison_tests.csv", index=False, encoding="utf-8-sig")

    show = res.copy()
    short = {c: c.split(" ")[0] for c in CLASSES}
    show = show.rename(columns={f"f1_{c}": f"F1 {short[c]}" for c in CLASSES})
    summary = [f"RESULTATS (validation croisee imbriquee, macro-F1 moyen sur {args.repeats} repetitions ; cible {TARGET})", "",
               show.drop(columns=["config_la_plus_choisie"], errors="ignore").round(3).to_string(index=False), "",
               "NB vs RF (test t corrige) :", pd.DataFrame(tests).round(3).to_string(index=False)]
    (outdir / "model_comparison_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    log("\n".join(summary))


if __name__ == "__main__":
    main()