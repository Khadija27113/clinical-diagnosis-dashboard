"""
feature_selection.py
Selection de variables (chi2 + information mutuelle) pour la classification a 3 classes.

Entrees : merged_data.csv -> feature_engineering.build_features (125 variables numeriques)
          + cible = diagnostic_mapping.map_model (3 classes ; "Autre" = hors perimetre, exclu)
Sorties : ../data/processed/feature_selection_report.csv   (une ligne par variable)
          ../data/processed/feature_selection_summary.txt (lecture rapide, pour le memoire)

Ce que fait ce script
---------------------
1. Definit 3 jeux de variables a comparer :
     baseline17 : les 17 variables d'origine (ages, sexe, scolarite, 13 symptomes)
     A          : variables factuelles (niveau A)
     A+B        : toutes (A + tableau clinique documente, niveau B)
2. Classe les variables par chi2, par information mutuelle (MI) et par rang combine.
3. Mesure la STABILITE : a quelle frequence chaque variable est dans le top-k quand on
   refait le classement sur 50 echantillons d'apprentissage (validation croisee 5x10).
4. Calcule un SEUIL DE BRUIT : le meilleur score qu'une variable obtient par hasard parmi N
   quand on melange les etiquettes (95e centile du maximum, 100 permutations).
5. Demontre le biais d'une selection faite AVANT la validation croisee (etiquettes melangees).

Regle d'or : toute selection utilisee pour evaluer un modele doit etre refaite sur le
jeu d'apprentissage de chaque pli (fonction select_columns). Les scores calcules sur
l'ensemble des patients ne servent qu'a decrire, jamais a evaluer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import diagnostic_mapping as dm
import feature_engineering as fe

OUTPUT_DIR = fe.OUTPUT_DIR
BASELINE17_SOURCES = ["age_apparition_premiers_symptomes", "age_premiere_consultation_generale",
                      "sexe", "niveau_scolaire"] + fe.SYMPTOM_COLS
SHORT = {dm.CLASS_ANXIETE: "anxiete", dm.CLASS_HUMEUR: "humeur", dm.CLASS_PSY: "psychotique"}
METHODS = ("chi2", "mi", "combined")
_quiet = lambda *_: None


# ------------------------------------------------------------
# 1. Donnees : matrice, cible, jeux de variables
# ------------------------------------------------------------
def load_dataset():
    """X (variables numeriques), y (3 classes), D (dictionnaire), ids ; patients hors perimetre exclus."""
    merged = pd.read_csv(fe.DATA_PATH, encoding="utf-8")
    X_all, D = fe.build_features(merged, _quiet)
    y_all = merged["diagnostic_principal"].apply(dm.map_model)
    keep = (y_all != dm.OUT_OF_SCOPE).values
    X = X_all[keep].reset_index(drop=True)
    y = y_all[keep].reset_index(drop=True)
    ids = merged.loc[keep, fe.ID_COL].reset_index(drop=True)
    return X, y, D, ids


def get_feature_sets(D):
    return {
        "baseline17": D.loc[D.source_column.isin(BASELINE17_SOURCES), "feature"].tolist(),
        "A": D.loc[D.tier == "A", "feature"].tolist(),
        "A+B": D["feature"].tolist(),
    }


def discrete_mask(X, max_levels=10):
    """Variables a peu de valeurs distinctes = discretes pour l'information mutuelle
    (decision non supervisee : ne regarde pas la cible)."""
    return (X.nunique() <= max_levels).values


# ------------------------------------------------------------
# 2. Scores (calcules sur le jeu qu'on leur donne : a appeler sur l'apprentissage SEUL)
# ------------------------------------------------------------
def score_chi2(X, y):
    """chi2 sur variables ramenees a [0,1] (chi2 exige >= 0 et depend de l'echelle)."""
    Xs = MinMaxScaler().fit_transform(np.asarray(X, dtype=float))
    return np.nan_to_num(chi2(Xs, np.asarray(y))[0])


def score_mi(X, y, mask, seed=0):
    return mutual_info_classif(np.asarray(X, dtype=float), np.asarray(y),
                               discrete_features=mask, random_state=seed)


def score_combined(chi2_scores, mi_scores):
    """Rang moyen chi2/MI (compare des echelles differentes) ; plus haut = meilleur."""
    return -(rankdata(-chi2_scores) + rankdata(-mi_scores)) / 2


def rank_columns(X_train, y_train, method="combined", mask=None):
    """TOUTES les colonnes, de la meilleure a la moins bonne, calculees sur (X_train, y_train)
    UNIQUEMENT. A utiliser a l'interieur de chaque pli ; le top-k s'obtient par tranche
    ([:k]), ce qui evite de recalculer le classement pour chaque valeur de k."""
    assert method in METHODS, method
    mask = discrete_mask(X_train) if mask is None else mask
    s_chi2 = score_chi2(X_train, y_train) if method in ("chi2", "combined") else None
    s_mi = score_mi(X_train, y_train, mask) if method in ("mi", "combined") else None
    scores = {"chi2": s_chi2, "mi": s_mi,
              "combined": score_combined(s_chi2, s_mi) if method == "combined" else None}[method]
    return list(X_train.columns[np.argsort(-scores, kind="stable")])


def select_columns(X_train, y_train, k, method="combined", mask=None):
    """Les k meilleures colonnes, calculees sur (X_train, y_train) UNIQUEMENT."""
    return rank_columns(X_train, y_train, method, mask)[:k]


# ------------------------------------------------------------
# 3. Stabilite et seuil de bruit
# ------------------------------------------------------------
def stability(X, y, columns, mask_full, ks=(10, 20), n_splits=5, n_repeats=10, seed=42):
    """Frequence (0-1) a laquelle chaque variable est dans le top-k (rang combine)
    quand on recalcule le classement sur chaque jeu d'apprentissage."""
    cols = list(columns)
    idx = [X.columns.get_loc(c) for c in cols]
    mask = mask_full[idx]
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    hits = {k: np.zeros(len(cols)) for k in ks}
    n_folds = 0
    for tr, _ in cv.split(X, y):
        Xtr, ytr = X.iloc[tr][cols], y.iloc[tr]
        comb = score_combined(score_chi2(Xtr, ytr), score_mi(Xtr, ytr, mask))
        order = np.argsort(-comb, kind="stable")
        for k in ks:
            hits[k][order[:k]] += 1
        n_folds += 1
    return {k: pd.Series(v / n_folds, index=cols) for k, v in hits.items()}


def noise_threshold(X, y, columns, mask_full, n_perm=100, q=95, seed=0):
    """Meilleur score qu'une variable obtient PAR HASARD parmi len(columns) : on melange les
    etiquettes (en gardant les effectifs), on prend le maximum des scores, et on retient
    le q-ieme centile sur n_perm permutations. Un score observe sous ce seuil n'est pas
    distinguable du bruit."""
    cols = list(columns)
    mask = mask_full[[X.columns.get_loc(c) for c in cols]]
    rng = np.random.default_rng(seed)
    max_chi2, max_mi = [], []
    for _ in range(n_perm):
        yp = rng.permutation(y.values)
        max_chi2.append(score_chi2(X[cols], yp).max())
        max_mi.append(score_mi(X[cols], yp, mask).max())
    return float(np.percentile(max_chi2, q)), float(np.percentile(max_mi, q))


# ------------------------------------------------------------
# 4. Demonstration du biais de selection hors validation croisee
# ------------------------------------------------------------
def leakage_demo(X, y, columns, mask_full, k=10, n_perm=5, seed=0):
    """Avec des etiquettes MELANGEES (aucun signal), un modele ne doit pas depasser le hasard.
    - MAUVAIS protocole : selection sur tous les patients PUIS validation croisee.
    - BON protocole     : selection refaite dans chaque pli, sur l'apprentissage seul."""
    cols = list(columns)
    mask = mask_full[[X.columns.get_loc(c) for c in cols]]
    rng = np.random.default_rng(seed)
    wrong, right = [], []
    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    for _ in range(n_perm):
        yp = pd.Series(rng.permutation(y.values))
        cv = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e6)))
        sel_all = select_columns(X[cols], yp, k, "combined", mask)           # fuite : voit tous les patients
        for tr, te in cv.split(X, yp):
            m = model().fit(X.iloc[tr][sel_all], yp.iloc[tr])
            wrong.append(balanced_accuracy_score(yp.iloc[te], m.predict(X.iloc[te][sel_all])))
            sel_tr = select_columns(X.iloc[tr][cols], yp.iloc[tr], k, "combined", mask)   # correct
            m = model().fit(X.iloc[tr][sel_tr], yp.iloc[tr])
            right.append(balanced_accuracy_score(yp.iloc[te], m.predict(X.iloc[te][sel_tr])))
    return float(np.mean(wrong)), float(np.mean(right)), 1 / y.nunique()


# ------------------------------------------------------------
# 5. Rapport
# ------------------------------------------------------------
def main():
    X, y, D, ids = load_dataset()
    sets = get_feature_sets(D)
    mask = discrete_mask(X)
    lines = []
    out = lambda s="": (print(s), lines.append(s))

    out(f"{len(X)} patients (hors 'Autre') | classes : {y.value_counts().to_dict()}")
    out("Jeux de variables : " + " | ".join(f"{n} = {len(c)}" for n, c in sets.items()))
    out(f"Variables continues (> 10 valeurs) pour l'information mutuelle : {int((~mask).sum())}\n")

    # scores descriptifs sur tous les patients (NE SERVENT PAS a evaluer un modele)
    full_chi2 = pd.Series(score_chi2(X, y), index=X.columns)
    full_mi = pd.Series(score_mi(X, y, mask), index=X.columns)
    means = X.groupby(y).mean().T.rename(columns=SHORT)

    report = D.set_index("feature")[["source_column", "kind", "tier"]].copy()
    report["dans_baseline17"] = report.index.isin(sets["baseline17"])
    report = report.join(means.add_prefix("moy_"))
    report["chi2"], report["mi"] = full_chi2, full_mi

    thresholds = {}
    for name, cols in sets.items():
        freq = stability(X, y, cols, mask)
        thr_chi2, thr_mi = noise_threshold(X, y, cols, mask)
        thresholds[name] = (thr_chi2, thr_mi)
        safe = name.replace("+", "")
        for k, f in freq.items():
            report[f"freq_top{k}_{safe}"] = f
        sub = report.loc[cols]
        above = sub[(sub.chi2 > thr_chi2) | (sub.mi > thr_mi)]
        out(f"=== Jeu {name} ({len(cols)} variables) ===")
        out(f"Seuil de bruit (95e centile du max sous etiquettes melangees) : chi2 = {thr_chi2:.2f} | MI = {thr_mi:.3f}")
        out(f"Variables au-dessus du bruit : {len(above)} / {len(cols)}")
        top = sub.assign(stabilite_top10=freq[10]).sort_values(["stabilite_top10", "mi"], ascending=False).head(10)
        show = top[["tier", "chi2", "mi", "stabilite_top10"] + [c for c in top.columns if c.startswith("moy_")]]
        out(show.round(3).to_string())
        out()

    # rang combine sur tous les patients (descriptif)
    report["rang_combine_AB"] = pd.Series(
        rankdata(-score_combined(full_chi2.values, full_mi.values)), index=report.index)
    report["au_dessus_bruit_AB"] = (report.chi2 > thresholds["A+B"][0]) | (report.mi > thresholds["A+B"][1])
    report = report.sort_values("rang_combine_AB")
    report.round(4).to_csv(OUTPUT_DIR / "feature_selection_report.csv", encoding="utf-8-sig")

    wrong, right, chance = leakage_demo(X, y, sets["A+B"], mask)
    out("=== Biais d'une selection faite AVANT la validation croisee (etiquettes MELANGEES, aucun signal) ===")
    out(f"Hasard attendu (balanced accuracy)                  : {chance:.3f}")
    out(f"Selection sur tous les patients puis validation     : {wrong:.3f}   <- gonfle (a eviter)")
    out(f"Selection refaite dans chaque pli (bon protocole)   : {right:.3f}")
    (OUTPUT_DIR / "feature_selection_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFichiers : {OUTPUT_DIR / 'feature_selection_report.csv'} ; {OUTPUT_DIR / 'feature_selection_summary.txt'}")


if __name__ == "__main__":
    main()