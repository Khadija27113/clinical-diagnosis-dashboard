"""
naive_bayes.py
Naive Bayes ecrit a la main pour la matrice numerique de model_features.csv.

Chaque colonne recoit la loi adaptee a son type, DECIDE sur les donnees d'apprentissage :
  - binaire (valeurs dans {0,1})        -> Bernoulli
  - discrete (<= max_levels valeurs)    -> loi categorielle  (ordinales, compteurs)
  - continue                            -> loi normale       (ages, delais, rang relatif)

Deux lissages empechent les probabilites 0 % / 100 % sur les petites classes :
  alpha     : lissage additif vers la frequence GLOBALE
                P = (compte_classe + alpha * freq_globale) / (n_classe + alpha)
  shrink_k  : rapprochement vers la moyenne / l'ecart-type GLOBAUX (lois normales)
                w = n_classe / (n_classe + shrink_k)
                moyenne = w * moyenne_classe + (1 - w) * moyenne_globale   (idem pour l'ecart-type)

Priors de classe : "empirical" (frequences observees) ou "uniform" (1/C : plus equitable
pour la classe rare quand on vise le macro-F1).

explain() repond a "pourquoi cette categorie ?" : contribution de chaque variable au
rapport de vraisemblance entre la classe predite et la deuxieme classe.

Limites : les variables sont supposees independantes sachant la classe (les colonnes
one-hot d'une meme variable et les variables derivees redondantes sont donc comptees
plusieurs fois) ; les probabilites sont souvent trop confiantes.
"""

import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, alpha=2.0, shrink_k=5.0, prior="empirical", max_levels=10, floor=1e-6):
        assert prior in ("empirical", "uniform")
        self.alpha, self.shrink_k, self.prior = alpha, shrink_k, prior
        self.max_levels, self.floor = max_levels, floor

    # ---------------------------------------------------------
    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y)
        assert not X.isna().any().any(), "NaN dans X : imputer avant d'entrainer"
        self.columns_ = list(X.columns)
        self.classes_ = np.array(sorted(set(y)))
        C, n = len(self.classes_), len(X)
        self.class_counts_ = np.array([(y == c).sum() for c in self.classes_])
        assert (self.class_counts_ > 0).all()
        self.log_prior_ = (np.log(self.class_counts_ / n) if self.prior == "empirical"
                           else np.full(C, -np.log(C)))
        Xa = X.to_numpy(dtype=float)
        onehot_y = (y[:, None] == self.classes_[None, :]).astype(float)        # (n, C)

        # --- colonnes binaires : Bernoulli, calcul vectorise
        b_idx = np.where(np.all((Xa == 0) | (Xa == 1), axis=0))[0]
        Xb = Xa[:, b_idx]
        g = Xb.mean(axis=0)
        p = (onehot_y.T @ Xb + self.alpha * g) / (self.class_counts_[:, None] + self.alpha)   # (C, d_b)
        p = np.clip(p, self.floor, 1 - self.floor)
        self.bern_ = {"idx": b_idx, "logp1": np.log(p), "logp0": np.log1p(-p)}

        # --- autres colonnes : categorielle (peu de valeurs) ou normale (continue)
        self.cat_, self.gauss_ = [], []
        for j in np.setdiff1d(np.arange(Xa.shape[1]), b_idx):
            x = Xa[:, j]
            levels = np.unique(x)
            if len(levels) <= self.max_levels:
                gf = np.array([(x == v).mean() for v in levels])
                counts = np.array([[(x[y == c] == v).sum() for v in levels] for c in self.classes_])
                pc = (counts + self.alpha * gf) / (self.class_counts_[:, None] + self.alpha)
                self.cat_.append({"idx": j, "levels": levels, "logp": np.log(np.clip(pc, self.floor, 1.0))})
            else:
                g_mean, g_std = x.mean(), x.std(ddof=1)
                means, stds = np.zeros(C), np.zeros(C)
                for i, c in enumerate(self.classes_):
                    xc, n_c = x[y == c], self.class_counts_[i]
                    s_c = xc.std(ddof=1) if n_c > 1 else g_std
                    w = n_c / (n_c + self.shrink_k)
                    means[i] = w * xc.mean() + (1 - w) * g_mean
                    stds[i] = w * s_c + (1 - w) * g_std
                stds = np.maximum(stds, max(0.1 * g_std, 1e-3))
                self.gauss_.append({"idx": j, "mean": means, "std": stds})
        return self

    # ---------------------------------------------------------
    def _loglik(self, X):
        """Tenseur (n, d, C) : log P(valeur_j | classe) pour chaque patient et chaque variable."""
        Xa = pd.DataFrame(X)[self.columns_].to_numpy(dtype=float)
        n, d, C = Xa.shape[0], len(self.columns_), len(self.classes_)
        L = np.zeros((n, d, C))
        b = self.bern_
        L[:, b["idx"], :] = np.where(Xa[:, b["idx"], None] == 1, b["logp1"].T[None, :, :], b["logp0"].T[None, :, :])
        for prm in self.cat_:
            x, levels = Xa[:, prm["idx"]], prm["levels"]
            idx = np.clip(np.searchsorted(levels, x), 0, len(levels) - 1)
            known = levels[idx] == x
            L[:, prm["idx"], :] = np.where(known[:, None], prm["logp"][:, idx].T, np.log(self.floor))
        for prm in self.gauss_:
            z = (Xa[:, prm["idx"], None] - prm["mean"][None, :]) / prm["std"][None, :]
            L[:, prm["idx"], :] = -0.5 * z ** 2 - np.log(prm["std"])[None, :] - 0.5 * np.log(2 * np.pi)
        return L

    def predict_log_joint(self, X):
        return self.log_prior_[None, :] + self._loglik(X).sum(axis=1)

    def predict_proba(self, X):
        lj = self.predict_log_joint(X)
        lj = lj - lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_joint(X), axis=1)]

    # ---------------------------------------------------------
    def explain(self, x, top=5):
        """Pourquoi cette categorie ? Pour UN patient (Series ou dict de variables).
        Retourne la classe predite, la 2e classe, le terme de prior et, pour chaque variable,
        sa contribution au log du rapport de vraisemblance (predite / 2e classe) :
        positive = pousse vers la classe predite, negative = pousse vers la 2e."""
        row = pd.DataFrame([dict(x)])[self.columns_]
        L = self._loglik(row)[0]                                  # (d, C)
        joint = self.log_prior_ + L.sum(axis=0)
        order = np.argsort(-joint)
        pred, second = order[0], order[1]
        contrib = L[:, pred] - L[:, second]
        table = pd.DataFrame({"variable": self.columns_, "valeur": row.iloc[0].to_numpy(),
                              "contribution": contrib}).sort_values("contribution", ascending=False)
        return {
            "classe_predite": self.classes_[pred], "deuxieme_classe": self.classes_[second],
            "terme_prior": float(self.log_prior_[pred] - self.log_prior_[second]),
            "log_rapport_total": float(joint[pred] - joint[second]),
            "pour": table.head(top).reset_index(drop=True),
            "contre": table[table.contribution < 0].tail(top).iloc[::-1].reset_index(drop=True),
            "toutes": table.reset_index(drop=True),
        }