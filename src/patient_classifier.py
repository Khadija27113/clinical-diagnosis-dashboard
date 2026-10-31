"""
patient_classifier.py
Modele final de classification utilise par le dashboard.

    patients (merged_data.csv) -> encodage (feature_engineering) -> classement chi2+MI
    (feature_selection) -> K meilleures variables -> Naive Bayes (naive_bayes) -> probabilites
    CALIBREES -> categorie + explication pour UN patient saisi a la main.

Choix (voir model_comparison.py) :
  - jeu "A+B sans variables derivees" : un patient saisi a la main n'a pas a fournir de variables calculees ;
  - K = 10 : la courbe est plate de 5 a 15 variables (macro-F1 0.765 a 0.772) ; 10 donne la meme
    performance avec un formulaire plus court ;
  - reglages du Naive Bayes par defaut (alpha = 2, shrink_k = 5, prior empirique) : sans effet mesurable.

Probabilites calibrees : le Naive Bayes brut est trop confiant. Une temperature T (1 parametre) est ajustee
sur les predictions HORS ECHANTILLON (selection refaite dans chaque pli) ; les probabilites affichees sont
softmax(log_joint / T). La fiabilite par tranche de confiance est mesuree sur ces memes predictions.

IMPORTANT : la confiance NE permet PAS de reperer un patient hors perimetre (autre diagnostic) : le
modele repond toujours parmi ses 3 classes. Seuls les cas ambigus (confiance < UNCERTAIN_BELOW) sont signales.

Tout est entraine au demarrage a partir de merged_data.csv : aucun modele issu de donnees patients
n'est stocke dans le depot.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

import diagnostic_mapping as dm
import feature_engineering as fe
import feature_selection as fs
from naive_bayes import NaiveBayes

K_FEATURES = 10
UNCERTAIN_BELOW = 0.60
BANDS = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.0001)]

SOURCE_LABELS = {
    "anxiete_anticipatoire": "Anxiété anticipatoire", "attaques_panique_recentes": "Attaques de panique récentes",
    "compulsions": "Compulsions", "obsessions": "Obsessions", "symptomes_negatifs": "Symptômes négatifs",
    "syndrome_anxieux": "Syndrome anxieux", "syndrome_delirant": "Syndrome délirant",
    "syndrome_depressif": "Syndrome dépressif", "syndrome_desorganisation": "Syndrome de désorganisation",
    "syndrome_excitation_psychomotrice": "Excitation psychomotrice", "syndrome_hallucinatoire": "Syndrome hallucinatoire",
    "syndrome_phobique": "Syndrome phobique", "troubles_cognitifs": "Troubles cognitifs",
    "atcd_personnels_psychiatriques": "Antécédents psychiatriques personnels",
    "atcd_familiaux_psy_trouble": "Antécédents familiaux psychiatriques (trouble)",
    "atcd_familiaux_psy_membre": "Antécédents familiaux psychiatriques (proche atteint)",
    "atcd_familiaux_mc": "Antécédents familiaux médicaux", "atcd_personnels_mc": "Antécédents médicaux personnels",
    "atcd_tentative_suicide": "Antécédent de tentative de suicide", "nb_tentatives_suicide": "Nombre de tentatives de suicide",
    "niveau_suicidalite": "Niveau de suicidalité", "fonctionnement_socioprofessionnel": "Fonctionnement socioprofessionnel",
    "niveau_activite_sociale": "Activité sociale", "sommeil": "Sommeil", "alimentation": "Alimentation",
    "hetero_agressivite": "Hétéro-agressivité", "sexe": "Sexe", "niveau_scolaire": "Niveau scolaire",
    "niveau_socioeconomique": "Niveau socio-économique", "etat_civil": "État civil",
    "situation_professionnelle": "Situation professionnelle", "milieu_residence": "Milieu de résidence",
    "situation_familiale": "Situation familiale", "situation_residentielle": "Situation résidentielle",
    "type_traumatisme_emotionnel": "Traumatismes", "severite_traumatisme": "Sévérité du traumatisme",
    "atcd_abus_spa": "Substances psychoactives", "atcd_violence": "Antécédents de violence",
    "atcd_judiciaire": "Antécédents judiciaires", "consanguinite_parents": "Consanguinité des parents",
    "couverture_sociale": "Couverture sociale", "nombre_redoublements": "Nombre de redoublements",
    "rang_dans_fratrie": "Rang dans la fratrie", "taille_fratrie": "Taille de la fratrie",
    "age_apparition_premiers_symptomes": "Âge d'apparition des premiers symptômes",
    "age_premiere_consultation_generale": "Âge à la 1re consultation générale",
    "age_premiere_consultation_service": "Âge à la 1re consultation au service",
    "etat_clinique_premiere_consult": "État clinique à la 1re consultation",
}


def pretty(text):
    return str(text).replace("_", " ")


def source_label(src):
    return SOURCE_LABELS.get(src, src.replace("_", " ").capitalize())


def _modalities(value):
    """Cellule multi-valeurs -> ensemble de modalites (absences retirees)."""
    if pd.isna(value):
        return set()
    return {p.strip() for p in str(value).split(";")} - fe.ABSENCE_LABELS - {""}


def _feature_info(feat, row):
    """Comment encoder UNE variable du dictionnaire a partir de la valeur saisie de sa colonne source."""
    src, kind, definition = row["source_column"], row["kind"], row["definition"]
    info = {"feature": feat, "source": src, "kind": kind, "tier": row["tier"]}
    if kind == "numerique":
        info["mode"] = "number"
    elif kind == "ordinal":
        info.update(mode="ordinal", order=fe.ORDINAL[src][0])
    elif kind == "binaire" and feat == src:
        info["mode"] = "yes_no"
    elif kind == "binaire":                                   # 2 modalites : une seule colonne, 1 = modalite
        info.update(mode="equals", modality=definition.replace("1 = ", "", 1))
    elif kind in ("nominal", "multi-hot"):
        if feat.endswith("__autre"):
            info.update(mode="in_rare", rare=definition.split(" : ", 1)[1].split(" | "))
        else:
            info.update(mode="equals" if kind == "nominal" else "has", modality=definition.replace("1 = ", "", 1))
    else:
        raise NotImplementedError(f"variable derivee non supportee dans le formulaire : {feat}")
    return info


def _input_spec(src, merged):
    """Description du champ de saisie d'une colonne source (type de widget, choix, valeur par defaut)."""
    spec = {"source": src, "label": source_label(src)}
    if src in fe.YES_NO:
        spec.update(kind="yes_no", choices=["Non", "Oui"], default="Non")
    elif src in fe.ORDINAL:
        order = fe.ORDINAL[src][0]
        spec.update(kind="ordinal", choices=order, default=merged[src].mode().iloc[0])
    elif src in fe.NOMINAL or src in fe.BINARY_OTHER:
        choices = merged[src].dropna().value_counts().index.tolist()
        spec.update(kind="choice", choices=choices, default=choices[0])
    elif src in fe.MULTIHOT:
        counts = merged[src].apply(_modalities).explode().dropna().value_counts()
        spec.update(kind="multi", choices=counts.index.tolist(), default=[])
    elif src in fe.NUMERIC:
        v = merged[src]
        spec.update(kind="number", min=int(v.min()), max=int(v.max()), default=int(v.median()))
    else:
        raise ValueError(f"colonne source inconnue : {src}")
    return spec


def _typical_value(spec, rows):
    """Valeur la plus representative d'une colonne parmi les patients d'une classe."""
    s = rows[spec["source"]]
    if spec["kind"] == "yes_no":
        return "Oui" if (s == "Oui").mean() >= 0.5 else "Non"
    if spec["kind"] in ("choice", "ordinal"):
        return s.mode().iloc[0]
    if spec["kind"] == "multi":
        sets = s.apply(_modalities)
        return [m for m in spec["choices"] if sets.apply(lambda x, m=m: m in x).mean() >= 0.5]
    return int(round(s.median()))


def _softmax(z):
    z = np.asarray(z, dtype=float)
    return np.exp(z - logsumexp(z, axis=-1, keepdims=True))


class PatientClassifier:
    # ---------------------------------------------------------
    @classmethod
    def train(cls, merged, k=K_FEATURES, n_splits=5, n_repeats=3, seed=100):
        """merged : DataFrame tel que produit par preprocessing.py (colonnes d'origine uniquement)."""
        self = cls()
        merged = merged.reset_index(drop=True)
        X_all, D = fe.build_features(merged, lambda *_: None)
        y_all = merged["diagnostic_principal"].apply(dm.map_model)
        in_scope = (y_all != dm.OUT_OF_SCOPE).values
        cols = D.loc[D.source_column != "derivee", "feature"].tolist()
        X = X_all.loc[in_scope, cols].reset_index(drop=True)
        y = y_all[in_scope].reset_index(drop=True)
        mask = fs.discrete_mask(X)
        classes = sorted(set(y))
        y_idx = np.array([classes.index(v) for v in y])

        # --- predictions hors echantillon (selection refaite dans chaque pli) : calibration et fiabilite
        LJ = np.zeros((n_repeats, len(y), len(classes)))
        for r in range(n_repeats):
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=seed + r)
            for tr, te in cv.split(X, y):
                sel = fs.rank_columns(X.iloc[tr], y.iloc[tr], "combined", mask)[:k]
                m = NaiveBayes().fit(X.iloc[tr][sel], y.iloc[tr])
                LJ[r, te] = m.predict_log_joint(X.iloc[te][sel])
        flat = LJ.reshape(-1, len(classes))
        idx = np.tile(y_idx, n_repeats)

        def nll(T):
            z = flat / T
            return float(-(z[np.arange(len(z)), idx] - logsumexp(z, axis=1)).mean())

        T = float(minimize_scalar(nll, bounds=(0.5, 20.0), method="bounded").x)
        P = _softmax(flat / T)
        pmax, ok = P.max(axis=1), P.argmax(axis=1) == idx
        self.calibration = {"T": T, "log_loss_brut": nll(1.0), "log_loss_calibre": nll(T)}
        self.bands = [{"low": lo, "high": min(hi, 1.0), "n": int(s.sum()), "share": float(s.mean()),
                       "accuracy": float(ok[s].mean()) if s.any() else float("nan")}
                      for lo, hi in BANDS for s in [(pmax >= lo) & (pmax < hi)]]
        per_rep = [f1_score(y_idx, LJ[r].argmax(axis=1), average=None, labels=range(len(classes)))
                   for r in range(n_repeats)]
        f1_cls = np.mean(per_rep, axis=0)
        self.performance = {"macro_f1": float(f1_cls.mean()), "f1_par_classe": dict(zip(classes, f1_cls.round(3))),
                            "n_par_classe": y.value_counts().to_dict(), "n": len(y), "k": k, "repetitions": n_repeats}

        # --- modele final : classement sur tous les patients du perimetre, top K, Naive Bayes
        self.features = fs.rank_columns(X, y, "combined", mask)[:k]
        self.model = NaiveBayes().fit(X[self.features], y)
        self.classes = list(self.model.classes_)
        self.T = T

        # --- formulaire : une saisie par colonne source, dans l'ordre d'importance des variables
        Dd = D.set_index("feature")
        self.feature_info = {f: _feature_info(f, Dd.loc[f]) for f in self.features}
        sources = list(dict.fromkeys(i["source"] for i in self.feature_info.values()))
        self.inputs = [_input_spec(s, merged) for s in sources]
        rows = merged[in_scope].reset_index(drop=True)
        self.typical = {c: {sp["source"]: _typical_value(sp, rows[(y == c).values]) for sp in self.inputs}
                        for c in self.classes}
        return self

    # ---------------------------------------------------------
    def default_inputs(self):
        return {sp["source"]: sp["default"] for sp in self.inputs}

    def typical_inputs(self, class_name):
        """Profil type (valeurs les plus representatives) de la classe, pour pre-remplir le formulaire."""
        return dict(self.typical[class_name])

    def raw_from_record(self, record):
        """Valeurs de saisie correspondant a une ligne de merged_data.csv (utile pour verifier l'encodeur)."""
        raw = {}
        for sp in self.inputs:
            v = record[sp["source"]]
            raw[sp["source"]] = (sorted(_modalities(v)) if sp["kind"] == "multi"
                                 else float(v) if sp["kind"] == "number" else str(v))
        return raw

    def encode(self, raw):
        """Valeurs saisies -> variables encodees du modele (memes regles que feature_engineering)."""
        out = {}
        for feat, info in self.feature_info.items():
            v, mode = raw[info["source"]], info["mode"]
            if mode == "number":
                out[feat] = float(v)
            elif mode == "ordinal":
                out[feat] = info["order"].index(v)
            elif mode == "yes_no":
                out[feat] = int(v == "Oui")
            elif mode == "equals":
                out[feat] = int(v == info["modality"])
            elif mode == "has":
                out[feat] = int(info["modality"] in v)
            else:   # in_rare
                vals = v if isinstance(v, (list, tuple, set)) else [v]
                out[feat] = int(any(m in info["rare"] for m in vals))
        return pd.Series(out)[self.features]

    def _label(self, info):
        src_label, mode = source_label(info["source"]), info["mode"]
        if mode in ("number", "ordinal", "yes_no"):
            return src_label
        if mode == "in_rare":
            return f"{src_label} : autres modalités rares"
        return f"{src_label} : {pretty(info['modality'])}"

    def _shown_value(self, info, raw, x):
        if info["mode"] == "number":
            return f"{x:g}"
        if info["mode"] == "ordinal":
            return pretty(raw[info["source"]])
        return "Oui" if x == 1 else "Non"

    # ---------------------------------------------------------
    def predict(self, raw):
        """Categorie, probabilites calibrees, fiabilite et explication pour UN patient."""
        x = self.encode(raw)
        lj = self.model.predict_log_joint(x.to_frame().T)[0]
        p = _softmax(lj / self.T)
        order = np.argsort(-p)
        best, second = order[0], order[1]
        e = self.model.explain(x, top=len(self.features))
        table = e["toutes"].copy()
        table["contribution"] = table["contribution"] / self.T          # meme echelle que les probas affichees
        table["variable"] = [self._label(self.feature_info[f]) for f in table["variable"]]
        feats = [f for f in self.features]
        label_of = {self._label(self.feature_info[f]): f for f in feats}
        table["niveau"] = [self.feature_info[label_of[v]]["tier"] for v in table["variable"]]
        table["valeur_saisie"] = [self._shown_value(self.feature_info[label_of[v]], raw, x[label_of[v]])
                                  for v in table["variable"]]
        conf = float(p[best])
        band = next(b for b in self.bands if b["low"] <= conf < (b["high"] if b["high"] < 1 else 1.0001))
        return {
            "classe": self.classes[best], "deuxieme_classe": self.classes[second],
            "probas": dict(zip(self.classes, p.tolist())), "confiance": conf,
            "incertain": conf < UNCERTAIN_BELOW, "bande": band,
            "terme_prior": e["terme_prior"] / self.T, "log_rapport": e["log_rapport_total"] / self.T,
            "explication": table[["variable", "valeur_saisie", "contribution", "niveau"]].reset_index(drop=True),
        }


if __name__ == "__main__":
    clf = PatientClassifier.train(pd.read_csv(fe.DATA_PATH, encoding="utf-8"))
    print("Variables retenues :", *[f"  {i + 1}. {clf._label(clf.feature_info[f])}  [{clf.feature_info[f]['tier']}]"
                                    for i, f in enumerate(clf.features)], sep="\n")
    print("Performance (hors echantillon, k fixe) :", clf.performance)
    print("Calibration :", {a: round(b, 3) for a, b in clf.calibration.items()})
    for b in clf.bands:
        print(f"  confiance [{b['low']:.2f}, {b['high']:.2f}) : {b['share'] * 100:4.1f} % des cas, "
              f"justesse {b['accuracy']:.2f} (n={b['n']})")