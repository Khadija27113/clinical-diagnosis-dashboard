"""
profiling_model.py
Modele generatif de profils patients par categorie diagnostique.

A partir de merged_data.csv (sortie de preprocessing.py), ce script :
  1. Regroupe diagnostic_principal en 3 categories simplifiees
     (Psychotique, Bipolaire, Anxio-Depressif), comme dans exploratory_analysis.py
  2. Apprend, pour chaque categorie, la distribution conditionnelle de :
       - l'age (age d'apparition des 1ers symptomes, age 1ere consultation)
       - le sexe
       - le niveau scolaire
       - 13 symptomes/syndromes binaires (Oui/Non)
  3. Permet de generer ("echantillonner") un ou plusieurs profils patients
     synthetiques plausibles pour une categorie donnee.

Approche retenue : modele generatif de type "Naive Bayes" (les variables
sont modelisees independamment, conditionnellement a la classe) :
    - age            -> loi normale (moyenne/ecart-type par classe)
    - sexe / scolarite -> loi categorielle empirique (lissage de Laplace)
    - symptomes      -> loi de Bernoulli par symptome (lissage de Laplace)

Ce choix est volontairement simple et interpretable : avec ~250 patients
au total et des classes desequilibrees (Psychotique n=10, Bipolaire n=23,
Anxio-Depressif n=102), un modele plus complexe (GAN, VAE...) surapprendrait
et produirait des profils peu fiables. Le lissage vers la moyenne/frequence
globale (parametre alpha / age_shrink_k) evite que les classes a faible
effectif (surtout Psychotique) ne generent des profils instables ou des
probabilites de 0%/100% artificielles.

AVERTISSEMENT : avec seulement 10 patients "Psychotique" dans les donnees,
les profils generes pour cette classe doivent etre interpretes avec beaucoup
de prudence (grande incertitude statistique).

Usage (depuis DataAnalyseProject/src) :
    python profiling_model.py --categorie Psychotique --n 3
    python profiling_model.py --categorie Bipolaire --mode describe
    python profiling_model.py --categorie "Anxio-Depressif" --n 5 --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "merged_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "diagnostic_profile_model.json"

AGE_ONSET_COL = "age_apparition_premiers_symptomes"
AGE_CONSULT_COL = "age_premiere_consultation_generale"
NUMERIC_COLS = [AGE_ONSET_COL, AGE_CONSULT_COL]

SEXE_COL = "sexe"
EDU_COL = "niveau_scolaire"
CATEGORICAL_COLS = [SEXE_COL, EDU_COL]

SYMPTOM_COLS = [
    "anxiete_anticipatoire", "attaques_panique_recentes", "compulsions",
    "obsessions", "symptomes_negatifs", "syndrome_anxieux",
    "syndrome_delirant", "syndrome_depressif", "syndrome_desorganisation",
    "syndrome_excitation_psychomotrice", "syndrome_hallucinatoire",
    "syndrome_phobique", "troubles_cognitifs",
]

DIAG_MAP = {
    "Schizophrénie": "Psychotique",
    "Trouble Schizo-Affectif": "Psychotique",
    "Trouble Psychotique Aigu Et Transitoire": "Psychotique",
    "Trouble Bipolaire Type I": "Bipolaire",
    "Trouble Bipolaire Type Ii": "Bipolaire",
    "Cyclothymie": "Bipolaire",
    "Trouble Dépressif Caractérisé": "Anxio-Dépressif",
    "Trouble Dépressif Récurrent": "Anxio-Dépressif",
    "Trouble Dépressif Persistant": "Anxio-Dépressif",
    "Trouble Anxieux Généralisé": "Anxio-Dépressif",
    "Trouble Panique": "Anxio-Dépressif",
    "Phobie Sociale": "Anxio-Dépressif",
    "Trouble Obsessionnel-Compulsif (Toc)": "Anxio-Dépressif",
    "État De Stress Post-Traumatique (Tspt)": "Anxio-Dépressif",
    "Trouble De L'Adaptation": "Anxio-Dépressif",
}


# ------------------------------------------------------------
# Modele
# ------------------------------------------------------------
class DiagnosticProfileModel:
    """Modele generatif Naive-Bayes conditionne sur la categorie diagnostique."""

    def __init__(self, alpha=2.0, age_shrink_k=5):
        self.alpha = alpha              # lissage de Laplace (categoriel / binaire)
        self.age_shrink_k = age_shrink_k  # force du lissage vers la moyenne globale (age)
        self.fitted = False

    def fit(self, df: pd.DataFrame, diag_col: str = "diag_cat"):
        df = df.copy()
        self.classes_ = sorted(c for c in df[diag_col].dropna().unique() if c != "Autre")
        self.class_counts_ = {
            c: int((df[diag_col] == c).sum()) for c in self.classes_
        }
        self.class_stats_ = {c: {} for c in self.classes_}
        self.global_stats_ = {}

        # --- variables numeriques (age) ---
        for col in NUMERIC_COLS:
            g_mean, g_std = df[col].mean(), df[col].std()
            self.global_stats_[col] = {"mean": g_mean, "std": g_std}
            for c in self.classes_:
                sub = df.loc[df[diag_col] == c, col].dropna()
                n = len(sub)
                m_c = sub.mean() if n else g_mean
                s_c = sub.std() if n > 1 else g_std
                w = n / (n + self.age_shrink_k)  # lissage vers la moyenne globale
                mean_c = w * m_c + (1 - w) * g_mean
                std_c = w * (s_c if pd.notna(s_c) else g_std) + (1 - w) * g_std
                self.class_stats_[c][col] = {"mean": float(mean_c), "std": float(max(std_c, 0.5))}

        # --- variables categorielles (sexe, niveau scolaire) ---
        for col in CATEGORICAL_COLS:
            global_counts = df[col].value_counts(dropna=True)
            categories = global_counts.index.tolist()
            global_probs = (global_counts / global_counts.sum()).to_dict()
            self.global_stats_[col] = global_probs
            for c in self.classes_:
                sub_counts = df.loc[df[diag_col] == c, col].value_counts(dropna=True)
                n = sub_counts.sum()
                probs = {}
                for cat in categories:
                    count = sub_counts.get(cat, 0)
                    probs[cat] = (count + self.alpha * global_probs[cat]) / (n + self.alpha)
                self.class_stats_[c][col] = probs

        # --- symptomes binaires (Oui/Non) ---
        for col in SYMPTOM_COLS:
            g_prob = (df[col] == "Oui").mean()
            self.global_stats_[col] = float(g_prob)
            for c in self.classes_:
                sub = df.loc[df[diag_col] == c, col]
                n = sub.notna().sum()
                pos = (sub == "Oui").sum()
                prob = (pos + self.alpha * g_prob) / (n + self.alpha)
                self.class_stats_[c][col] = float(prob)

        self.fitted = True
        return self

    # ---------------------------------------------------
    def _check_category(self, category):
        if not self.fitted:
            raise RuntimeError("Le modele doit d'abord etre entraine avec .fit(df).")
        if category not in self.classes_:
            raise ValueError(
                f"Categorie inconnue '{category}'. Categories disponibles : {self.classes_}"
            )

    def sample(self, category: str, n: int = 1, seed: int = None) -> pd.DataFrame:
        """Genere n profils patients synthetiques pour la categorie donnee."""
        self._check_category(category)
        rng = np.random.default_rng(seed)
        stats = self.class_stats_[category]
        rows = []
        for _ in range(n):
            row = {"diag_cat": category}
            for col in NUMERIC_COLS:
                s = stats[col]
                val = rng.normal(s["mean"], s["std"])
                row[col] = max(0, round(val))
            for col in CATEGORICAL_COLS:
                probs = stats[col]
                cats = list(probs.keys())
                p = np.array(list(probs.values()))
                p = p / p.sum()
                row[col] = rng.choice(cats, p=p)
            for col in SYMPTOM_COLS:
                row[col] = "Oui" if rng.random() < stats[col] else "Non"
            rows.append(row)
        cols_order = ["diag_cat"] + NUMERIC_COLS + CATEGORICAL_COLS + SYMPTOM_COLS
        return pd.DataFrame(rows)[cols_order]

    def describe(self, category: str) -> str:
        """Retourne une description textuelle du profil 'typique' de la categorie
        (valeurs les plus probables/moyennes, pas un tirage aleatoire)."""
        self._check_category(category)
        stats = self.class_stats_[category]
        n = self.class_counts_[category]

        age_onset = round(stats[AGE_ONSET_COL]["mean"])
        age_consult = round(stats[AGE_CONSULT_COL]["mean"])

        sexe_probs = stats[SEXE_COL]
        sexe_top = max(sexe_probs, key=sexe_probs.get)
        edu_probs = stats[EDU_COL]
        edu_top = max(edu_probs, key=edu_probs.get)

        symptomes_tries = sorted(
            ((col, stats[col]) for col in SYMPTOM_COLS), key=lambda x: -x[1]
        )
        symptomes_frequents = [
            (col.replace("_", " "), p) for col, p in symptomes_tries if p >= 0.5
        ]

        lines = []
        lines.append(f"Profil type — categorie « {category} » (n={n} patients dans les donnees) :")
        if n < 20:
            lines.append(
                f"  ATTENTION : effectif faible (n={n}), ce profil est peu fiable statistiquement."
            )
        lines.append(f"  - Sexe le plus frequent : {sexe_top} ({sexe_probs[sexe_top]*100:.0f}%)")
        lines.append(f"  - Age moyen d'apparition des symptomes : ~{age_onset} ans")
        lines.append(f"  - Age moyen a la 1ere consultation : ~{age_consult} ans")
        lines.append(f"  - Niveau scolaire le plus frequent : {edu_top} ({edu_probs[edu_top]*100:.0f}%)")
        if symptomes_frequents:
            lines.append("  - Symptomes/syndromes presents chez plus de 50% des patients :")
            for name, p in symptomes_frequents:
                lines.append(f"      * {name} ({p*100:.0f}%)")
        else:
            lines.append("  - Aucun symptome present chez plus de 50% des patients de cette categorie.")
        return "\n".join(lines)

    # ---------------------------------------------------
    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "age_shrink_k": self.age_shrink_k,
            "classes_": self.classes_,
            "class_counts_": self.class_counts_,
            "class_stats_": self.class_stats_,
            "global_stats_": self.global_stats_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Modele sauvegarde : {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(alpha=payload["alpha"], age_shrink_k=payload["age_shrink_k"])
        model.classes_ = payload["classes_"]
        model.class_counts_ = payload["class_counts_"]
        model.class_stats_ = payload["class_stats_"]
        model.global_stats_ = payload["global_stats_"]
        model.fitted = True
        return model


# ------------------------------------------------------------
# Entrainement a partir des donnees du projet
# ------------------------------------------------------------
def load_and_prepare_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier {DATA_PATH} introuvable. Lancez d'abord preprocessing.py."
        )
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["diag_cat"] = df["diagnostic_principal"].map(DIAG_MAP).fillna("Autre")
    return df


def train_model() -> DiagnosticProfileModel:
    df = load_and_prepare_data()
    model = DiagnosticProfileModel().fit(df)
    return model


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Modele generatif de profils patients par categorie diagnostique.")
    parser.add_argument("--categorie", required=True, choices=["Psychotique", "Bipolaire", "Anxio-Dépressif"],
                         help="Categorie diagnostique cible.")
    parser.add_argument("--mode", choices=["generate", "describe"], default="generate",
                         help="'generate' = tire des profils synthetiques aleatoires ; "
                              "'describe' = decrit le profil type (valeurs moyennes/majoritaires).")
    parser.add_argument("--n", type=int, default=1, help="Nombre de profils a generer (mode 'generate').")
    parser.add_argument("--seed", type=int, default=None, help="Graine aleatoire pour reproductibilite.")
    args = parser.parse_args()

    model = train_model()

    if args.mode == "describe":
        print(model.describe(args.categorie))
    else:
        profiles = model.sample(args.categorie, n=args.n, seed=args.seed)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(profiles.to_string(index=False))


if __name__ == "__main__":
    main()