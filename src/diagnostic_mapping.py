"""
diagnostic_mapping.py
SOURCE UNIQUE des regles qui transforment `diagnostic_principal` en categorie.

Deux mappings, volontairement differents (valides avec la residente) :

  map_eda(raw)    7 categories cliniques + "Autre"   -> analyse exploratoire, dashboard
  map_model(raw)  3 classes + "Autre"                -> modeles de classification

Chaque diagnostic composite (plusieurs diagnostics separes par ';') est
resolu par des regles de priorite. `explain(raw)` donne la categorie ET la
regle appliquee ; `python diagnostic_mapping.py` ecrit `mapping_audit.csv`
(chaque combinaison de diagnostics -> classe + regle) pour relecture clinique.

------------------------------------------------------------------
MAPPING EDA (7 categories + Autre)
  1. Trouble Psychotique        l'emporte sur tout (inclut personnalites du cluster A)
  2. Troubles Bipolaires        l'emporte sur tout le reste
  3. TOC                        l'emporte sur anxiete / depression / personnalite
  4. Troubles De La Personnalite domine par Troubles Anxieux ou Trouble Depressif
  5. Trouble De L'Adaptation    categorie propre si isole ; ignore si combine
  6. Anxiete + Depression       sans element tranchant -> Autre (vraie comorbidite)

MAPPING MODELE (3 classes + Autre)
  1. Troubles psychotiques                 l'emporte sur tout
  2. Troubles de l'humeur (bipolaire)      l'emporte ensuite
  3. TOC (classe anxiete)                  seule exception : l'emporte sur la depression
  4. Troubles de l'humeur (depression)     l'emporte sur les autres troubles anxieux
  5. Troubles anxieux / lies au stress     (inclut TOC, adaptation isole, agoraphobie)
  - Trouble De L'Adaptation combine : ignore (l'autre diagnostic decide)
  - Troubles de la personnalite : PAS de classe (effectif trop faible une fois les
    comorbidites absorbees) -> ignores dans les composites, "Autre" s'ils sont seuls
  - Tout diagnostic hors de ces listes (TDAH, autisme, trouble somatique...) est
    "hors perimetre" : ignore dans les composites, "Autre" s'il est seul.
------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "merged_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUT_OF_SCOPE = "Autre"

# ------------------------------------------------------------
# Libelles de diagnostic, tels qu'ils apparaissent dans merged_data.csv
# ------------------------------------------------------------
ANXIETE = [
    "Trouble Anxieux Non Caractérisé", "État De Stress Aigu (Esa)", "Anxiété De Performance",
    "Phobies Spécifiques", "État De Stress Post-Traumatique (Tspt)", "Trouble Anxieux Généralisé",
    "Trouble Panique", "Phobie Sociale", "Agoraphobie",
]
DEPRESSION = [
    "Trouble Dépressif Caractérisé", "Trouble Dépressif Récurrent", "Trouble Dépressif Persistant",
    "Depression Réactionnelle",
]
BIPOLAIRE = ["Trouble Bipolaire Type I", "Trouble Bipolaire Type Ii", "Cyclothymie"]
PSYCHOTIQUE_STRICT = [
    "Trouble Délirant", "Trouble Psychotique Aigu Et Transitoire", "Trouble Schizo-Affectif", "Schizophrénie",
]
PERSONNALITE_CLUSTER_A = [
    "Trouble De La Personnalité Schizotypique", "Trouble De La Personnalité Paranoïaque",
    "Trouble De La Personnalité Schizoide",
]
PERSONNALITE = [   # hors cluster A ("histrionique" s'ecrit "Hystérique" dans les donnees)
    "Trouble De La Personnalité Borderline", "Trouble De La Personnalité Hystérique",
    "Trouble De La Personnalité Évitante", "Trouble De La Personnalité Obsessionelle-Compulsive",
]
TOC_LIST = ["Trouble Obsessionnel-Compulsif (Toc)"]
ADAPTATION = "Trouble De L'Adaptation"

PSYCHOTIQUE = PSYCHOTIQUE_STRICT + PERSONNALITE_CLUSTER_A   # definition du mapping EDA

# ------------------------------------------------------------
# MAPPING EDA : 7 categories + Autre
# ------------------------------------------------------------
DIAG_MAP = {}
for _labels, _cat in [
    (ANXIETE, "Troubles Anxieux"), (DEPRESSION, "Trouble Dépressif"), (PSYCHOTIQUE, "Trouble Psychotique"),
    (BIPOLAIRE, "Troubles Bipolaires"), (TOC_LIST, "TOC"), (PERSONNALITE, "Troubles De La Personnalité"),
]:
    for _label in _labels:
        DIAG_MAP[_label] = _cat

# Ordre d'affichage standard (par effectif decroissant)
DIAG_ORDER = [
    "Troubles Anxieux", "Troubles Bipolaires", "Trouble Dépressif", "Trouble Psychotique",
    "Trouble De L'Adaptation", "TOC", "Troubles De La Personnalité", "Autre",
]


def _split(raw):
    return [p.strip() for p in str(raw).split(";")]


def _eda(raw):
    if pd.isna(raw):
        return OUT_OF_SCOPE, "valeur manquante"
    parts = _split(raw)
    if parts == [ADAPTATION]:
        return "Trouble De L'Adaptation", "adaptation isole -> categorie propre"
    if any(p in PSYCHOTIQUE for p in parts):
        return "Trouble Psychotique", "priorite 1 : psychotique"
    if any(p in BIPOLAIRE for p in parts):
        return "Troubles Bipolaires", "priorite 2 : bipolaire"
    if any(p in TOC_LIST for p in parts):
        return "TOC", "priorite 3 : TOC"
    cats = {DIAG_MAP[p] for p in parts if p != ADAPTATION and p in DIAG_MAP}
    rule = "une seule categorie reconnue"
    if "Troubles De La Personnalité" in cats and len(cats) > 1:
        cats.discard("Troubles De La Personnalité")
        rule = "personnalite dominee par anxiete/depression"
    if len(cats) == 1:
        return cats.pop(), rule
    if len(cats) > 1:
        return OUT_OF_SCOPE, "comorbidite entre categories (anxiete + depression) -> Autre"
    return OUT_OF_SCOPE, "aucun diagnostic reconnu"


def map_eda(raw):
    """Categorie du mapping EDA (7 categories + 'Autre')."""
    return _eda(raw)[0]


# ------------------------------------------------------------
# MAPPING MODELE : 3 classes + Autre
# ------------------------------------------------------------
CLASS_ANXIETE = "Troubles anxieux / liés au stress"
CLASS_HUMEUR = "Troubles de l'humeur"
CLASS_PSY = "Troubles psychotiques"
MODEL_CLASSES = [CLASS_ANXIETE, CLASS_HUMEUR, CLASS_PSY]


def _model(raw):
    if pd.isna(raw):
        return OUT_OF_SCOPE, "valeur manquante"
    parts = _split(raw)
    if parts == [ADAPTATION]:
        return CLASS_ANXIETE, "adaptation isole -> anxiete / stress"
    known = set(parts) - {ADAPTATION}          # adaptation combine : ignore
    if known & set(PSYCHOTIQUE_STRICT):
        return CLASS_PSY, "priorite 1 : psychotique"
    if known & set(BIPOLAIRE):
        return CLASS_HUMEUR, "priorite 2 : bipolaire"
    if known & set(TOC_LIST):
        return CLASS_ANXIETE, "priorite 3 : TOC l'emporte sur la depression"
    if known & set(DEPRESSION):
        return CLASS_HUMEUR, "priorite 4 : depression l'emporte sur l'anxiete"
    if known & set(ANXIETE):
        return CLASS_ANXIETE, "priorite 5 : anxiete"
    return OUT_OF_SCOPE, "hors perimetre (aucun diagnostic des 3 classes)"


def map_model(raw):
    """Classe du mapping modele (3 classes + 'Autre' = hors perimetre)."""
    return _model(raw)[0]


# ------------------------------------------------------------
# Audit
# ------------------------------------------------------------
def explain(raw):
    """Les deux classes et la regle appliquee pour une valeur de diagnostic_principal."""
    c_eda, r_eda = _eda(raw)
    c_mod, r_mod = _model(raw)
    return {"classe_eda": c_eda, "regle_eda": r_eda, "classe_modele": c_mod, "regle_modele": r_mod}


def audit_table(diag):
    """Une ligne par combinaison de diagnostics distincte, avec effectif, classes et regles."""
    rows = []
    for raw, n in diag.value_counts(dropna=False).items():
        rows.append({"diagnostic_principal": "" if pd.isna(raw) else raw, "n_patients": int(n), **explain(raw)})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    diag = df["diagnostic_principal"]
    print(f"{len(df)} patients\n")
    print("MAPPING EDA :")
    print(diag.apply(map_eda).value_counts().reindex(DIAG_ORDER, fill_value=0).to_string())
    print("\nMAPPING MODELE :")
    print(diag.apply(map_model).value_counts().reindex(MODEL_CLASSES + [OUT_OF_SCOPE], fill_value=0).to_string())
    audit = audit_table(diag)
    audit.to_csv(OUTPUT_DIR / "mapping_audit.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(audit)} combinaisons de diagnostics -> {OUTPUT_DIR / 'mapping_audit.csv'}")


if __name__ == "__main__":
    main()