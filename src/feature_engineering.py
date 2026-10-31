"""
feature_engineering.py
Etape de preparation des variables pour la modelisation.

Entree  : ../data/processed/merged_data.csv   (sortie de preprocessing.py)
Sorties : ../data/processed/model_features.csv        matrice 100 % numerique (1 ligne / patient)
          ../data/processed/feature_dictionary.csv    une ligne par variable creee (source, type, niveau, support)
          ../data/processed/feature_pool_decisions.csv  decision pour CHAQUE colonne source (incluse / exclue + raison)

Principes
---------
1. Chaque colonne de merged_data.csv doit etre classee explicitement (incluse ou
   exclue, avec la raison). Si une colonne n'est pas classee, le script s'arrete :
   aucune variable n'entre dans le pool "par accident".
2. Aucune information sur la cible n'est utilisee ici (pas de target encoding,
   pas de selection) : le fichier produit est donc sans fuite et peut etre
   joint a n'importe quel mapping diagnostique par patient_id.
3. Niveaux de variables. L'annotation n'a retenu que ce qui est ECRIT dans les
   dossiers medicaux : un element n'est annote que si le medecin l'a consigne
   (annotation centree sur le PREMIER et le DERNIER episode documentes, pas sur
   l'episode actuel ; "Non" = non mentionne, pas forcement absent). Les valeurs refletent donc ce que le clinicien a documente,
   et le dossier est redige avec le diagnostic (ou une hypothese) en tete :
     A = anamnese factuelle, recueillie de facon systematique quel que soit le
         diagnostic (age, sexe, antecedents familiaux, evenements de vie, contexte
         social...).
     B = tableau clinique documente a l'appui du diagnostic (symptomes, sommeil,
         suicidalite, antecedents psychiatriques personnels...) : le risque de
         circularite vient du dossier, pas de l'annotateur.
   Le niveau est conserve dans le dictionnaire pour comparer plus tard les
   performances avec et sans le niveau B.
4. Les modalites rares (< MIN_COUNT patients) sont regroupees ("__autre") ou
   ecartees : une colonne presque vide n'apporte que du bruit.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "merged_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

MIN_COUNT = 5                      # modalite rare : moins de 5 patients
ID_COL = "patient_id"
ABSENCE_LABELS = {"Rien À Signaler", "Rien À Mentionner", "Aucun", "Ras"}

# ------------------------------------------------------------
# 1. DECISIONS SUR LE POOL (explicites, une par colonne source)
# ------------------------------------------------------------
FOLLOWUP_REASON = "variable de suivi / resultat : posterieure au diagnostic"
EXCLUDED = {
    ID_COL: "identifiant",
    "diagnostic_principal": "variable cible",
    "symptomatologie_1er_episode": "fuite : libelles quasi identiques aux categories diagnostiques",
    "motif_premiere_consultation": "fuite : texte libre mentionnant le diagnostic",
    "lieu_residence": "texte a haute cardinalite (17 valeurs), peu informatif",
    "origine_geographique": "texte a haute cardinalite (33 valeurs), peu informatif",
}
for _c in ["etat_clinique_derniere_consult", "observance_derniere_consult", "observance_consult_reference",
           "tolerance_derniere_consult", "tolerance_consult_reference", "orientation_premiere_consult",
           "orientation_derniere_consult", "orientation_consult_reference", "rechute_12_mois",
           "rechute_scolaire_recente", "hospitalisation_12_mois", "hospitalisation_suivi",
           "nb_hospitalisations_int", "nb_consultations"]:
    EXCLUDED[_c] = FOLLOWUP_REASON

SYMPTOM_COLS = [
    "anxiete_anticipatoire", "attaques_panique_recentes", "compulsions", "obsessions",
    "symptomes_negatifs", "syndrome_anxieux", "syndrome_delirant", "syndrome_depressif",
    "syndrome_desorganisation", "syndrome_excitation_psychomotrice", "syndrome_hallucinatoire",
    "syndrome_phobique", "troubles_cognitifs",
]

# colonne -> niveau (A/B)
NUMERIC = {c: "A" for c in ["age_apparition_premiers_symptomes", "age_premiere_consultation_generale",
                            "age_premiere_consultation_service", "nombre_redoublements",
                            "rang_dans_fratrie", "taille_fratrie", "nb_tentatives_suicide"]}
YES_NO = {c: "A" for c in ["atcd_judiciaire", "atcd_tentative_suicide", "consanguinite_parents",
                           "couverture_sociale"]}
YES_NO.update({c: "B" for c in SYMPTOM_COLS})
# colonne -> (modalite codee 1, niveau)
BINARY_OTHER = {"sexe": ("Masculin", "A"), "fonctionnement_socioprofessionnel": ("Altere", "B")}
# colonne -> (ordre croissant, niveau)
ORDINAL = {
    "severite_traumatisme": (["Aucun", "Léger", "Modéré", "Sévère"], "A"),
    "niveau_socioeconomique": (["Faible", "Moyen", "Élevé"], "A"),
    "niveau_suicidalite": (["Aucune_Ideation", "Idees_Mort_Passives", "Idees_Suicidaires_Actives",
                            "Tentative_Ou_Automutilation"], "B"),
    "niveau_activite_sociale": (["Absentes", "Limitées", "Modérées", "Bonnes"], "B"),
}
NOMINAL = {"niveau_scolaire": "A", "etat_civil": "A", "milieu_residence": "A",
           "situation_professionnelle": "A", "atcd_violence": "A", "alimentation": "B",
           "sommeil": "B", "hetero_agressivite": "B", "etat_clinique_premiere_consult": "B"}
MULTIHOT = {c: "A" for c in ["atcd_abus_spa", "atcd_familiaux_mc", "atcd_familiaux_psy_membre",
                             "atcd_familiaux_psy_trouble", "atcd_personnels_mc", "situation_familiale",
                             "situation_residentielle", "type_traumatisme_emotionnel"]}
MULTIHOT["atcd_personnels_psychiatriques"] = "B"   # contient episodes (hypo)maniaques : proche du diagnostic

# ------------------------------------------------------------
# 2. OUTILS D'ENCODAGE
# ------------------------------------------------------------
dictionary_rows = []


def slug(text):
    return re.sub(r"[^0-9A-Za-zÀ-ÿ]+", "_", str(text)).strip("_")


def register(out, feature, source, kind, tier, definition):
    s = out[feature]
    n_one = int((s > 0).sum())
    dictionary_rows.append({
        "feature": feature, "source_column": source, "kind": kind, "tier": tier,
        "definition": definition, "n_non_zero": n_one,
        "low_support": bool(kind in {"binaire", "nominal", "multi-hot"} and
                            (n_one < MIN_COUNT or len(s) - n_one < MIN_COUNT)),
    })


def encode_numeric(df, col, tier, out):
    assert df[col].notna().all(), f"NaN inattendu dans {col} (relancer preprocessing.py)"
    out[col] = df[col].astype(float)
    register(out, col, col, "numerique", tier, "valeur brute")


def encode_yes_no(df, col, tier, out):
    mapped = df[col].map({"Oui": 1, "Non": 0})
    assert mapped.notna().all(), f"{col} : valeurs hors Oui/Non"
    out[col] = mapped.astype(int)
    register(out, col, col, "binaire", tier, "1 = Oui, 0 = Non")


def encode_binary_other(df, col, positive, tier, out):
    assert df[col].notna().all(), f"NaN inattendu dans {col}"
    name = f"{col}__{slug(positive)}"
    out[name] = (df[col] == positive).astype(int)
    register(out, name, col, "binaire", tier, f"1 = {positive}")


def encode_ordinal(df, col, order, tier, out):
    mapped = df[col].map({v: i for i, v in enumerate(order)})
    assert mapped.notna().all(), f"{col} : modalite hors de l'ordre defini {set(df[col]) - set(order)}"
    out[col] = mapped.astype(int)
    register(out, col, col, "ordinal", tier, "0.." + str(len(order) - 1) + " = " + " < ".join(order))


def _add_indicator_columns(out, col, tier, kind, membership, index):
    """membership : dict modalite -> Series booleenne. Regroupe les modalites rares."""
    rare_union = pd.Series(False, index=index)
    rare_names = []
    for modality, present in membership.items():
        if int(present.sum()) >= MIN_COUNT:
            name = f"{col}__{slug(modality)}"
            out[name] = present.astype(int)
            register(out, name, col, kind, tier, f"1 = {modality}")
        else:
            rare_union |= present
            rare_names.append(str(modality))
    if int(rare_union.sum()) >= MIN_COUNT:
        name = f"{col}__autre"
        out[name] = rare_union.astype(int)
        # la liste des modalites regroupees est conservee : necessaire pour encoder un patient isole
        register(out, name, col, kind, tier,
                 "1 = modalites rares regroupees (< %d patients chacune) : %s" % (MIN_COUNT, " | ".join(sorted(rare_names))))


def encode_nominal(df, col, tier, out):
    assert df[col].notna().all(), f"NaN inattendu dans {col}"
    membership = {m: df[col] == m for m in df[col].unique()}
    _add_indicator_columns(out, col, tier, "nominal", membership, df.index)


def encode_multihot(df, col, tier, out):
    sets = df[col].apply(lambda v: set() if pd.isna(v) else {p.strip() for p in str(v).split(";")}
                         - ABSENCE_LABELS - {""})
    modalities = sorted(set().union(*sets))
    membership = {m: sets.apply(lambda s, m=m: m in s) for m in modalities}
    _add_indicator_columns(out, col, tier, "multi-hot", membership, df.index)
    return sets


# ------------------------------------------------------------
# 3. VARIABLES DERIVEES (a partir de variables deja presentes, aucune info nouvelle)
# ------------------------------------------------------------
def add_derived(df, out, sets_by_col, log):
    def add(name, series, tier, definition):
        out[name] = series
        register(out, name, "derivee", "derivee", tier, definition)

    delai = df["age_premiere_consultation_generale"] - df["age_apparition_premiers_symptomes"]
    log(f"delai_apparition_consultation : {int((delai < 0).sum())} valeur(s) negative(s) ramenee(s) a 0")
    add("delai_apparition_consultation", delai.clip(lower=0).astype(float), "A",
        "age 1re consultation generale - age d'apparition (borne a 0)")
    delai2 = df["age_premiere_consultation_service"] - df["age_premiere_consultation_generale"]
    log(f"delai_generale_service : {int((delai2 < 0).sum())} valeur(s) negative(s) ramenee(s) a 0")
    add("delai_generale_service", delai2.clip(lower=0).astype(float), "A",
        "age 1re consultation service - age 1re consultation generale (borne a 0)")

    yn = lambda c: (df[c] == "Oui").astype(int)
    add("nb_symptomes_total", sum(yn(c) for c in SYMPTOM_COLS), "B", "nombre de symptomes/syndromes presents (sur 13)")
    add("score_anxieux", sum(yn(c) for c in ["anxiete_anticipatoire", "attaques_panique_recentes",
                                            "syndrome_anxieux", "syndrome_phobique"]), "B",
        "anxiete_anticipatoire + attaques_panique_recentes + syndrome_anxieux + syndrome_phobique")
    add("score_psychotique", sum(yn(c) for c in ["syndrome_delirant", "syndrome_hallucinatoire",
                                                "syndrome_desorganisation", "symptomes_negatifs"]), "B",
        "syndrome_delirant + syndrome_hallucinatoire + syndrome_desorganisation + symptomes_negatifs")
    add("score_obsessionnel", yn("obsessions") + yn("compulsions"), "B", "obsessions + compulsions")

    for name, col, tier, label in [
        ("nb_types_traumatisme", "type_traumatisme_emotionnel", "A", "nombre de types de traumatisme"),
        ("nb_substances", "atcd_abus_spa", "A", "nombre de substances (hors 'Inconnu')"),
        ("nb_atcd_familiaux_psy", "atcd_familiaux_psy_trouble", "A", "nombre de troubles psychiatriques familiaux"),
        ("nb_membres_famille_atteints", "atcd_familiaux_psy_membre", "A", "nombre de categories de proches atteints"),
        ("nb_atcd_personnels_psy", "atcd_personnels_psychiatriques", "B", "nombre d'antecedents psychiatriques personnels"),
    ]:
        s = sets_by_col[col]
        if col == "atcd_abus_spa":
            s = s.apply(lambda x: x - {"Inconnu"})
        add(name, s.apply(len).astype(int), tier, label)

    taille_eff = np.maximum(np.maximum(df["taille_fratrie"], df["rang_dans_fratrie"]), 1)
    add("rang_relatif_fratrie", (df["rang_dans_fratrie"] / taille_eff).astype(float), "A",
        "rang / max(taille, rang, 1) : 1 = benjamin(e), valeurs faibles = aine(e)")
    add("est_enfant_unique", (df["taille_fratrie"] == 1).astype(int), "A", "1 = taille_fratrie vaut 1")


# ------------------------------------------------------------
# 4. CONSTRUCTION
# ------------------------------------------------------------
def build_features(df, log=print):
    included = (set(NUMERIC) | set(YES_NO) | set(BINARY_OTHER) | set(ORDINAL) | set(NOMINAL) | set(MULTIHOT))
    unclassified = [c for c in df.columns if c not in included and c not in EXCLUDED]
    if unclassified:
        raise ValueError(f"Colonnes non classees (a inclure ou exclure explicitement) : {unclassified}")
    missing_cols = [c for c in included if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes attendues absentes : {missing_cols}")

    dictionary_rows.clear()
    out = {}
    sets_by_col = {}
    for col, tier in NUMERIC.items():
        encode_numeric(df, col, tier, out)
    for col, tier in YES_NO.items():
        encode_yes_no(df, col, tier, out)
    for col, (pos, tier) in BINARY_OTHER.items():
        encode_binary_other(df, col, pos, tier, out)
    for col, (order, tier) in ORDINAL.items():
        encode_ordinal(df, col, order, tier, out)
    for col, tier in NOMINAL.items():
        encode_nominal(df, col, tier, out)
    for col, tier in MULTIHOT.items():
        sets_by_col[col] = encode_multihot(df, col, tier, out)
    add_derived(df, out, sets_by_col, log)

    out = pd.DataFrame(out, index=df.index)
    assert out.notna().all().all(), "NaN dans la matrice finale"
    assert (out.dtypes != object).all(), "colonne non numerique dans la matrice finale"
    assert (out.min() >= 0).all(), "valeur negative (chi2 exige des valeurs >= 0)"
    return out, pd.DataFrame(dictionary_rows)


def pool_decisions():
    rows = []
    for c, r in EXCLUDED.items():
        rows.append({"source_column": c, "decision": "exclue", "niveau": "", "traitement": r})
    groups = [(NUMERIC, "numerique (valeur brute)"), (YES_NO, "binaire Oui/Non -> 0/1"),
              (NOMINAL, "nominal -> one-hot"), (MULTIHOT, "multi-valeurs -> multi-hot")]
    for mapping, label in groups:
        for c, t in mapping.items():
            rows.append({"source_column": c, "decision": "incluse", "niveau": t, "traitement": label})
    for c, (pos, t) in BINARY_OTHER.items():
        rows.append({"source_column": c, "decision": "incluse", "niveau": t, "traitement": f"binaire (1 = {pos})"})
    for c, (order, t) in ORDINAL.items():
        rows.append({"source_column": c, "decision": "incluse", "niveau": t, "traitement": "ordinal : " + " < ".join(order)})
    return pd.DataFrame(rows).sort_values(["decision", "niveau", "source_column"]).reset_index(drop=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} introuvable. Lancez d'abord preprocessing.py.")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    assert df[ID_COL].is_unique, "patient_id non unique"
    features, dictionary = build_features(df)

    features = pd.concat([df[[ID_COL]], features], axis=1)
    features.to_csv(OUTPUT_DIR / "model_features.csv", index=False, encoding="utf-8")
    dictionary.to_csv(OUTPUT_DIR / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    pool_decisions().to_csv(OUTPUT_DIR / "feature_pool_decisions.csv", index=False, encoding="utf-8-sig")

    print(f"\nMatrice : {features.shape[0]} patients x {features.shape[1] - 1} variables")
    print(dictionary.groupby(["tier", "kind"]).size().unstack(fill_value=0).to_string())
    print(f"Variables a faible support (< {MIN_COUNT} cas dans une modalite) : {int(dictionary.low_support.sum())}")
    print(f"Fichiers ecrits dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()