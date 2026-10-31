"""
test_pipeline.py
Tests automatiques du pipeline : prétraitement -> feature engineering -> mapping.

Lancement (depuis DataAnalyseProject/src) :
    python test_pipeline.py          # lanceur intégré, aucune dépendance
    pytest -q test_pipeline.py       # si pytest est installé

Quatre familles de tests :
  1. UNITAIRES   : chaque encodeur sur de petits jeux synthétiques (cas limites, erreurs attendues)
  2. CONTRAT     : merged_data.csv (sortie de preprocessing.py) respecte les règles promises
  3. COHÉRENCE   : model_features.csv est recompté de façon indépendante à partir des données brutes
  4. MAPPING     : règles de priorité des deux mappings (EDA 7 catégories, modèle 3 classes), leur cohérence,
                   et vérification qu'il n'existe qu'une seule copie de la logique (diagnostic_mapping.py)

À relancer après toute modification d'un script ou après un nouveau lot de données :
un test qui échoue indique exactement quelle promesse n'est plus tenue.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import feature_engineering as fe  # noqa: E402

DATA = fe.DATA_PATH
FEATURES_CSV = fe.OUTPUT_DIR / "model_features.csv"
_quiet = lambda *_: None


def load_raw():
    assert DATA.exists(), f"{DATA} introuvable : lancez d'abord preprocessing.py"
    return pd.read_csv(DATA, encoding="utf-8")


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def fresh():
    fe.dictionary_rows.clear()
    return {}


_cache = {}


def built():
    if "b" not in _cache:
        df = load_raw()
        X, D = fe.build_features(df, _quiet)
        _cache["b"] = (df, X, D)
    return _cache["b"]


def modalities(series):
    """Parsing indépendant d'une colonne multi-valeurs : patient -> ensemble de modalités."""
    return series.apply(lambda v: set() if pd.isna(v) else {p.strip() for p in str(v).split(";")})


# ============================================================
# 1. TESTS UNITAIRES DES ENCODEURS (données synthétiques)
# ============================================================
def test_slug_nettoie_les_noms_de_colonnes():
    assert fe.slug("Perte D'Un Proche") == "Perte_D_Un_Proche"
    assert fe.slug("  Violence Sexuelle/Abus ") == "Violence_Sexuelle_Abus"
    assert fe.slug("Échec Scolaire") == "Échec_Scolaire"


def test_encode_yes_no():
    out = fresh()
    fe.encode_yes_no(pd.DataFrame({"x": ["Oui", "Non", "Oui"]}), "x", "B", out)
    assert out["x"].tolist() == [1, 0, 1]
    assert raises(AssertionError, fe.encode_yes_no, pd.DataFrame({"x": ["Oui", None]}), "x", "B", {}), \
        "un NaN doit etre refuse"
    assert raises(AssertionError, fe.encode_yes_no, pd.DataFrame({"x": ["Oui", "Peut-etre"]}), "x", "B", {}), \
        "une valeur hors Oui/Non doit etre refusee"


def test_encode_binary_other_cree_une_seule_colonne():
    out = fresh()
    df = pd.DataFrame({"sexe": ["Féminin", "Masculin", "Féminin"]})
    fe.encode_binary_other(df, "sexe", "Masculin", "A", out)
    assert list(out) == ["sexe__Masculin"]
    assert out["sexe__Masculin"].tolist() == [0, 1, 0]


def test_encode_ordinal_respecte_l_ordre_et_refuse_l_inconnu():
    out = fresh()
    order = ["Aucun", "Léger", "Modéré", "Sévère"]
    fe.encode_ordinal(pd.DataFrame({"s": ["Sévère", "Aucun", "Léger", "Modéré"]}), "s", order, "A", out)
    assert out["s"].tolist() == [3, 0, 1, 2]
    assert raises(AssertionError, fe.encode_ordinal, pd.DataFrame({"s": ["Inconnu"]}), "s", order, "A", {})


def test_encode_nominal_one_hot_et_modalites_rares():
    # c(3) + d(2) = 5 patients rares -> regroupes dans __autre
    df = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 6 + ["c"] * 3 + ["d"] * 2})
    out = fresh()
    fe.encode_nominal(df, "c", "A", out)
    assert set(out) == {"c__a", "c__b", "c__autre"}
    assert out["c__autre"].sum() == 5
    assert (sum(out.values()) == 1).all(), "one-hot : exactement un 1 par ligne"
    # c(2) + d(1) = 3 < 5 -> pas de colonne __autre, ces patients n'ont que des 0
    df2 = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 6 + ["c"] * 2 + ["d"] * 1})
    out2 = fresh()
    fe.encode_nominal(df2, "c", "A", out2)
    assert set(out2) == {"c__a", "c__b"}
    assert sum(out2.values()).sum() == 16


def test_encode_multihot_absences_nan_et_espaces():
    df = pd.DataFrame({"t": ["A;B", "Aucun", "A", " B ; C", "Rien À Signaler", None] * 3})
    out = fresh()
    sets = fe.encode_multihot(df, "t", "A", out)
    assert set(out) == {"t__A", "t__B"}, "C (3 patients) est rare et seul : ecarte"
    assert out["t__A"].sum() == 6 and out["t__B"].sum() == 6
    assert out["t__A"][0] == 1 and out["t__B"][0] == 1, "multi-hot : plusieurs 1 possibles"
    for i in (1, 4, 5):   # Aucun, Rien À Signaler, NaN
        assert out["t__A"][i] == 0 and out["t__B"][i] == 0
        assert sets[i] == set()
    assert sets[3] == {"B", "C"}, "les espaces autour des modalites doivent etre retires"


def test_build_features_refuse_une_colonne_non_classee():
    df = load_raw()
    df["colonne_surprise"] = 1
    assert raises(ValueError, fe.build_features, df, _quiet)


# ============================================================
# 2. CONTRAT DE DONNÉES : sortie de preprocessing.py
# ============================================================
def test_data_patient_id_unique():
    df = load_raw()
    assert len(df) > 0 and df["patient_id"].is_unique


def test_data_symptomes_complets_et_oui_non():
    df = load_raw()
    for c in fe.SYMPTOM_COLS:
        assert df[c].notna().all(), f"{c} contient des NaN"
        assert set(df[c].unique()) <= {"Oui", "Non"}, f"{c} : valeurs hors Oui/Non"


def test_data_nan_restants_seulement_dans_colonnes_autorisees():
    df = load_raw()
    allowed = set(fe.EXCLUDED) | set(fe.MULTIHOT)   # suivi/resultats, texte libre, multi-valeurs
    bad = [c for c in df.columns[df.isna().any()] if c not in allowed]
    assert not bad, f"NaN inattendus (non traites par preprocessing.py) dans : {bad}"


def test_data_absences_structurelles_coherentes():
    df = load_raw()
    assert ((df["nb_hospitalisations_int"] == 0) == (df["hospitalisation_suivi"] == "Non")).all()
    assert ((df["nb_tentatives_suicide"] == 0) == (df["atcd_tentative_suicide"] == "Non")).all()
    assert df["severite_traumatisme"].notna().all()
    assert (df.loc[df["severite_traumatisme"] == "Aucun", "type_traumatisme_emotionnel"] == "Aucun").all()


def test_data_ages_et_compteurs_entiers_positifs():
    df = load_raw()
    for c in fe.NUMERIC:
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c} n'est pas numerique"
        assert (df[c] >= 0).all(), f"{c} contient des valeurs negatives"
        assert (df[c] % 1 == 0).all(), f"{c} contient des valeurs non entieres"
    for c in ["age_apparition_premiers_symptomes", "age_premiere_consultation_generale",
              "age_premiere_consultation_service"]:
        assert df[c].max() <= 70, f"{c} : age improbable"


def test_data_domaines_des_variables_cles():
    df = load_raw()
    assert set(df["sexe"].unique()) <= {"Féminin", "Masculin"}
    assert set(df["niveau_scolaire"].unique()) <= {"Universitaire", "Secondaire", "Formation Professionnelle"}


def test_data_libelles_normalises():
    df = load_raw()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or c == "motif_premiere_consultation":
            continue
        s = df[c].dropna().astype(str)
        assert (s == s.str.strip()).all(), f"{c} : espaces en trop"
        variants = s.groupby(s.str.casefold()).nunique()
        assert (variants <= 1).all(), f"{c} : variantes de casse {list(variants[variants > 1].index)}"


# ============================================================
# 3. COHÉRENCE DE LA MATRICE DE VARIABLES (recomptage indépendant)
# ============================================================
def test_features_forme_types_et_valeurs():
    df, X, D = built()
    assert len(X) == len(df) and X.index.equals(df.index)
    assert X.columns.is_unique
    assert X.notna().all().all()
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    assert (X.min() >= 0).all(), "chi2 exige des valeurs >= 0"


def test_features_aucune_colonne_exclue_ni_cible():
    df, X, D = built()
    for src in fe.EXCLUDED:
        bad = [c for c in X.columns if c == src or c.startswith(src + "__")]
        assert not bad, f"colonne exclue '{src}' presente dans la matrice : {bad}"
    assert not any("diagnostic" in c for c in X.columns), "la cible ne doit pas etre dans les variables"


def test_features_dictionnaire_coherent():
    df, X, D = built()
    assert list(D["feature"]) == list(X.columns)
    for _, r in D.iterrows():
        assert r.source_column == "derivee" or r.source_column in df.columns, r.feature
        assert r.n_non_zero == int((X[r.feature] > 0).sum()), f"{r.feature} : n_non_zero faux"
    assert set(D["tier"]) <= {"A", "B"}


def test_features_recomptage_independant():
    df, X, D = built()
    checked = 0
    for _, r in D[D.kind.isin(["binaire", "nominal", "multi-hot"])].iterrows():
        if r.feature.endswith("__autre"):
            continue
        src = r.source_column
        if r.kind == "binaire" and "__" not in r.feature:
            expected = df[src] == "Oui"
        else:
            modality = r.definition.replace("1 = ", "", 1)
            if src in fe.MULTIHOT:
                expected = modalities(df[src]).apply(lambda s, m=modality: m in s)
            else:
                expected = df[src] == modality
        assert (X[r.feature].values == expected.astype(int).values).all(), f"{r.feature} : recomptage different"
        checked += 1
    assert checked > 60, f"seulement {checked} variables verifiees"


def test_features_one_hot_exclusif():
    df, X, D = built()
    for src in fe.NOMINAL:
        cols = list(D.loc[D.source_column == src, "feature"])
        assert (X[cols].sum(axis=1) <= 1).all(), f"{src} : plusieurs modalites actives"
    cols = list(D.loc[D.source_column == "niveau_scolaire", "feature"])
    assert (X[cols].sum(axis=1) == 1).all()


def test_features_ordinal_respecte_l_ordre():
    df, X, D = built()
    for col, (order, _) in fe.ORDINAL.items():
        expected = df[col].map({v: i for i, v in enumerate(order)})
        assert (X[col] == expected).all(), col
        assert X[col].between(0, len(order) - 1).all(), col


def test_features_derivees_coherentes():
    df, X, D = built()
    yes = lambda c: (df[c] == "Oui").astype(int)
    assert (X["nb_symptomes_total"] == sum(yes(c) for c in fe.SYMPTOM_COLS)).all()
    assert (X["score_anxieux"] <= 4).all() and (X["score_psychotique"] <= 4).all()
    assert (X["score_obsessionnel"] <= 2).all()
    assert (X["nb_symptomes_total"] >= X["score_anxieux"] + X["score_psychotique"] + X["score_obsessionnel"]).all()
    expected = (df["age_premiere_consultation_generale"] - df["age_apparition_premiers_symptomes"]).clip(lower=0)
    assert (X["delai_apparition_consultation"] == expected).all()
    assert (X["delai_generale_service"] >= 0).all()
    assert X["rang_relatif_fratrie"].between(0, 1).all() and (X["rang_relatif_fratrie"] > 0).all()
    assert set(X["est_enfant_unique"].unique()) <= {0, 1}
    n_trauma = df["type_traumatisme_emotionnel"].apply(
        lambda v: 0 if pd.isna(v) else len({p.strip() for p in str(v).split(";")} - fe.ABSENCE_LABELS - {""}))
    assert (X["nb_types_traumatisme"] == n_trauma).all()


def test_features_deterministes():
    df = load_raw()
    X1, _ = fe.build_features(df, _quiet)
    X2, _ = fe.build_features(df, _quiet)
    assert X1.equals(X2), "deux executions donnent des resultats differents"


def test_features_fichier_csv_a_jour():
    assert FEATURES_CSV.exists(), "model_features.csv absent : lancez feature_engineering.py"
    df, X, D = built()
    saved = pd.read_csv(FEATURES_CSV)
    assert saved["patient_id"].tolist() == df["patient_id"].tolist()
    assert list(saved.columns[1:]) == list(X.columns), "colonnes differentes : relancez feature_engineering.py"
    assert np.allclose(saved[X.columns].values.astype(float), X.values.astype(float)), \
        "model_features.csv n'est plus a jour : relancez feature_engineering.py"


# ============================================================
# 4. MAPPING DES DIAGNOSTICS (diagnostic_mapping.py)
# ============================================================
import diagnostic_mapping as dm  # noqa: E402


def test_mapping_eda_regles_de_priorite():
    m = dm.map_eda
    cases = {
        "Schizophrénie;Trouble Dépressif Caractérisé": "Trouble Psychotique",
        "Trouble Schizo-Affectif;Trouble De L'Adaptation": "Trouble Psychotique",
        "Trouble Bipolaire Type I;Trouble De La Personnalité Borderline": "Troubles Bipolaires",
        "Trouble Bipolaire Type Ii;Phobie Sociale": "Troubles Bipolaires",
        "Trouble Dépressif Caractérisé;Trouble Obsessionnel-Compulsif (Toc)": "TOC",
        "Trouble Dépressif Caractérisé;Trouble De La Personnalité Borderline": "Trouble Dépressif",
        "Phobie Sociale;Trouble De La Personnalité Évitante": "Troubles Anxieux",
        "Trouble De L'Adaptation": "Trouble De L'Adaptation",
        "Trouble De L'Adaptation;Trouble Panique": "Troubles Anxieux",
        "Trouble Dépressif Caractérisé;Trouble Panique": "Autre",
        "Trouble Somatique": "Autre",
        "Trouble De La Personnalité Borderline": "Troubles De La Personnalité",
        "Agoraphobie": "Troubles Anxieux",
    }
    for raw, expected in cases.items():
        assert m(raw) == expected, f"{raw!r} -> {m(raw)!r} (attendu {expected!r})"
    assert m(np.nan) == "Autre"


def test_mapping_modele_regles_de_priorite():
    m = dm.map_model
    A, H, P, O = dm.CLASS_ANXIETE, dm.CLASS_HUMEUR, dm.CLASS_PSY, dm.OUT_OF_SCOPE
    cases = {
        "Schizophrénie;Trouble Dépressif Caractérisé": P,                       # psychotique > tout
        "Trouble Schizo-Affectif;Phobie Sociale": P,
        "Trouble Bipolaire Type I;Trouble Panique": H,                           # bipolaire > anxiete
        "Trouble Dépressif Caractérisé;Trouble Panique": H,                      # depression > anxiete
        "Trouble Dépressif Caractérisé;Trouble Obsessionnel-Compulsif (Toc)": A,  # exception TOC
        "Trouble Bipolaire Type Ii;Trouble Obsessionnel-Compulsif (Toc)": H,     # bipolaire > TOC
        "Trouble De L'Adaptation": A,                                            # isole -> anxiete/stress
        "Trouble De L'Adaptation;Trouble Dépressif Caractérisé": H,              # combine -> ignore
        "Trouble De L'Adaptation;Trouble Somatique": O,
        "Agoraphobie": A,
        "Trouble De La Personnalité Borderline": O,                              # pas de classe personnalite
        "Trouble Dépressif Caractérisé;Trouble De La Personnalité Borderline": H,
        "Trouble De La Personnalité Schizoide": O,   # cluster A hors classe (decision a confirmer)
        "Trouble Somatique": O,
    }
    for raw, expected in cases.items():
        assert m(raw) == expected, f"{raw!r} -> {m(raw)!r} (attendu {expected!r})"
    assert m(np.nan) == O


def test_mapping_modele_sans_classe_personnalite():
    df = load_raw()
    produced = set(df["diagnostic_principal"].apply(dm.map_model))
    assert produced <= set(dm.MODEL_CLASSES + [dm.OUT_OF_SCOPE])
    assert not any("personnalit" in c.lower() for c in produced)


def test_mapping_coherence_entre_les_deux_mappings():
    """Quand le mapping EDA donne une categorie, le mapping modele donne la classe correspondante."""
    df = load_raw()
    projection = {
        "Troubles Anxieux": dm.CLASS_ANXIETE, "Trouble De L'Adaptation": dm.CLASS_ANXIETE,
        "TOC": dm.CLASS_ANXIETE, "Trouble Dépressif": dm.CLASS_HUMEUR,
        "Troubles Bipolaires": dm.CLASS_HUMEUR, "Trouble Psychotique": dm.CLASS_PSY,
    }
    for raw in df["diagnostic_principal"].dropna().unique():
        if any(p.strip() in dm.PERSONNALITE_CLUSTER_A for p in raw.split(";")):
            continue   # divergence voulue et documentee : cluster A psychotique en EDA, hors classe en modele
        eda = dm.map_eda(raw)
        if eda in projection:
            assert dm.map_model(raw) == projection[eda], \
                f"{raw!r} : EDA = {eda!r} mais modele = {dm.map_model(raw)!r}"


def test_mapping_couverture_sur_les_donnees():
    df = load_raw()
    diag = df["diagnostic_principal"]
    eda = diag.apply(dm.map_eda).value_counts()
    assert eda.get("Autre", 0) / len(df) < 0.15, "trop de patients dans 'Autre' (EDA)"
    for c in [x for x in dm.DIAG_ORDER if x != "Autre"]:
        assert eda.get(c, 0) >= 10, f"categorie EDA '{c}' trop petite ({eda.get(c, 0)})"
    mod = diag.apply(dm.map_model).value_counts()
    assert mod.get("Autre", 0) / len(df) < 0.20, "trop de patients hors perimetre (modele)"
    for c in dm.MODEL_CLASSES:
        assert mod.get(c, 0) >= 20, f"classe modele '{c}' trop petite ({mod.get(c, 0)})"


def test_mapping_audit_complet():
    df = load_raw()
    audit = dm.audit_table(df["diagnostic_principal"])
    assert audit["n_patients"].sum() == len(df)
    assert set(audit["classe_eda"]) <= set(dm.DIAG_ORDER)
    assert set(audit["classe_modele"]) <= set(dm.MODEL_CLASSES + [dm.OUT_OF_SCOPE])
    assert audit["regle_eda"].notna().all() and audit["regle_modele"].notna().all()


def test_mapping_source_unique():
    """Plus aucune copie de la logique : les scripts importent diagnostic_mapping."""
    import profiling_model as pm
    assert pm.map_diagnostic_category is dm.map_eda
    cwd = os.getcwd()
    os.chdir(SRC_DIR)
    try:
        import exploratory_analysis as ea
    except Exception as e:   # matplotlib absent, etc. : on ne bloque pas
        print(f"     (ignore : import de exploratory_analysis impossible : {type(e).__name__})")
        return
    finally:
        os.chdir(cwd)
    assert ea.map_diagnostic_category is dm.map_eda


# ============================================================
# 5. DASHBOARD (test de fumee avec AppTest de Streamlit)
# ============================================================
def _dashboard_app():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("     (ignore : streamlit non installe)")
        return None
    for name in ("dashboard_app.py", "dashboard.py"):
        if (SRC_DIR / name).exists():
            return AppTest.from_file(str(SRC_DIR / name), default_timeout=120)
    print("     (ignore : dashboard introuvable dans src/)")
    return None


def test_dashboard_filtre_par_defaut_affiche_tous_les_patients():
    """Garde-fou contre une panne silencieuse : un filtre aux anciens noms de categories
    n'affichait que 26 patients sur 252, sans aucune erreur."""
    at = _dashboard_app()
    if at is None:
        return
    at.run()
    assert not at.exception, at.exception[0].value
    assert list(at.sidebar.multiselect[0].options) == list(dm.DIAG_ORDER), \
        "le filtre ne propose pas les categories de diagnostic_mapping.py"
    n = len(load_raw())
    caption = [c.value for c in at.sidebar.caption if "patients" in c.value][0]
    assert caption.startswith(f"{n} / {n}"), f"filtre par defaut : {caption!r}"


def test_dashboard_toutes_les_pages_s_affichent():
    at = _dashboard_app()
    if at is None:
        return
    at.run()
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"page '{page}' : {at.exception[0].value}"
        assert len(at.get("plotly_chart")) + len(at.get("dataframe")) > 0, f"page '{page}' vide"


# ============================================================
# 6. SELECTION DE VARIABLES (feature_selection.py)
# ============================================================
import feature_selection as fs  # noqa: E402


def test_selection_jeu_de_donnees_a_trois_classes():
    X, y, D, ids = fs.load_dataset()
    raw = load_raw()
    expected = int((raw["diagnostic_principal"].apply(dm.map_model) != dm.OUT_OF_SCOPE).sum())
    assert len(X) == len(y) == len(ids) == expected
    assert set(y.unique()) == set(dm.MODEL_CLASSES), "les patients hors perimetre doivent etre exclus"
    assert ids.is_unique and X.notna().all().all()
    assert list(X.columns) == list(D["feature"])


def test_selection_jeux_de_variables():
    X, y, D, ids = fs.load_dataset()
    sets = fs.get_feature_sets(D)
    tier_b = set(D.loc[D.tier == "B", "feature"])
    assert set(sets["A"]).isdisjoint(tier_b) and set(sets["A"]) | tier_b == set(sets["A+B"]) == set(X.columns)
    base = set(sets["baseline17"])
    assert set(fe.SYMPTOM_COLS) <= base and "sexe__Masculin" in base
    assert "age_apparition_premiers_symptomes" in base and "age_premiere_consultation_generale" in base
    assert not (base & set(D.loc[D.source_column == "derivee", "feature"])), "pas de variable derivee dans baseline17"


def test_selection_variables_continues_detectees():
    X, y, D, ids = fs.load_dataset()
    mask = fs.discrete_mask(X)
    assert not mask[X.columns.get_loc("age_apparition_premiers_symptomes")], "un age est continu"
    assert mask[X.columns.get_loc("syndrome_depressif")], "un indicateur 0/1 est discret"


def test_selection_retrouve_un_signal_plante():
    """3 classes, 40 colonnes de bruit, 1 colonne binaire et 1 colonne continue informatives :
    chaque methode doit les classer en tete (valide scores, rangs et alignement du masque)."""
    rng = np.random.default_rng(0)
    n = 180
    y = pd.Series(np.repeat(["a", "b", "c"], n // 3))
    Xs = pd.DataFrame(rng.integers(0, 2, size=(n, 40)), columns=[f"bruit_{i}" for i in range(40)])
    flip = rng.random(n) < 0.1
    Xs["signal_binaire"] = ((y == "a") ^ flip).astype(int)
    Xs["signal_continu"] = rng.normal(0, 1, n) + (y == "c").astype(float) * 2.5
    for method in fs.METHODS:
        chosen = fs.select_columns(Xs, y, k=2, method=method)
        assert set(chosen) == {"signal_binaire", "signal_continu"}, f"{method} : {chosen}"


def test_selection_deterministe_et_bonne_taille():
    X, y, D, ids = fs.load_dataset()
    for method in fs.METHODS:
        a = fs.select_columns(X, y, k=12, method=method)
        b = fs.select_columns(X, y, k=12, method=method)
        assert a == b and len(a) == 12 and len(set(a)) == 12, method


def test_selection_chi2_insensible_a_l_echelle():
    """chi2 est ramene a [0,1] : multiplier une colonne par 100 ne doit pas changer son rang."""
    X, y, D, ids = fs.load_dataset()
    X2 = X.copy()
    X2["age_apparition_premiers_symptomes"] *= 100
    s1, s2 = fs.score_chi2(X, y), fs.score_chi2(X2, y)
    assert np.allclose(s1, s2)


# ============================================================
# 7. NAIVE BAYES ECRIT A LA MAIN (naive_bayes.py)
# ============================================================
from naive_bayes import NaiveBayes  # noqa: E402


def _toy(seed=0, n=120):
    rng = np.random.default_rng(seed)
    y = np.repeat(["a", "b", "c"], n // 3)
    X = pd.DataFrame({
        "bin": (rng.random(n) < np.where(y == "a", 0.85, 0.15)).astype(int),
        "ord": np.clip(rng.integers(0, 3, n) + (y == "b").astype(int), 0, 3),
        "cont": rng.normal(0, 1, n) + (y == "c") * 2.5,
        "bruit": rng.integers(0, 2, n),
    })
    return X, y


def test_nb_identique_a_sklearn_bernoulli():
    """Colonnes binaires de frequence globale exactement 0.5 : notre lissage (c + a*0.5)/(n + a)
    doit egaler celui de sklearn avec alpha = a/2."""
    from sklearn.naive_bayes import BernoulliNB
    rng = np.random.default_rng(1)
    n, d = 120, 12
    y = np.repeat(["a", "b", "c"], n // 3)
    X = pd.DataFrame({f"v{j}": rng.permutation([0] * (n // 2) + [1] * (n // 2)) for j in range(d)})
    ours = NaiveBayes(alpha=2.0, prior="uniform").fit(X, y)
    sk = BernoulliNB(alpha=1.0, fit_prior=False).fit(X, y)
    Xt = pd.DataFrame(rng.integers(0, 2, size=(40, d)), columns=X.columns)
    assert np.abs(ours.predict_proba(Xt) - sk.predict_proba(Xt)).max() < 1e-9
    assert (ours.predict(Xt) == sk.predict(Xt)).all()


def test_nb_probabilites_valides_et_jamais_nulles():
    X, y = _toy()
    p = NaiveBayes().fit(X, y).predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all() and (p <= 1).all()


def test_nb_lissage_evite_les_probabilites_nulles():
    X, y = _toy()
    X["jamais_vu_en_a"] = (y == "b").astype(int)      # jamais egal a 1 dans la classe "a"
    lisse = NaiveBayes(alpha=2.0).fit(X, y)
    j = lisse.columns_.index("jamais_vu_en_a")
    k = list(lisse.bern_["idx"]).index(j)
    p_a = np.exp(lisse.bern_["logp1"][0, k])
    assert 0 < p_a < 0.1
    brut = NaiveBayes(alpha=0.0).fit(X, y)
    assert np.exp(brut.bern_["logp1"][0, k]) <= 1e-5, "sans lissage : plancher"


def test_nb_retrecissement_gaussien():
    X, y = _toy()
    j = lambda m: [p for p in m.gauss_ if m.columns_[p["idx"]] == "cont"][0]
    libre = NaiveBayes(shrink_k=0.0).fit(X, y)
    colle = NaiveBayes(shrink_k=1e9).fit(X, y)
    for i, c in enumerate(libre.classes_):
        assert abs(j(libre)["mean"][i] - X["cont"][y == c].mean()) < 1e-9
    assert np.ptp(j(colle)["mean"]) < 1e-6, "k enorme : toutes les classes ramenees a la moyenne globale"


def test_nb_prior_empirique_ou_uniforme():
    X, y = _toy()
    y2 = np.array(["a"] * 90 + ["b"] * 30)
    emp = NaiveBayes(prior="empirical").fit(X.iloc[:120], y2)
    uni = NaiveBayes(prior="uniform").fit(X.iloc[:120], y2)
    assert np.allclose(np.exp(emp.log_prior_), [0.75, 0.25]) and np.allclose(np.exp(uni.log_prior_), [0.5, 0.5])


def test_nb_explication_exacte():
    X, y = _toy()
    m = NaiveBayes().fit(X, y)
    for i in (0, 50, 100):
        e = m.explain(X.iloc[i])
        assert abs(e["terme_prior"] + e["toutes"]["contribution"].sum() - e["log_rapport_total"]) < 1e-9
        assert e["classe_predite"] == m.predict(X.iloc[[i]])[0]


def test_nb_retrouve_un_signal_plante():
    X, y = _toy()
    Xt, yt = _toy(seed=7)
    assert (NaiveBayes().fit(X, y).predict(Xt) == yt).mean() > 0.85


def test_nb_niveau_ou_valeur_jamais_vue():
    X, y = _toy()
    m = NaiveBayes().fit(X, y)
    out = m.predict(pd.DataFrame({"bin": [1], "ord": [9], "cont": [1e6], "bruit": [0]}))
    assert out[0] in ("a", "b", "c")


# ============================================================
# 8. PROTOCOLE DE COMPARAISON (model_comparison.py)
# ============================================================
import model_comparison as mc  # noqa: E402


def test_protocole_signal_reel_et_hasard_sur_etiquettes_melangees():
    """Controle de bon sens : bien au-dessus du hasard avec les vraies etiquettes, au niveau du
    hasard avec des etiquettes melangees. (Ne detecte que les grosses fuites : voir le test suivant.)"""
    X, y, D, ids = fs.load_dataset()
    mask = fs.discrete_mask(X)
    cols = fs.get_feature_sets(D)["baseline17"]
    kw = dict(models=("NB",), k_grid=[5, 10], nb_grid=[dict(alpha=2.0, shrink_k=5.0, prior="uniform")],
              n_repeats=1, seed=3)
    vrai = mc.run_set(X, y, cols, mask, **kw)["nested"]["NB"]["macro_f1"]
    y_perm = pd.Series(np.random.default_rng(5).permutation(y.values))
    melange = mc.run_set(X, y_perm, cols, mask, **kw)["nested"]["NB"]["macro_f1"]
    assert vrai > 0.60, f"vraies etiquettes : {vrai:.3f}"
    assert melange < 0.45, f"etiquettes melangees : {melange:.3f} (hasard ~0.33) -> fuite possible"


def test_protocole_le_classement_ne_voit_jamais_le_pli_de_test():
    """Test structurel (deterministe) : on espionne rank_columns et on verifie que, pour chaque pli
    externe, les 4 classements (3 internes + 1 externe) ne contiennent AUCUN patient du pli de test."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    X, y, D, ids = fs.load_dataset()
    mask = fs.discrete_mask(X)
    cols = fs.get_feature_sets(D)["baseline17"]
    calls, original = [], fs.rank_columns

    def spy(Xtr, ytr, *a, **k):
        calls.append(set(Xtr.index))
        return original(Xtr, ytr, *a, **k)

    fs.rank_columns = spy
    try:
        mc.run_set(X, y, cols, mask, models=("NB",), k_grid=[5],
                   nb_grid=[dict(alpha=2.0, shrink_k=5.0, prior="uniform")], n_repeats=1, seed=3)
    finally:
        fs.rank_columns = original
    folds = list(RepeatedStratifiedKFold(n_splits=mc.N_SPLITS, n_repeats=1, random_state=3).split(X[cols], y))
    assert len(calls) == 4 * len(folds), f"{len(calls)} classements pour {len(folds)} plis"
    for f, (tr, te) in enumerate(folds):
        for c in calls[4 * f:4 * f + 4]:
            assert not (c & set(X.index[te])), f"pli {f} : le classement des variables a vu des patients du pli de test"
            assert c <= set(X.index[tr])


def test_protocole_grilles_de_configuration():
    ks = lambda n: [c["k"] for c in mc.make_configs("RF", n)]
    assert ks(19) == [5, 8, 10, 15, 19], "k >= nb de colonnes : ramene a 'toutes', sans doublon"
    assert ks(125) == [5, 8, 10, 15, 20, 125]
    assert len(mc.make_configs("NB", 125)) == 6 * len(mc.NB_GRID)


def test_protocole_resume_des_metriques():
    y = pd.Series(mc.CLASSES * 10)
    parfait = np.array([y.values, y.values], dtype=object)
    s = mc.summarize(parfait, y)
    assert s["macro_f1"] == 1.0 and s["balanced_acc"] == 1.0
    tout_premier = np.array([[mc.CLASSES[0]] * 30], dtype=object)    # precision 1/3, rappel 1 -> F1 = 0.5
    s = mc.summarize(tout_premier, y)
    assert abs(s[f"f1_{mc.CLASSES[0]}"] - 0.5) < 1e-9 and abs(s["macro_f1"] - 0.5 / 3) < 1e-9


def test_protocole_test_t_corrige():
    from scipy.stats import ttest_rel
    a = np.array([0.70, 0.72, 0.69, 0.74, 0.71, 0.73])
    b = np.array([0.66, 0.70, 0.66, 0.71, 0.70, 0.69])
    diff, p_corr = mc.corrected_ttest(a, b)
    assert abs(diff - (a - b).mean()) < 1e-12
    assert p_corr > ttest_rel(a, b).pvalue, "la correction doit rendre le test plus prudent"
    assert mc.corrected_ttest(a, a) == (0.0, 1.0)


# ============================================================
# 9. CLASSIFIEUR DE PATIENT (patient_classifier.py) ET PAGES DU DASHBOARD QUI L'UTILISENT
# ============================================================
import patient_classifier as pc  # noqa: E402

_clf_cache = {}


def _clf():
    if "c" not in _clf_cache:
        _clf_cache["c"] = pc.PatientClassifier.train(load_raw(), n_repeats=1)
    return _clf_cache["c"]


def test_classifieur_variables_retenues():
    c, D = _clf(), built()[2]
    assert len(c.features) == pc.K_FEATURES and len(set(c.features)) == len(c.features)
    assert not set(c.features) & set(D.loc[D.source_column == "derivee", "feature"]), "pas de variable derivee"
    src = D.set_index("feature").loc[c.features, "source_column"]
    assert not set(src) & set(fe.EXCLUDED), "aucune colonne exclue (cible, fuite, suivi) ne doit servir au modele"


def test_features_dictionnaire_conserve_les_modalites_rares():
    D = built()[2]
    for _, r in D[D.feature.str.endswith("__autre")].iterrows():
        assert " : " in r.definition and r.definition.split(" : ", 1)[1].strip(), r.feature


def test_classifieur_encodeur_unitaire_identique_a_l_encodage_par_lot():
    """Le meme patient, encode seul a partir des valeurs saisies, doit donner exactement la meme ligne
    que l'encodage de tous les patients d'un coup (feature_engineering.py)."""
    c, raw = _clf(), load_raw()
    X = built()[1]
    for i in range(len(raw)):
        enc = c.encode(c.raw_from_record(raw.iloc[i]))
        assert (enc.values == X.loc[i, c.features].values).all(), f"patient {i}"


def test_classifieur_modalites_rares_et_variables_derivees():
    info = pc._feature_info("col__autre", {"source_column": "col", "kind": "nominal", "tier": "A",
                                           "definition": "1 = modalites rares regroupees (< 5 patients chacune) : A | B"})
    assert info["mode"] == "in_rare" and info["rare"] == ["A", "B"]
    assert raises(NotImplementedError, pc._feature_info, "nb_symptomes_total",
                  {"source_column": "derivee", "kind": "derivee", "tier": "B", "definition": "x"})


def test_classifieur_probabilites_et_classe_coherentes():
    c, raw = _clf(), load_raw()
    for i in (0, 25, 120, 200):
        r = c.raw_from_record(raw.iloc[i])
        out = c.predict(r)
        p = np.array(list(out["probas"].values()))
        assert abs(p.sum() - 1) < 1e-9 and (p > 0).all()
        lj = c.model.predict_log_joint(c.encode(r).to_frame().T)[0]
        assert out["classe"] == c.classes[int(np.argmax(lj))]
        assert abs(out["confiance"] - p.max()) < 1e-12
        assert out["incertain"] == (out["confiance"] < pc.UNCERTAIN_BELOW)


def test_classifieur_explication_exacte():
    """prior + somme des contributions = log-rapport, et ce log-rapport est celui des probabilites affichees."""
    c, raw = _clf(), load_raw()
    for i in (3, 77, 150):
        out = c.predict(c.raw_from_record(raw.iloc[i]))
        e, pr = out["explication"], out["probas"]
        assert len(e) == len(c.features)
        assert abs(out["terme_prior"] + e["contribution"].sum() - out["log_rapport"]) < 1e-9
        assert abs(np.log(pr[out["classe"]] / pr[out["deuxieme_classe"]]) - out["log_rapport"]) < 1e-9


def test_classifieur_calibration():
    c = _clf()
    assert c.calibration["T"] > 1.0 and c.calibration["log_loss_calibre"] < c.calibration["log_loss_brut"]
    acc = [b["accuracy"] for b in c.bands if b["n"] >= 30]
    assert acc == sorted(acc), f"la justesse doit croitre avec la confiance : {acc}"


def test_classifieur_profils_types_classes_dans_leur_categorie():
    c = _clf()
    for name in c.classes:
        assert c.predict(c.typical_inputs(name))["classe"] == name


def test_classifieur_formulaire_complet():
    c = _clf()
    assert {i["source"] for i in c.feature_info.values()} == {sp["source"] for sp in c.inputs}
    for sp in c.inputs:
        assert sp["kind"] in ("yes_no", "choice", "ordinal", "multi", "number")
        assert sp["kind"] == "number" or sp["choices"]
    assert c.predict(c.default_inputs())["classe"] in c.classes


def test_classifieur_deterministe():
    a = pc.PatientClassifier.train(load_raw(), n_repeats=1)
    assert a.features == _clf().features and abs(a.T - _clf().T) < 1e-12


def test_profiling_model_profil_type_uniquement():
    import profiling_model as pm
    assert hasattr(pm.DiagnosticProfileModel, "describe")
    assert not hasattr(pm.DiagnosticProfileModel, "sample"), "la generation aleatoire a ete retiree"
    assert not hasattr(pm, "compare_classifiers"), "la comparaison de modeles vit dans model_comparison.py"


def test_dashboard_profil_type_sans_generation_aleatoire():
    at = _dashboard_app()
    if at is None:
        return
    at.run()
    at.sidebar.radio[0].set_value("Profil type par catégorie").run()
    assert not at.exception, at.exception[0].value
    assert len(at.button) == 0 and len(at.slider) == 0, "plus de bouton ni de curseur de generation aleatoire"
    assert len(at.get("plotly_chart")) >= 2


def test_dashboard_classification_profils_types_et_interaction():
    at = _dashboard_app()
    if at is None:
        return
    at.run()
    at.sidebar.radio[0].set_value("Classification d'un patient").run()
    assert not at.exception, at.exception[0].value
    confiance = lambda: [m.value for m in at.metric if m.label.startswith("Confiance")][0]
    for cls in at.selectbox(key="prefill").options[1:]:
        at.selectbox(key="prefill").select(cls).run()
        assert not at.exception, at.exception[0].value
        assert at.header[0].value == cls, f"profil type '{cls}' classe en '{at.header[0].value}'"
    before = (at.header[0].value, confiance())
    radio = next(w for w in at.radio if w.key and w.key.startswith("in_"))
    radio.set_value("Oui" if radio.value == "Non" else "Non").run()
    assert not at.exception and (at.header[0].value, confiance()) != before, "changer une reponse doit changer le resultat"


# ============================================================
# LANCEUR (sans pytest)
# ============================================================
def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Exception as e:
            failed += 1
            print(f"ECHEC {name}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests reussis")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()