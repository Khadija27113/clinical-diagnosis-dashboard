"""
dashboard_app.py
Tableau de bord Streamlit — synthese finale du projet.

Reunit dans une seule interface interactive :
  1. Les analyses exploratoires (exploratory_analysis.py), sur le mapping a 7 categories
     defini dans diagnostic_mapping.py (source unique des categories)
  2. Le profil type par categorie (profiling_model.py) -- aucune generation aleatoire
  3. La classification d'un patient saisi a la main, avec explication (patient_classifier.py)

Lancement (depuis DataAnalyseProject/src) :
    streamlit run dashboard_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from profiling_model import (
    DiagnosticProfileModel,
    AGE_ONSET_COL,
    AGE_CONSULT_COL,
    SEXE_COL,
    EDU_COL,
    SYMPTOM_COLS,
    DATA_PATH,
    train_model,
)
from diagnostic_mapping import DIAG_ORDER, map_eda
from patient_classifier import UNCERTAIN_BELOW, PatientClassifier, pretty

# ==============================================================
# Configuration generale
# ==============================================================
st.set_page_config(
    page_title="Tableau de bord clinique — Analyse psychiatrique",
    page_icon="🧠",
    layout="wide",
)

PALETTE = {
    "Troubles Anxieux": "#0EA5E9",
    "Troubles Bipolaires": "#F59E0B",
    "Trouble Dépressif": "#6366F1",
    "Trouble Psychotique": "#7C3AED",
    "Trouble De L'Adaptation": "#10B981",
    "TOC": "#EC4899",
    "Troubles De La Personnalité": "#EF4444",
    "Autre": "#94A3B8",
}
SUICIDE_COLORS = {
    "Aucune_Ideation": "#16A34A", "Idees_Mort_Passives": "#A3E635",
    "Idees_Suicidaires_Actives": "#FB923C", "Tentative_Ou_Automutilation": "#B91C1C",
}


# ==============================================================
# Chargement des donnees (reprend la logique de exploratory_analysis.py)
# ==============================================================
def ensure_data_available():
    """Sur Streamlit Cloud, data/processed/ est exclu du repo (.gitignore, donnees
    patients sensibles). Si le fichier est absent, on propose un import manuel
    (session uniquement, jamais commite/persiste)."""
    if DATA_PATH.exists():
        return

    st.title("🧠 Tableau de bord clinique — Analyse psychiatrique")
    st.warning(
        "Fichier de données introuvable sur ce serveur "
        "(`data/processed/merged_data.csv` est volontairement exclu du dépôt Git "
        "car il contient des données patients).\n\n"
        "Importez le fichier `merged_data.csv` pour utiliser le tableau de bord. "
        "Il n'est ni sauvegardé de façon permanente, ni commité — seulement chargé "
        "en mémoire pour cette session."
    )
    uploaded = st.file_uploader("Importer merged_data.csv", type="csv")
    if uploaded is None:
        st.stop()

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "wb") as f:
        f.write(uploaded.getbuffer())
    st.rerun()


ensure_data_available()


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    df["diag_cat"] = df["diagnostic_principal"].apply(map_eda)

    df["high_risk"] = df["niveau_suicidalite"].isin(
        ["Idees_Suicidaires_Actives", "Tentative_Ou_Automutilation"]
    )

    def eval_evolution(row):
        first = row.get("etat_clinique_premiere_consult", None)
        last = row.get("etat_clinique_derniere_consult", None)
        if pd.isna(first) or pd.isna(last):
            return "Non renseigné"
        if last == "Stabilise" and first != "Stabilise":
            return "Amélioration"
        elif last == "Stabilise" and first == "Stabilise":
            return "Stable"
        elif last == "Amelioration_Partielle":
            return "Amélioration partielle"
        elif last == "Aggravation_Recente":
            return "Aggravation"
        elif last == "Symptomes_Actifs":
            return "Symptômes persistants" if first == "Symptomes_Actifs" else "Aggravation"
        return "Autre"

    df["evolution"] = df.apply(eval_evolution, axis=1)
    df["ameliorated"] = df["evolution"].isin(["Amélioration", "Amélioration partielle", "Stable"])

    def classify_substance(s):
        if pd.isna(s) or s == "Rien À Signaler":
            return "Aucune"
        if s == "Inconnu":
            return "Inconnu"
        substances = [x.strip() for x in s.split(";")]
        if len(substances) >= 2:
            return "Polysubstances"
        elif "Tabac" in substances:
            return "Tabac"
        elif "Alcool" in substances:
            return "Alcool"
        elif "Cannabis" in substances:
            return "Cannabis"
        return "Autre"

    df["substance_cat"] = df["atcd_abus_spa"].apply(classify_substance)

    return df


@st.cache_resource
def get_profiling_model() -> DiagnosticProfileModel:
    return train_model()


@st.cache_resource(show_spinner="Entraînement du modèle de classification (environ 15 s, une seule fois)…")
def get_classifier() -> PatientClassifier:
    """Entraine le classifieur au premier affichage de la page (aucun modele n'est stocke dans le depot)."""
    return PatientClassifier.train(pd.read_csv(DATA_PATH, encoding="utf-8"))


df_full = load_data()
model = get_profiling_model()


def color_map(categories):
    return {c: PALETTE.get(c, "#CBD5E1") for c in categories}


# ==============================================================
# Barre laterale — navigation + filtre global
# ==============================================================
st.sidebar.title("🧠 Navigation")
page = st.sidebar.radio(
    "Section",
    [
        "Vue d'ensemble",
        "Profil académique & social",
        "Indicateurs de risque",
        "Suivi thérapeutique",
        "Synthèse par profil",
        "Questions de recherche",
        "Profil type par catégorie",
        "Classification d'un patient",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtre global")
selected_cats = st.sidebar.multiselect(
    "Catégories diagnostiques incluses",
    options=DIAG_ORDER,
    default=DIAG_ORDER,
    help="S'applique aux sections d'analyse. Les sections 'Profil type par catégorie' et "
    "'Classification d'un patient' utilisent toujours l'ensemble des données (modèles entraînés sur tous les patients).",
)
df = df_full[df_full["diag_cat"].isin(selected_cats)] if selected_cats else df_full.iloc[0:0]

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(df)} / {len(df_full)} patients affichés")


def bar_pct(series: pd.Series, order=None, title="", color="#0EA5E9"):
    counts = series.value_counts()
    if order:
        counts = counts.reindex(order, fill_value=0)
    total = counts.sum()
    pct = (counts / total * 100).round(1) if total else counts
    fig = px.bar(x=counts.index, y=counts.values, text=[f"{p}%" for p in pct])
    fig.update_traces(marker_color=color, textposition="outside")
    fig.update_layout(title=title, xaxis_title="", yaxis_title="Nombre de patients", showlegend=False)
    return fig


def stacked_pct(data, value_col, value_order=None, colors=None, title=""):
    """Repartition (%) d'une variable au sein de chaque categorie diagnostique.
    L'effectif n de chaque categorie est affiche : les petites categories sont instables."""
    counts = data["diag_cat"].value_counts()
    ct = pd.crosstab(data["diag_cat"], data[value_col], normalize="index") * 100
    ct = ct.reindex([c for c in DIAG_ORDER if c in ct.index])
    if value_order:
        ct = ct.reindex(columns=[v for v in value_order if v in ct.columns])
    ct.index = [f"{c}<br>(n={counts[c]})" for c in ct.index]
    long = ct.reset_index(names="categorie").melt(id_vars="categorie", var_name=value_col, value_name="pct")
    fig = px.bar(long, x="categorie", y="pct", color=value_col, barmode="stack",
                 category_orders={"categorie": list(ct.index)}, color_discrete_map=colors or {})
    fig.update_layout(title=title, xaxis_title="", yaxis_title="% de patients",
                      yaxis_range=[0, 100], legend_title_text="")
    return fig


def pie(series: pd.Series, order=None, title=""):
    counts = series.value_counts()
    if order:
        counts = counts.reindex(order, fill_value=0)
    fig = px.pie(names=counts.index, values=counts.values, title=title, hole=0.35)
    fig.update_traces(textinfo="percent+label")
    return fig


# ==============================================================
# PAGE 1 — Vue d'ensemble
# ==============================================================
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble de la cohorte")
    st.caption("Projet d'analyse de données cliniques psychiatriques — synthèse finale")

    if df.empty:
        st.warning("Aucun patient ne correspond au filtre sélectionné.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients (filtrés)", len(df))
        c2.metric("Âge moyen d'apparition", f"{df[AGE_ONSET_COL].mean():.0f} ans")
        c3.metric("% risque suicidaire élevé", f"{df['high_risk'].mean()*100:.0f}%")
        c4.metric("Hospitalisations / patient", f"{df['nb_hospitalisations_int'].mean():.2f}")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                pie(df["diag_cat"], order=DIAG_ORDER, title="Répartition des catégories diagnostiques"),
                width="stretch",
            )
        with col2:
            st.plotly_chart(
                pie(df[SEXE_COL], title="Répartition par sexe"),
                width="stretch",
            )

        st.markdown(
            "Ce tableau de bord reprend l'ensemble du pipeline du projet : nettoyage des exports "
            "Label Studio → prétraitement/fusion → analyse exploratoire → profil type par catégorie "
            "→ classification d'un patient. Utilisez le menu à gauche pour naviguer entre les sections, et le filtre "
            "global pour restreindre l'analyse à une ou plusieurs catégories diagnostiques. Les "
            "7 catégories (et « Autre » pour les diagnostics hors périmètre) suivent le mapping validé "
            "avec la résidente ; voir `diagnostic_mapping.py`."
        )

# ==============================================================
# PAGE 2 — Profil académique & social
# ==============================================================
elif page == "Profil académique & social":
    st.title("Profil académique et social")

    if df.empty:
        st.warning("Aucun patient ne correspond au filtre sélectionné.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pie(df["niveau_scolaire"], title="Niveau scolaire"), width="stretch")
        with col2:
            st.plotly_chart(pie(df["niveau_socioeconomique"], title="Niveau socio-économique"), width="stretch")

        st.subheader("Redoublements moyens par catégorie diagnostique")
        redoub = df.groupby("diag_cat")["nombre_redoublements"].mean().reindex(
            [c for c in DIAG_ORDER if c in df["diag_cat"].unique()]
        )
        fig = px.bar(x=redoub.index, y=redoub.values, color=redoub.index, color_discrete_map=color_map(redoub.index))
        fig.update_layout(xaxis_title="", yaxis_title="Nombre moyen de redoublements", showlegend=False)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Niveau scolaire au sein de chaque catégorie diagnostique")
        st.plotly_chart(stacked_pct(df.dropna(subset=["niveau_scolaire"]), "niveau_scolaire"),
                        width="stretch")

# ==============================================================
# PAGE 3 — Indicateurs de risque
# ==============================================================
elif page == "Indicateurs de risque":
    st.title("Indicateurs de risque")

    if df.empty:
        st.warning("Aucun patient ne correspond au filtre sélectionné.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            order_suic = ["Aucune_Ideation", "Idees_Mort_Passives", "Idees_Suicidaires_Actives", "Tentative_Ou_Automutilation"]
            st.plotly_chart(pie(df["niveau_suicidalite"], order=order_suic, title="Niveau de suicidalité"), width="stretch")
        with col2:
            st.plotly_chart(pie(df["substance_cat"], title="Consommation de substances psychoactives"), width="stretch")

        st.subheader("Niveau de suicidalité au sein de chaque catégorie diagnostique")
        st.plotly_chart(
            stacked_pct(df.dropna(subset=["niveau_suicidalite"]), "niveau_suicidalite",
                        value_order=order_suic, colors=SUICIDE_COLORS),
            width="stretch",
        )

        st.subheader("Proportion de patients à risque suicidaire élevé selon les facteurs")
        factors = {
            "Antécédents familiaux": df["atcd_familiaux_psy_trouble"].notna() & (df["atcd_familiaux_psy_trouble"] != "Rien À Signaler"),
            "Traumatisme": df["type_traumatisme_emotionnel"].notna() & (df["type_traumatisme_emotionnel"] != "Aucun"),
            "Abus substances": df["atcd_abus_spa"].notna() & (df["atcd_abus_spa"] != "Rien À Signaler") & (df["atcd_abus_spa"] != "Inconnu"),
            "Conflits familiaux": df["situation_familiale"].str.contains("Conflit|Divorcé", na=False),
        }
        risk_rates = {name: df[cond]["high_risk"].mean() * 100 for name, cond in factors.items() if cond.any()}
        if risk_rates:
            fig = px.bar(x=list(risk_rates.keys()), y=list(risk_rates.values()), text=[f"{v:.0f}%" for v in risk_rates.values()])
            fig.update_traces(marker_color="#EF4444", textposition="outside")
            fig.update_layout(yaxis_title="% avec risque élevé", xaxis_title="", yaxis_range=[0, 100])
            st.plotly_chart(fig, width="stretch")

        st.subheader("Nombre moyen d'hospitalisations par patient, par catégorie diagnostique")
        hosp = df.groupby("diag_cat")["nb_hospitalisations_int"].mean().reindex(
            [c for c in DIAG_ORDER if c in df["diag_cat"].unique()]
        )
        fig = px.bar(x=hosp.index, y=hosp.values, color=hosp.index, color_discrete_map=color_map(hosp.index))
        fig.update_layout(xaxis_title="", yaxis_title="Hospitalisations par patient", showlegend=False)
        st.plotly_chart(fig, width="stretch")

# ==============================================================
# PAGE 4 — Suivi therapeutique
# ==============================================================
elif page == "Suivi thérapeutique":
    st.title("Suivi thérapeutique")

    if df.empty:
        st.warning("Aucun patient ne correspond au filtre sélectionné.")
    else:
        order_evol = ["Amélioration", "Amélioration partielle", "Stable", "Symptômes persistants", "Aggravation", "Autre", "Non renseigné"]
        st.plotly_chart(bar_pct(df["evolution"], order=order_evol, title="Évolution clinique (1ère → dernière consultation)", color="#22C55E"), width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            order_obs = ["Bonne", "Moyenne", "Mauvaise", "Naif_Traitement", "Arret_Traitement"]
            st.plotly_chart(pie(df["observance_derniere_consult"], order=order_obs, title="Observance du traitement"), width="stretch")
        with col2:
            st.plotly_chart(pie(df["tolerance_derniere_consult"], title="Tolérance au traitement"), width="stretch")

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Amélioration selon l'observance")
            valid = df.dropna(subset=["evolution", "observance_derniere_consult"])
            valid = valid[~valid["evolution"].isin(["Non renseigné", "Autre"])]
            order_obs2 = ["Bonne", "Moyenne", "Mauvaise"]
            obs_impact = valid.groupby("observance_derniere_consult")["ameliorated"].mean().reindex(order_obs2, fill_value=0) * 100
            fig = px.bar(x=obs_impact.index, y=obs_impact.values, text=[f"{v:.0f}%" for v in obs_impact.values])
            fig.update_traces(marker_color="#0D9488", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="% amélioré/stable", yaxis_range=[0, 100])
            st.plotly_chart(fig, width="stretch")
        with col4:
            st.subheader("Rechute à 12 mois selon la sévérité du traumatisme")
            valid = df.dropna(subset=["severite_traumatisme", "rechute_12_mois"])
            order_sev = ["Léger", "Modéré", "Sévère"]
            relapse = valid.groupby("severite_traumatisme")["rechute_12_mois"].apply(lambda x: (x == "Oui").mean() * 100).reindex(order_sev, fill_value=0)
            fig = px.bar(x=relapse.index, y=relapse.values, text=[f"{v:.0f}%" for v in relapse.values])
            fig.update_traces(marker_color="#F87171", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="% de rechute", yaxis_range=[0, 100])
            st.plotly_chart(fig, width="stretch")

        st.subheader("Rechute scolaire récente par catégorie diagnostique")
        valid = df.dropna(subset=["rechute_scolaire_recente"])
        school = valid.groupby("diag_cat")["rechute_scolaire_recente"].apply(
            lambda x: (x == "Oui").mean() * 100
        ).reindex([c for c in DIAG_ORDER if c in valid["diag_cat"].unique()])
        fig = px.bar(x=school.index, y=school.values, color=school.index,
                     color_discrete_map=color_map(school.index), text=[f"{v:.0f}%" for v in school.values])
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="% de rechute scolaire", yaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig, width="stretch")

# ==============================================================
# PAGE 5 — Synthese par profil
# ==============================================================
elif page == "Synthèse par profil":
    st.title("Synthèse par profil symptomatique")

    if df.empty:
        st.warning("Aucun patient ne correspond au filtre sélectionné.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(pie(df["diag_cat"], order=DIAG_ORDER, title="Répartition des profils"), width="stretch")
        with col2:
            profile_stats = df.groupby("diag_cat").agg(
                **{
                    "Âge moyen": (AGE_ONSET_COL, "mean"),
                    "Hospitalisations / patient": ("nb_hospitalisations_int", "mean"),
                    "% Risque élevé": ("high_risk", "mean"),
                    "% Traumatismes": ("type_traumatisme_emotionnel", lambda x: (x.notna() & (x != "Aucun")).mean()),
                }
            ).round(2)
            profile_stats["% Risque élevé"] = (profile_stats["% Risque élevé"] * 100).round(0)
            profile_stats["% Traumatismes"] = (profile_stats["% Traumatismes"] * 100).round(0)
            st.dataframe(profile_stats, width="stretch")

# ==============================================================
# PAGE 6 — Questions de recherche
# ==============================================================
elif page == "Questions de recherche":
    st.title("Réponses aux questions de recherche")

    if df.empty:
        st.warning("Aucun patient ne correspond au filtre sélectionné.")
    else:
        st.subheader("Rechute scolaire selon la sévérité des traumatismes")
        valid = df.dropna(subset=["severite_traumatisme", "rechute_scolaire_recente"])
        order_sev = ["Léger", "Modéré", "Sévère"]
        relapse = valid.groupby("severite_traumatisme")["rechute_scolaire_recente"].apply(lambda x: (x == "Oui").mean() * 100).reindex(order_sev, fill_value=0)
        fig = px.bar(x=relapse.index, y=relapse.values, text=[f"{v:.0f}%" for v in relapse.values])
        fig.update_traces(marker_color="#6366F1", textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="% de rechute scolaire", yaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")

        st.subheader("Score prédictif du risque suicidaire élevé")
        factors = {
            "Antécédents familiaux TS": df["atcd_familiaux_psy_trouble"].str.contains("Ts/Suicide", na=False),
            "Traumatismes": df["type_traumatisme_emotionnel"].notna() & (df["type_traumatisme_emotionnel"] != "Aucun"),
            "Abus substances": df["atcd_abus_spa"].notna() & (df["atcd_abus_spa"] != "Rien À Signaler") & (df["atcd_abus_spa"] != "Inconnu"),
            "Dépression sévère": df["diagnostic_principal"].str.contains("Dépressif", na=False),
            "Isolement social": df["situation_residentielle"].str.contains("Seul|Colocataires", na=False),
        }
        scores = {name: df[cond]["high_risk"].mean() * 100 for name, cond in factors.items() if cond.any()}
        if scores:
            max_val = max(scores.values()) or 1
            scores_norm = {k: v / max_val * 10 for k, v in scores.items()}
            fig = px.bar(x=list(scores_norm.values()), y=list(scores_norm.keys()), orientation="h", text=[f"{v:.1f}" for v in scores_norm.values()])
            fig.update_traces(marker_color="#991B1B", textposition="outside")
            fig.update_layout(xaxis_title="Score (/10)", yaxis_title="", xaxis_range=[0, 10])
            st.plotly_chart(fig, width="stretch")

        st.subheader("Amélioration selon les facteurs favorables")
        factors2 = {
            "Observance": df["observance_derniere_consult"] == "Bonne",
            "Bonne tolérance": df["tolerance_derniere_consult"] == "Bonne",
            "Soutien social": df["situation_familiale"].str.contains("Conjoint|Famille", na=False),
            "Absence de trauma": df["type_traumatisme_emotionnel"].isna() | (df["type_traumatisme_emotionnel"] == "Aucun"),
        }
        improvement = {name: df[cond]["ameliorated"].mean() * 100 for name, cond in factors2.items() if cond.any()}
        if improvement:
            fig = px.bar(x=list(improvement.keys()), y=list(improvement.values()), text=[f"{v:.0f}%" for v in improvement.values()])
            fig.update_traces(marker_color="#EAB308", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="% amélioré", yaxis_range=[0, 100])
            st.plotly_chart(fig, width="stretch")

# ==============================================================
# PAGE 7 — Profil type par categorie (aucune generation aleatoire)
# ==============================================================
elif page == "Profil type par catégorie":
    st.title("Profil type par catégorie")
    st.caption(
        "Pour chaque catégorie diagnostique : les valeurs les plus probables ou moyennes parmi les patients "
        "de cette catégorie (voir `profiling_model.py`). Calculé sur les 7 catégories cliniques, hors « Autre », "
        "indépendamment du filtre global."
    )

    for cat in model.classes_:
        n = model.class_counts_[cat]
        if n < 20:
            st.warning(f"⚠️ Catégorie **{cat}** : seulement {n} patients dans les données, profil peu fiable statistiquement.")

    category = st.selectbox("Catégorie diagnostique", options=model.classes_)
    stats = model.class_stats_[category]
    n = model.class_counts_[category]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients dans les données", n)
    c2.metric("Âge moyen (apparition)", f"{stats[AGE_ONSET_COL]['mean']:.0f} ans")
    c3.metric("Âge moyen (1ère consult.)", f"{stats[AGE_CONSULT_COL]['mean']:.0f} ans")
    sexe_top = max(stats[SEXE_COL], key=stats[SEXE_COL].get)
    c4.metric("Sexe majoritaire", f"{sexe_top} ({stats[SEXE_COL][sexe_top]*100:.0f}%)")

    edu_probs = stats[EDU_COL]
    edu_df = pd.DataFrame({"Niveau scolaire": list(edu_probs.keys()),
                           "Probabilité (%)": [round(v * 100, 1) for v in edu_probs.values()]}
                          ).sort_values("Probabilité (%)", ascending=False)
    st.plotly_chart(px.bar(edu_df, x="Niveau scolaire", y="Probabilité (%)", title="Distribution du niveau scolaire"),
                    width="stretch")

    symptom_df = pd.DataFrame({"Symptôme": [pretty(c) for c in SYMPTOM_COLS],
                               "Probabilité (%)": [stats[c] * 100 for c in SYMPTOM_COLS]}
                              ).sort_values("Probabilité (%)", ascending=True)
    fig = px.bar(symptom_df, x="Probabilité (%)", y="Symptôme", orientation="h",
                 title="Probabilité de présence de chaque symptôme")
    fig.add_vline(x=50, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")

    with st.expander("📝 Description textuelle du profil type"):
        st.text(model.describe(category))
    with st.expander("ℹ️ À propos"):
        st.markdown(
            "- Chaque variable (âge, sexe, niveau scolaire, symptômes) est modélisée **indépendamment**, "
            "sachant la catégorie.\n"
            "- Un **lissage** vers la fréquence globale évite que les petites catégories (TOC, Personnalité : n ≈ 15) "
            "ne produisent des probabilités instables (0 % / 100 %).\n"
            "- Le profil type est une **synthèse de valeurs probables**, pas un patient réel et pas un patient tiré au hasard.\n"
            "- Limite : les corrélations entre symptômes ne sont pas capturées."
        )

# ==============================================================
# PAGE 8 — Classification d'un patient (categorie + explication)
# ==============================================================
elif page == "Classification d'un patient":
    try:
        clf = get_classifier()
    except Exception as exc:   # fichier televerse au mauvais format (colonnes manquantes ou ajoutees)
        st.title("Classification d'un patient")
        st.error("Impossible d'entraîner le modèle avec le fichier de données chargé. Il doit être la sortie de "
                 "`preprocessing.py` (colonnes d'origine, sans colonne ajoutée). "
                 f"Détail : {type(exc).__name__} : {str(exc)[:250]}")
        st.stop()
    perf = clf.performance
    f1_psy = perf["f1_par_classe"].get("Troubles psychotiques", float("nan"))
    n_psy = perf["n_par_classe"].get("Troubles psychotiques", 0)

    st.title("Classification d'un patient")
    st.caption("Saisissez les informations d'un dossier : le modèle propose une catégorie parmi 3 et explique sa décision.")
    st.warning(
        "**Outil de recherche, pas un outil de diagnostic.**\n\n"
        "- Le modèle ne connaît que **3 catégories** (anxiété / stress, humeur, psychotiques). Un patient d'une autre "
        "catégorie sera quand même classé dans l'une d'elles, **sans signal d'alerte** : la confiance ne permet pas de le détecter.\n"
        "- Les variables viennent de dossiers **annotés en connaissant le diagnostic**. Les "
        f"{len(clf.features)} variables retenues sont toutes issues du tableau clinique documenté : la performance mesure "
        "la cohérence du dossier avec lui-même, pas une capacité à diagnostiquer à partir d'observations indépendantes.\n"
        f"- Catégorie psychotique : seulement {n_psy} patients (F1 ≈ {f1_psy:.2f}) : à interpréter avec prudence."
    )

    keys = {sp["source"]: f"in_{sp['source']}" for sp in clf.inputs}
    NEUTRAL = "— (valeurs neutres)"

    def set_inputs(values):
        for src, key in keys.items():
            v = values[src]
            st.session_state[key] = list(v) if isinstance(v, list) else v

    def apply_prefill():
        choice = st.session_state["prefill"]
        set_inputs(clf.default_inputs() if choice == NEUTRAL else clf.typical_inputs(choice))

    for sp in clf.inputs:
        if keys[sp["source"]] not in st.session_state:
            v = sp["default"]
            st.session_state[keys[sp["source"]]] = list(v) if isinstance(v, list) else v

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Informations du dossier")
        st.selectbox("Pré-remplir avec le profil type de…", [NEUTRAL] + clf.classes, key="prefill",
                     on_change=apply_prefill,
                     help="Remplit le formulaire avec les valeurs les plus représentatives de la catégorie choisie.")
        st.caption("Seules les informations utilisées par le modèle sont demandées (classées par importance). "
                   "« Non » ou « aucun » = non mentionné dans le dossier.")
        for sp in clf.inputs:
            key = keys[sp["source"]]
            if sp["kind"] == "yes_no":
                st.radio(sp["label"], sp["choices"], key=key, horizontal=True)
            elif sp["kind"] in ("choice", "ordinal"):
                st.selectbox(sp["label"], sp["choices"], key=key, format_func=pretty)
            elif sp["kind"] == "multi":
                st.multiselect(sp["label"], sp["choices"], key=key, format_func=pretty,
                               help="Laisser vide si aucune modalité n'est mentionnée.")
            else:
                st.number_input(sp["label"], min_value=sp["min"], max_value=sp["max"], step=1, key=key)

    res = clf.predict({src: st.session_state[key] for src, key in keys.items()})
    best, second = res["classe"], res["deuxieme_classe"]

    with right:
        st.subheader("Résultat")
        st.header(best)
        m1, m2 = st.columns(2)
        m1.metric("Confiance (calibrée)", f"{res['confiance'] * 100:.0f} %")
        m2.metric("Deuxième catégorie", f"{res['probas'][second] * 100:.0f} %", help=second)
        if res["incertain"]:
            st.warning(f"**Confiance faible** (< {UNCERTAIN_BELOW:.0%}) : cas ambigu entre « {best} » et « {second} ». "
                       "À considérer comme indécis.")

        order = sorted(res["probas"], key=res["probas"].get)
        fig = go.Figure(go.Bar(
            x=[res["probas"][c] * 100 for c in order], y=order, orientation="h",
            marker_color=["#1D4ED8" if c == best else "#94A3B8" for c in order],
            text=[f"{res['probas'][c] * 100:.0f} %" for c in order], textposition="outside"))
        fig.update_layout(title="Probabilité de chaque catégorie", xaxis_range=[0, 105], xaxis_title="%",
                          height=230, margin=dict(l=0, r=10, t=40, b=30))
        st.plotly_chart(fig, width="stretch")

        b = res["bande"]
        st.info(f"**Fiabilité observée.** En validation croisée, quand la confiance affichée était entre "
                f"{b['low']:.0%} et {b['high']:.0%} ({b['share']:.0%} des cas), la catégorie proposée était correcte "
                f"dans **{b['accuracy']:.0%}** des cas.")

    st.subheader("Pourquoi cette catégorie ?")
    odds = float(np.exp(res["log_rapport"]))
    st.markdown(
        f"Le modèle compare « **{best}** » à la deuxième catégorie « {second} ». Point de départ (fréquence de chaque "
        f"catégorie dans les données) : **{res['terme_prior']:+.1f}** ; éléments du dossier : "
        f"**{res['log_rapport'] - res['terme_prior']:+.1f}** ; total : **{res['log_rapport']:+.1f}** → "
        f"« {best} » est environ **{odds:.1f} fois** plus probable que « {second} »."
    )
    expl = res["explication"].copy()
    expl["abs"] = expl["contribution"].abs()
    expl = expl.sort_values("abs", ascending=True)
    fig = go.Figure(go.Bar(
        x=expl["contribution"], y=[f"{v} = {val}" for v, val in zip(expl["variable"], expl["valeur_saisie"])],
        orientation="h", marker_color=["#16A34A" if c >= 0 else "#EA580C" for c in expl["contribution"]],
        text=[f"{c:+.2f}" for c in expl["contribution"]], textposition="outside"))
    fig.update_layout(title=f"Contribution de chaque variable : vers « {best} » (+) ou vers « {second} » (−)",
                      xaxis_title="Contribution au log du rapport de probabilités",
                      height=120 + 38 * len(expl), margin=dict(l=0, r=30, t=60, b=40))
    st.plotly_chart(fig, width="stretch")

    with st.expander("📋 Détail par variable"):
        st.dataframe(res["explication"].rename(columns={"variable": "Variable", "valeur_saisie": "Valeur saisie",
                                                         "contribution": "Contribution", "niveau": "Niveau"}),
                     width="stretch", hide_index=True)
        st.caption("Niveau A = anamnèse factuelle ; niveau B = tableau clinique documenté avec le diagnostic en tête.")

    with st.expander("ℹ️ À propos du modèle"):
        st.markdown(
            f"- **Modèle** : Naive Bayes sur les {len(clf.features)} variables les mieux classées (chi² + information mutuelle), "
            "classement refait à chaque pli de validation. Probabilités **calibrées** (température "
            f"T = {clf.calibration['T']:.2f}) : le modèle brut est trop confiant.\n"
            f"- **Performance** (validation croisée sur {perf['n']} patients, {perf['repetitions']} répétitions) : "
            f"macro-F1 ≈ **{perf['macro_f1']:.2f}**. Estimation légèrement optimiste (le nombre de variables a été choisi "
            "après examen des résultats) ; l'étude comparative avec validation imbriquée donne ≈ 0,76.\n"
            "- **Variables factuelles seules** (niveau A) : macro-F1 ≈ 0,40 contre 0,32 pour le hasard : insuffisant.\n"
            "- Comparé à un Random Forest : aucune différence significative (étude `model_comparison.py`)."
        )
        perf_df = pd.DataFrame({"Catégorie": list(perf["f1_par_classe"]),
                                "Patients": [perf["n_par_classe"][c] for c in perf["f1_par_classe"]],
                                "F1": list(perf["f1_par_classe"].values())})
        st.dataframe(perf_df, width="stretch", hide_index=True)
        bands = pd.DataFrame([{"Confiance affichée": f"{x['low']:.0%} – {x['high']:.0%}", "Part des cas": f"{x['share']:.0%}",
                               "Justesse observée": f"{x['accuracy']:.0%}"} for x in clf.bands])
        st.dataframe(bands, width="stretch", hide_index=True)
        st.markdown("**Variables utilisées** (par ordre d'importance) :")
        st.markdown("\n".join(f"{i + 1}. {clf._label(clf.feature_info[f])} (niveau {clf.feature_info[f]['tier']})"
                               for i, f in enumerate(clf.features)))

st.sidebar.markdown("---")
st.sidebar.caption("Projet d'analyse de données cliniques psychiatriques")