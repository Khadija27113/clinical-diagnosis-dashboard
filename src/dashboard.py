"""
dashboard_app.py
Tableau de bord Streamlit — synthese finale du projet.

Reunit dans une seule interface interactive :
  1. Les analyses exploratoires (exploratory_analysis.py)
  2. Le modele de profilage generatif (profiling_model.py)

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
    DIAG_MAP,
    DATA_PATH,
    train_model,
)

# ==============================================================
# Configuration generale
# ==============================================================
st.set_page_config(
    page_title="Tableau de bord clinique — Analyse psychiatrique",
    page_icon="🧠",
    layout="wide",
)

PALETTE = {
    "Psychotique": "#7C3AED",
    "Bipolaire": "#F59E0B",
    "Anxio-Dépressif": "#0EA5E9",
    "Autre": "#94A3B8",
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

    df["diag_cat"] = df["diagnostic_principal"].map(DIAG_MAP).fillna("Autre")

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


df_full = load_data()
model = get_profiling_model()

DIAG_ORDER = ["Psychotique", "Bipolaire", "Anxio-Dépressif", "Autre"]


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
        "Modèle de profilage génératif",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtre global")
selected_cats = st.sidebar.multiselect(
    "Catégories diagnostiques incluses",
    options=DIAG_ORDER,
    default=DIAG_ORDER,
    help="S'applique à toutes les sections sauf 'Modèle de profilage génératif' "
    "(le modèle reste entraîné sur l'ensemble des données pour rester fiable).",
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
        c4.metric("Hospitalisations moyennes", f"{df['nb_hospitalisations_int'].mean():.1f}")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                pie(df["diag_cat"], order=DIAG_ORDER, title="Répartition des catégories diagnostiques"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                pie(df[SEXE_COL], title="Répartition par sexe"),
                use_container_width=True,
            )

        st.markdown(
            "Ce tableau de bord reprend l'ensemble du pipeline du projet : nettoyage des exports "
            "Label Studio → prétraitement/fusion → analyse exploratoire → modèle de profilage "
            "génératif. Utilisez le menu à gauche pour naviguer entre les sections, et le filtre "
            "global pour restreindre l'analyse à une ou plusieurs catégories diagnostiques."
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
            st.plotly_chart(pie(df["niveau_scolaire"], title="Niveau scolaire"), use_container_width=True)
        with col2:
            st.plotly_chart(pie(df["niveau_socioeconomique"], title="Niveau socio-économique"), use_container_width=True)

        st.subheader("Redoublements moyens par catégorie diagnostique")
        redoub = df.groupby("diag_cat")["nombre_redoublements"].mean().reindex(
            [c for c in DIAG_ORDER if c in df["diag_cat"].unique()]
        )
        fig = px.bar(x=redoub.index, y=redoub.values, color=redoub.index, color_discrete_map=color_map(redoub.index))
        fig.update_layout(xaxis_title="", yaxis_title="Nombre moyen de redoublements", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(pie(df["niveau_suicidalite"], order=order_suic, title="Niveau de suicidalité"), use_container_width=True)
        with col2:
            st.plotly_chart(pie(df["substance_cat"], title="Consommation de substances psychoactives"), use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Hospitalisations moyennes par catégorie diagnostique")
        hosp = df.groupby("diag_cat")["nb_hospitalisations_int"].mean().reindex(
            [c for c in DIAG_ORDER if c in df["diag_cat"].unique()]
        )
        fig = px.bar(x=hosp.index, y=hosp.values, color=hosp.index, color_discrete_map=color_map(hosp.index))
        fig.update_layout(xaxis_title="", yaxis_title="Moyenne d'hospitalisations", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# PAGE 4 — Suivi therapeutique
# ==============================================================
elif page == "Suivi thérapeutique":
    st.title("Suivi thérapeutique")

    if df.empty:
        st.warning("Aucun patient ne correspond au filtre sélectionné.")
    else:
        order_evol = ["Amélioration", "Amélioration partielle", "Stable", "Symptômes persistants", "Aggravation", "Autre", "Non renseigné"]
        st.plotly_chart(bar_pct(df["evolution"], order=order_evol, title="Évolution clinique (1ère → dernière consultation)", color="#22C55E"), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            order_obs = ["Bonne", "Moyenne", "Mauvaise", "Naif_Traitement", "Arret_Traitement"]
            st.plotly_chart(pie(df["observance_derniere_consult"], order=order_obs, title="Observance du traitement"), use_container_width=True)
        with col2:
            st.plotly_chart(pie(df["tolerance_derniere_consult"], title="Tolérance au traitement"), use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.subheader("Rechute à 12 mois selon la sévérité du traumatisme")
            valid = df.dropna(subset=["severite_traumatisme", "rechute_12_mois"])
            order_sev = ["Léger", "Modéré", "Sévère"]
            relapse = valid.groupby("severite_traumatisme")["rechute_12_mois"].apply(lambda x: (x == "Oui").mean() * 100).reindex(order_sev, fill_value=0)
            fig = px.bar(x=relapse.index, y=relapse.values, text=[f"{v:.0f}%" for v in relapse.values])
            fig.update_traces(marker_color="#F87171", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="% de rechute", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(pie(df["diag_cat"], order=DIAG_ORDER, title="Répartition des profils"), use_container_width=True)
        with col2:
            profile_stats = df.groupby("diag_cat").agg(
                **{
                    "Âge moyen": (AGE_ONSET_COL, "mean"),
                    "Hospitalisations moyennes": ("nb_hospitalisations_int", "mean"),
                    "% Risque élevé": ("high_risk", "mean"),
                    "% Traumatismes": ("type_traumatisme_emotionnel", lambda x: (x.notna() & (x != "Aucun")).mean()),
                }
            ).round(2)
            profile_stats["% Risque élevé"] = (profile_stats["% Risque élevé"] * 100).round(0)
            profile_stats["% Traumatismes"] = (profile_stats["% Traumatismes"] * 100).round(0)
            st.dataframe(profile_stats, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# PAGE 7 — Modele de profilage generatif
# ==============================================================
elif page == "Modèle de profilage génératif":
    st.title("Modèle de profilage génératif")
    st.caption(
        "Modèle Naive Bayes conditionné sur la catégorie diagnostique. "
        "Entraîné sur l'ensemble des 3 catégories cibles, indépendamment du filtre global."
    )

    for cat in model.classes_:
        n = model.class_counts_[cat]
        if n < 20:
            st.warning(f"⚠️ Catégorie **{cat}** : seulement {n} patients dans les données — profils peu fiables statistiquement.")

    category = st.selectbox("Catégorie diagnostique", options=model.classes_)
    mode = st.radio("Mode", ["Profil type (déterministe)", "Générer des profils synthétiques (aléatoire)"], horizontal=True)

    if mode == "Profil type (déterministe)":
        st.markdown(f"### Profil type — {category}")
        stats = model.class_stats_[category]
        n = model.class_counts_[category]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients dans les données", n)
        c2.metric("Âge moyen (apparition)", f"{stats[AGE_ONSET_COL]['mean']:.0f} ans")
        c3.metric("Âge moyen (1ère consult.)", f"{stats[AGE_CONSULT_COL]['mean']:.0f} ans")
        sexe_top = max(stats[SEXE_COL], key=stats[SEXE_COL].get)
        c4.metric("Sexe majoritaire", f"{sexe_top} ({stats[SEXE_COL][sexe_top]*100:.0f}%)")

        edu_probs = stats[EDU_COL]
        edu_df = pd.DataFrame({"Niveau scolaire": list(edu_probs.keys()), "Probabilité (%)": [round(v * 100, 1) for v in edu_probs.values()]}).sort_values("Probabilité (%)", ascending=False)
        st.plotly_chart(px.bar(edu_df, x="Niveau scolaire", y="Probabilité (%)", title="Distribution du niveau scolaire"), use_container_width=True)

        symptom_probs = {col.replace("_", " "): stats[col] * 100 for col in SYMPTOM_COLS}
        symptom_df = pd.DataFrame({"Symptôme": list(symptom_probs.keys()), "Probabilité (%)": list(symptom_probs.values())}).sort_values("Probabilité (%)", ascending=True)
        fig = px.bar(symptom_df, x="Probabilité (%)", y="Symptôme", orientation="h", title="Probabilité de présence de chaque symptôme")
        fig.add_vline(x=50, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    else:
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Nombre de profils à générer", 1, 20, 5)
        with col2:
            use_seed = st.checkbox("Fixer une graine aléatoire (reproductibilité)")
            seed = st.number_input("Graine", value=42, step=1, disabled=not use_seed) if use_seed else None

        if st.button("🎲 Générer les profils", type="primary"):
            profiles = model.sample(category, n=n_samples, seed=seed)
            st.dataframe(profiles, use_container_width=True)
            st.download_button(
                "Télécharger en CSV",
                data=profiles.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"profils_synthetiques_{category}.csv",
                mime="text/csv",
            )

    with st.expander("ℹ️ À propos du modèle"):
        st.markdown(
            "- Chaque variable (âge, sexe, niveau scolaire, symptômes) est modélisée "
            "**indépendamment**, conditionnellement à la catégorie diagnostique (hypothèse Naive Bayes).\n"
            "- Un **lissage** (paramètres `alpha` et `age_shrink_k`) évite que les catégories à faible "
            "effectif (Psychotique, n=10) ne produisent des probabilités instables (0%/100%).\n"
            "- Le mode *aléatoire* tire un échantillon selon les distributions apprises ; le mode "
            "*déterministe* affiche la valeur la plus probable/moyenne pour chaque variable.\n"
            "- Limite : les corrélations entre symptômes ne sont pas capturées (voir la discussion "
            "dans le rapport)."
        )

st.sidebar.markdown("---")
st.sidebar.caption("Projet d'analyse de données cliniques psychiatriques")