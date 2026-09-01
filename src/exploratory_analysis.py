# exploratory_analysis.py
# Analyse exploratoire complète des données cliniques et psychiatriques
# Génère des graphiques pour le tableau de bord

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Configuration des styles
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16

# Dossiers
OUTPUT_DIR = "../reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chargement des données
def load_data():
    data_path = "../data/processed/merged_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier {data_path} introuvable. Lancez d'abord preprocessing.py.")
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"Données chargées : {len(df)} lignes, {len(df.columns)} colonnes")
    return df

df = load_data()

# ------------------------------------------------------------
# 0. Préparation : création de colonnes supplémentaires
# ------------------------------------------------------------

# Créer une catégorie diagnostique simplifiée
diag_map = {
    'Schizophrénie': 'Psychotique',
    'Trouble Schizo-Affectif': 'Psychotique',
    'Trouble Psychotique Aigu Et Transitoire': 'Psychotique',
    'Trouble Bipolaire Type I': 'Bipolaire',
    'Trouble Bipolaire Type Ii': 'Bipolaire',
    'Cyclothymie': 'Bipolaire',
    'Trouble Dépressif Caractérisé': 'Anxio-Dépressif',
    'Trouble Dépressif Récurrent': 'Anxio-Dépressif',
    'Trouble Dépressif Persistant': 'Anxio-Dépressif',
    'Trouble Anxieux Généralisé': 'Anxio-Dépressif',
    'Trouble Panique': 'Anxio-Dépressif',
    'Phobie Sociale': 'Anxio-Dépressif',
    'Trouble Obsessionnel-Compulsif (Toc)': 'Anxio-Dépressif',
    'État De Stress Post-Traumatique (Tspt)': 'Anxio-Dépressif',
    'Trouble De L\'Adaptation': 'Anxio-Dépressif',
}
df['diag_cat'] = df['diagnostic_principal'].map(diag_map).fillna('Autre')

# Créer colonne de risque élevé (suicidalité)
df['high_risk'] = df['niveau_suicidalite'].isin(['Idees_Suicidaires_Actives', 'Tentative_Ou_Automutilation'])

# Créer colonne d'évolution clinique
def eval_evolution(row):
    first = row.get('etat_clinique_premiere_consult', None)
    last = row.get('etat_clinique_derniere_consult', None)
    if pd.isna(first) or pd.isna(last):
        return 'Non renseigné'
    if last == 'Stabilise' and first != 'Stabilise':
        return 'Amélioration'
    elif last == 'Stabilise' and first == 'Stabilise':
        return 'Stable'
    elif last == 'Amelioration_Partielle':
        return 'Amélioration partielle'
    elif last == 'Aggravation_Recente':
        return 'Aggravation'
    elif last == 'Symptomes_Actifs':
        if first == 'Symptomes_Actifs':
            return 'Symptômes persistants'
        else:
            return 'Aggravation'
    else:
        return 'Autre'
df['evolution'] = df.apply(eval_evolution, axis=1)

# Créer une colonne amélioration pour les analyses
df['ameliorated'] = df['evolution'].isin(['Amélioration', 'Amélioration partielle', 'Stable'])

# ------------------------------------------------------------
# 1. PROFIL ACADÉMIQUE ET SOCIAL
# ------------------------------------------------------------

def plot_education():
    edu_counts = df['niveau_scolaire'].value_counts()
    plt.figure()
    plt.pie(edu_counts, labels=edu_counts.index, autopct='%1.0f%%', startangle=90)
    plt.title("Niveau scolaire des patients")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/education_pie.png", dpi=150)
    plt.close()
    print("✅ Graphique 'education_pie.png' généré")

def plot_redoublements_by_diagnosis():
    mean_redoub = df.groupby('diag_cat')['nombre_redoublements'].mean().sort_values(ascending=False)
    plt.figure()
    mean_redoub.plot(kind='bar', color='skyblue')
    plt.title("Nombre moyen de redoublements par catégorie diagnostique")
    plt.ylabel("Moyenne des redoublements")
    plt.xlabel("Catégorie diagnostique")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/redoublements_by_diag.png", dpi=150)
    plt.close()
    print("✅ Graphique 'redoublements_by_diag.png' généré")

def plot_socioeco():
    socio_counts = df['niveau_socioeconomique'].value_counts()
    plt.figure()
    plt.pie(socio_counts, labels=socio_counts.index, autopct='%1.0f%%', startangle=90)
    plt.title("Niveau socio-économique des patients")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/socioeco_pie.png", dpi=150)
    plt.close()
    print("✅ Graphique 'socioeco_pie.png' généré")

# ------------------------------------------------------------
# 2. INDICATEURS DE RISQUE
# ------------------------------------------------------------

def plot_suicidality():
    suic_counts = df['niveau_suicidalite'].value_counts()
    order = ['Aucune_Ideation', 'Idees_Mort_Passives', 'Idees_Suicidaires_Actives', 'Tentative_Ou_Automutilation']
    suic_counts = suic_counts.reindex(order, fill_value=0)
    plt.figure()
    plt.pie(suic_counts, labels=suic_counts.index, autopct='%1.0f%%', startangle=90)
    plt.title("Niveau de suicidalité des patients")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/suicidality_pie.png", dpi=150)
    plt.close()
    print("✅ Graphique 'suicidality_pie.png' généré")

def plot_risk_factors():
    factors = {
        'Antécédents familiaux': df['atcd_familiaux_psy_trouble'].notna() & (df['atcd_familiaux_psy_trouble'] != 'Rien À Signaler'),
        'Traumatisme': df['type_traumatisme_emotionnel'].notna() & (df['type_traumatisme_emotionnel'] != 'Aucun'),
        'Abus substances': df['atcd_abus_spa'].notna() & (df['atcd_abus_spa'] != 'Rien À Signaler') & (df['atcd_abus_spa'] != 'Inconnu'),
        'Conflits familiaux': df['situation_familiale'].str.contains('Conflit|Divorcé', na=False)
    }
    risk_rates = {}
    for name, condition in factors.items():
        if condition.any():
            rate = df[condition]['high_risk'].mean() * 100
            risk_rates[name] = rate
    plt.figure()
    plt.bar(risk_rates.keys(), risk_rates.values(), color='coral')
    plt.title("Proportion de patients à risque suicidaire élevé selon les facteurs")
    plt.ylabel("% avec risque élevé")
    plt.ylim(0, 100)
    for i, v in enumerate(risk_rates.values()):
        plt.text(i, v+2, f"{v:.0f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/risk_factors.png", dpi=150)
    plt.close()
    print("✅ Graphique 'risk_factors.png' généré")

def plot_hospitalizations():
    hosp_mean = df.groupby('diag_cat')['nb_hospitalisations_int'].mean().sort_values(ascending=False)
    plt.figure()
    hosp_mean.plot(kind='bar', color='lightgreen')
    plt.title("Nombre moyen d'hospitalisations par catégorie diagnostique")
    plt.ylabel("Moyenne d'hospitalisations")
    plt.xlabel("Catégorie diagnostique")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hospitalizations_by_diag.png", dpi=150)
    plt.close()
    print("✅ Graphique 'hospitalizations_by_diag.png' généré")

def plot_substance_use():
    def classify_substance(s):
        if pd.isna(s) or s == 'Rien À Signaler':
            return 'Aucune'
        if s == 'Inconnu':
            return 'Inconnu'
        substances = [x.strip() for x in s.split(';')]
        if len(substances) >= 2:
            return 'Polysubstances'
        elif 'Tabac' in substances:
            return 'Tabac'
        elif 'Alcool' in substances:
            return 'Alcool'
        elif 'Cannabis' in substances:
            return 'Cannabis'
        else:
            return 'Autre'
    df['substance_cat'] = df['atcd_abus_spa'].apply(classify_substance)
    sub_counts = df['substance_cat'].value_counts()
    plt.figure()
    plt.pie(sub_counts, labels=sub_counts.index, autopct='%1.0f%%', startangle=90)
    plt.title("Consommation de substances psychoactives")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/substance_pie.png", dpi=150)
    plt.close()
    print("✅ Graphique 'substance_pie.png' généré")

# ------------------------------------------------------------
# 3. SUIVI THÉRAPEUTIQUE
# ------------------------------------------------------------

def plot_clinical_evolution():
    evol_counts = df['evolution'].value_counts()
    order = ['Amélioration', 'Amélioration partielle', 'Stable', 'Symptômes persistants', 'Aggravation', 'Autre', 'Non renseigné']
    evol_counts = evol_counts.reindex(order, fill_value=0)
    plt.figure()
    evol_counts.plot(kind='bar', color=['green', 'lightgreen', 'blue', 'orange', 'red', 'gray', 'lightgray'])
    plt.title("Évolution de l'état clinique entre la première et la dernière consultation")
    plt.ylabel("Nombre de patients")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clinical_evolution.png", dpi=150)
    plt.close()
    print("✅ Graphique 'clinical_evolution.png' généré")

def plot_observance():
    obs_counts = df['observance_derniere_consult'].value_counts()
    order = ['Bonne', 'Moyenne', 'Mauvaise', 'Naif_Traitement', 'Arret_Traitement']
    obs_counts = obs_counts.reindex(order, fill_value=0)
    plt.figure()
    plt.pie(obs_counts, labels=obs_counts.index, autopct='%1.0f%%', startangle=90)
    plt.title("Observance du traitement (dernière consultation)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/observance_pie.png", dpi=150)
    plt.close()
    print("✅ Graphique 'observance_pie.png' généré")

def plot_observance_impact():
    valid = df.dropna(subset=['evolution', 'observance_derniere_consult'])
    valid = valid[~valid['evolution'].isin(['Non renseigné', 'Autre'])]
    obs_impact = valid.groupby('observance_derniere_consult')['ameliorated'].mean() * 100
    order = ['Bonne', 'Moyenne', 'Mauvaise']
    obs_impact = obs_impact.reindex(order, fill_value=0)
    plt.figure()
    obs_impact.plot(kind='bar', color='teal')
    plt.title("Proportion de patients améliorés/stabilisés selon l'observance")
    plt.ylabel("% de patients en amélioration ou stable")
    plt.ylim(0, 100)
    for i, v in enumerate(obs_impact.values):
        plt.text(i, v+2, f"{v:.0f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/observance_impact.png", dpi=150)
    plt.close()
    print("✅ Graphique 'observance_impact.png' généré")

def plot_tolerance():
    tol_counts = df['tolerance_derniere_consult'].value_counts()
    plt.figure()
    plt.pie(tol_counts, labels=tol_counts.index, autopct='%1.0f%%', startangle=90)
    plt.title("Tolérance au traitement (dernière consultation)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tolerance_pie.png", dpi=150)
    plt.close()
    print("✅ Graphique 'tolerance_pie.png' généré")

def plot_relapse_by_trauma():
    valid = df.dropna(subset=['severite_traumatisme', 'rechute_12_mois'])
    relapse_rate = valid.groupby('severite_traumatisme')['rechute_12_mois'].apply(lambda x: (x == 'Oui').mean() * 100)
    order = ['Léger', 'Modéré', 'Sévère']
    relapse_rate = relapse_rate.reindex(order, fill_value=0)
    plt.figure()
    relapse_rate.plot(kind='bar', color='salmon')
    plt.title("Taux de rechute à 12 mois selon la sévérité du traumatisme")
    plt.ylabel("% de rechute")
    plt.ylim(0, 100)
    for i, v in enumerate(relapse_rate.values):
        plt.text(i, v+2, f"{v:.0f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/relapse_by_trauma.png", dpi=150)
    plt.close()
    print("✅ Graphique 'relapse_by_trauma.png' généré")

# ------------------------------------------------------------
# 4. SYNTHÈSE PAR PROFIL DE PATIENT
# ------------------------------------------------------------

def plot_profiles():
    profile_counts = df['diag_cat'].value_counts()
    plt.figure()
    plt.pie(profile_counts, labels=profile_counts.index, autopct='%1.0f%%', startangle=90)
    plt.title("Répartition des profils symptomatiques")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/profiles_pie.png", dpi=150)
    plt.close()
    print("✅ Graphique 'profiles_pie.png' généré")

    # Statistiques par profil
    profile_stats = df.groupby('diag_cat').agg({
        'age_apparition_premiers_symptomes': 'mean',
        'nb_hospitalisations_int': 'mean',
        'high_risk': 'mean',
        'type_traumatisme_emotionnel': lambda x: (x.notna() & (x != 'Aucun')).mean()
    }).round(2)
    profile_stats.columns = ['Âge moyen', 'Hospitalisations moyennes', '% Risque élevé', '% Traumatismes']
    profile_stats.to_csv(f"{OUTPUT_DIR}/profile_stats.csv")
    print("✅ Tableau des profils sauvegardé dans 'profile_stats.csv'")

# ------------------------------------------------------------
# 5. RÉPONSES AUX QUESTIONS DE RECHERCHE
# ------------------------------------------------------------

def plot_trauma_school_relapse():
    valid = df.dropna(subset=['severite_traumatisme', 'rechute_scolaire_recente'])
    relapse_rate = valid.groupby('severite_traumatisme')['rechute_scolaire_recente'].apply(lambda x: (x == 'Oui').mean() * 100)
    order = ['Léger', 'Modéré', 'Sévère']
    relapse_rate = relapse_rate.reindex(order, fill_value=0)
    plt.figure()
    relapse_rate.plot(kind='bar', color='indigo')
    plt.title("Rechute scolaire selon la sévérité des traumatismes")
    plt.ylabel("% de rechute scolaire")
    plt.ylim(0, 100)
    for i, v in enumerate(relapse_rate.values):
        plt.text(i, v+2, f"{v:.0f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/trauma_school_relapse.png", dpi=150)
    plt.close()
    print("✅ Graphique 'trauma_school_relapse.png' généré")

def plot_suicide_predictors():
    factors = {
        'Antécédents familiaux TS': (df['atcd_familiaux_psy_trouble'].str.contains('Ts/Suicide', na=False)),
        'Traumatismes': (df['type_traumatisme_emotionnel'].notna() & (df['type_traumatisme_emotionnel'] != 'Aucun')),
        'Abus substances': (df['atcd_abus_spa'].notna() & (df['atcd_abus_spa'] != 'Rien À Signaler') & (df['atcd_abus_spa'] != 'Inconnu')),
        'Dépression sévère': (df['diagnostic_principal'].str.contains('Dépressif', na=False)),
        'Isolement social': (df['situation_residentielle'].str.contains('Seul|Colocataires', na=False))
    }
    scores = {}
    for name, cond in factors.items():
        if cond.any():
            rate = df[cond]['high_risk'].mean() * 100
            scores[name] = rate
    if not scores:
        print("Aucun facteur pour le score prédictif")
        return
    max_val = max(scores.values())
    scores_norm = {k: v/max_val*10 for k,v in scores.items()}
    plt.figure()
    plt.barh(list(scores_norm.keys()), list(scores_norm.values()), color='darkred')
    plt.title("Score prédictif du risque suicidaire élevé")
    plt.xlabel("Score (/10)")
    plt.xlim(0, 10)
    for i, v in enumerate(scores_norm.values()):
        plt.text(v+0.2, i, f"{v:.1f}", va='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/suicide_predictors.png", dpi=150)
    plt.close()
    print("✅ Graphique 'suicide_predictors.png' généré")

def plot_hosp_by_profile():
    hosp_rate = df.groupby('diag_cat')['nb_hospitalisations_int'].mean().sort_values(ascending=False)
    plt.figure()
    hosp_rate.plot(kind='bar', color='teal')
    plt.title("Nombre moyen d'hospitalisations par profil symptomatique")
    plt.ylabel("Moyenne d'hospitalisations")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hosp_by_profile.png", dpi=150)
    plt.close()
    print("✅ Graphique 'hosp_by_profile.png' généré")

def plot_improvement_factors():
    factors = {
        'Observance': (df['observance_derniere_consult'] == 'Bonne'),
        'Bonne tolérance': (df['tolerance_derniere_consult'] == 'Bonne'),
        'Soutien social': (df['situation_familiale'].str.contains('Conjoint|Famille', na=False)),
        'Absence de trauma': (df['type_traumatisme_emotionnel'].isna() | (df['type_traumatisme_emotionnel'] == 'Aucun'))
    }
    improvement_rates = {}
    for name, cond in factors.items():
        if cond.any():
            rate = df[cond]['ameliorated'].mean() * 100
            improvement_rates[name] = rate
    plt.figure()
    plt.bar(improvement_rates.keys(), improvement_rates.values(), color='gold')
    plt.title("Proportion de patients améliorés/stabilisés selon les facteurs")
    plt.ylabel("% améliorés")
    plt.ylim(0, 100)
    for i, v in enumerate(improvement_rates.values()):
        plt.text(i, v+2, f"{v:.0f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/improvement_factors.png", dpi=150)
    plt.close()
    print("✅ Graphique 'improvement_factors.png' généré")

# ------------------------------------------------------------
# EXÉCUTION
# ------------------------------------------------------------

def run_all():
    print("Génération des graphiques d'analyse exploratoire...")
    plot_education()
    plot_redoublements_by_diagnosis()
    plot_socioeco()
    plot_suicidality()
    plot_risk_factors()
    plot_hospitalizations()
    plot_substance_use()
    plot_clinical_evolution()
    plot_observance()
    plot_observance_impact()
    plot_tolerance()
    plot_relapse_by_trauma()
    plot_profiles()
    plot_trauma_school_relapse()
    plot_suicide_predictors()
    plot_hosp_by_profile()
    plot_improvement_factors()
    print(f"\n✅ Tous les graphiques ont été générés dans le dossier '{OUTPUT_DIR}'")
    print("Vous pouvez les utiliser pour alimenter votre tableau de bord.")

if __name__ == "__main__":
    run_all()