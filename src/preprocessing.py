import pandas as pd
import numpy as np
from datetime import datetime
import os

# ------------------------------------------------------------
# 0. Initialisation du journal
# ------------------------------------------------------------
log_entries = []
def log_message(msg):
    print(msg)
    log_entries.append(msg)

# ------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------
bio = pd.read_csv('../data/cleaned/bio_ATCD.csv')
symp = pd.read_csv('../data/cleaned/Symptoms.csv')
OUTPUT_DIR = "../data/processed"
log_message("Fichiers chargés.")

# ------------------------------------------------------------
# 2. Suppression des colonnes spécifiées (point 1)
# ------------------------------------------------------------
cols_bio_to_drop = ['task_id', 'annotateur_id', 'date_annotation', 'total_annotations',
                    'code_cim11', 'diagnostic_autre_precision']
cols_symp_to_drop = ['task_id', 'annotateur_id', 'date_annotation', 'total_annotations']

bio.drop(columns=[c for c in cols_bio_to_drop if c in bio.columns], inplace=True, errors='ignore')
symp.drop(columns=[c for c in cols_symp_to_drop if c in symp.columns], inplace=True, errors='ignore')
log_message("Colonnes inutiles supprimées.")

# ------------------------------------------------------------
# 3. Suppression des colonnes avec >90% de valeurs manquantes
# ------------------------------------------------------------
def drop_high_null(df, threshold=0.9, name=""):
    null_frac = df.isnull().mean()
    cols_to_drop = null_frac[null_frac > threshold].index.tolist()
    if cols_to_drop:
        log_message(f"{name} : colonnes supprimées (>90% null) : {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    return df

bio = drop_high_null(bio, 0.9, "bio_ATCD")
symp = drop_high_null(symp, 0.9, "Symptoms")

# ------------------------------------------------------------
# 4. Traitement des valeurs manquantes pour bio_ATCD (point 2)
# ------------------------------------------------------------
bio_cols_replace = ['atcd_abus_spa', 'atcd_familiaux_mc', 'atcd_familiaux_psy_membre',
                    'atcd_familiaux_psy_trouble', 'atcd_personnels_mc',
                    'atcd_personnels_psychiatriques', 'symptomatologie_1er_episode']
for col in bio_cols_replace:
    if col in bio.columns:
        bio[col] = bio[col].replace(['Aucun', 'RAS'], 'rien à signaler')
        bio[col] = bio[col].fillna('rien à signaler')
log_message("bio_ATCD : 'Aucun'/'RAS'/NaN remplacés par 'rien à signaler'.")

# ------------------------------------------------------------
# 5. Traitement des valeurs manquantes pour Symptoms (point 2)
# ------------------------------------------------------------
symp_cols_fill_non = ['anxiete_anticipatoire', 'attaques_panique_recentes', 'compulsions',
                      'obsessions', 'symptomes_negatifs', 'syndrome_anxieux',
                      'syndrome_delirant', 'syndrome_depressif', 'syndrome_desorganisation',
                      'syndrome_excitation_psychomotrice', 'syndrome_hallucinatoire',
                      'troubles_cognitifs']
for col in symp_cols_fill_non:
    if col in symp.columns:
        symp[col] = symp[col].fillna('non')
log_message("Symptoms : NaN remplacés par 'non' pour les items binaires.")

# ------------------------------------------------------------
# 6. NORMALISATION DES LIBELLÉS CATÉGORIELS (1.3)
# ------------------------------------------------------------
def normalize_categorical_series(series):
    """Met en forme : strip, title case, replace common variants."""
    if series.dtype == 'object':
        # Remplacer les variantes de Oui/Non
        series = series.replace(['OUI', 'oui', 'Oui '], 'Oui')
        series = series.replace(['NON', 'non', 'Non '], 'Non')
        # Mise en forme standard
        series = series.astype(str).str.strip()
        # Gestion spéciale du sexe
        if series.name == 'sexe':
            series = series.replace(['feminin', 'Feminin', 'FEMININ'], 'Féminin')
            series = series.replace(['masculin', 'Masculin', 'MASCULIN'], 'Masculin')
        else:
            series = series.str.title()
        # Remplacer les vides/NaN
        series = series.replace(['Nan', 'nan', 'None', ''], np.nan)
    return series

# Appliquer la normalisation à toutes les colonnes de type object
for col in bio.select_dtypes(include=['object']).columns:
    bio[col] = normalize_categorical_series(bio[col])
for col in symp.select_dtypes(include=['object']).columns:
    symp[col] = normalize_categorical_series(symp[col])
log_message("Normalisation des libellés catégoriels effectuée (casse, accents, synonymes).")

# ------------------------------------------------------------
# 7. ANONYMISATION (1.3) : suppression de la date de naissance
# ------------------------------------------------------------
if 'date_naissance' in bio.columns:
    bio.drop(columns=['date_naissance'], inplace=True)
    log_message("Anonymisation : colonne 'date_naissance' supprimée.")
else:
    log_message("Anonymisation : colonne 'date_naissance' déjà absente.")

# ------------------------------------------------------------
# 8. CONTRÔLES DE TYPE (1.3)
# ------------------------------------------------------------
# Convertir les colonnes numériques en float/int
numeric_cols_bio = ['age_apparition_premiers_symptomes', 'age_premiere_consultation_generale',
                    'age_premiere_consultation_service', 'atcd_judiciaire_duree_mois',
                    'atcd_judiciaire_nb', 'nb_consultations', 'nb_hospitalisations_int',
                    'nb_tentatives_suicide', 'nombre_redoublements', 'rang_dans_fratrie',
                    'taille_fratrie']
numeric_cols_symp = []  # aucune colonne numérique explicite dans Symptoms (sauf si patient_id)

for col in numeric_cols_bio:
    if col in bio.columns:
        bio[col] = pd.to_numeric(bio[col], errors='coerce')
log_message("Conversions de types numériques effectuées.")

# ------------------------------------------------------------
# 9. CONTRÔLES DE PLAGE ET COHÉRENCE INTER-VARIABLES (1.3)
# ------------------------------------------------------------
validation_issues = []

# 9.1 Âge d'apparition <= âge première consultation
if 'age_apparition_premiers_symptomes' in bio.columns and 'age_premiere_consultation_generale' in bio.columns:
    mask = (bio['age_apparition_premiers_symptomes'].notna()) & (bio['age_premiere_consultation_generale'].notna()) & \
           (bio['age_apparition_premiers_symptomes'] > bio['age_premiere_consultation_generale'])
    if mask.any():
        nb = mask.sum()
        issues = bio.loc[mask, 'patient_id'].tolist()
        validation_issues.append(f"Âge apparition > âge 1ère consultation générale : {nb} patient(s) ({issues})")
        # Correction : on met l'âge d'apparition à NaN pour ces lignes (ou on pourrait le fixer à l'âge de consultation)
        bio.loc[mask, 'age_apparition_premiers_symptomes'] = np.nan
        log_message("Correction appliquée : age_apparition > age_consultation -> mis à NaN.")

# 9.2 Cohérence tentative de suicide / nb_tentatives
if 'atcd_tentative_suicide' in bio.columns and 'nb_tentatives_suicide' in bio.columns:
    mask_ts = (bio['atcd_tentative_suicide'] == 'Non') & (bio['nb_tentatives_suicide'] > 0)
    if mask_ts.any():
        nb = mask_ts.sum()
        issues = bio.loc[mask_ts, 'patient_id'].tolist()
        validation_issues.append(f"atcd_tentative_suicide='Non' mais nb_tentatives>0 : {nb} patient(s) ({issues})")
        bio.loc[mask_ts, 'nb_tentatives_suicide'] = 0
        log_message("Correction : nb_tentatives_suicide mis à 0 pour les patients sans antécédent.")

# 9.3 Cohérence hospitalisation_suivi / nb_hospitalisations_int
if 'hospitalisation_suivi' in bio.columns and 'nb_hospitalisations_int' in bio.columns:
    mask_hosp = (bio['hospitalisation_suivi'] == 'Non') & (bio['nb_hospitalisations_int'] > 0)
    if mask_hosp.any():
        nb = mask_hosp.sum()
        issues = bio.loc[mask_hosp, 'patient_id'].tolist()
        validation_issues.append(f"hospitalisation_suivi='Non' mais nb_hospitalisations>0 : {nb} patient(s) ({issues})")
        bio.loc[mask_hosp, 'nb_hospitalisations_int'] = 0
        log_message("Correction : nb_hospitalisations_int mis à 0 pour les patients sans hospitalisation.")

# 9.4 Vérification des âges négatifs ou absurdes
for col in ['age_apparition_premiers_symptomes', 'age_premiere_consultation_generale', 'age_premiere_consultation_service']:
    if col in bio.columns:
        neg_mask = bio[col] < 0
        if neg_mask.any():
            nb = neg_mask.sum()
            issues = bio.loc[neg_mask, 'patient_id'].tolist()
            validation_issues.append(f"{col} négatif : {nb} patient(s) ({issues}) -> mis à NaN")
            bio.loc[neg_mask, col] = np.nan

# Sauvegarde du rapport de validation
if validation_issues:
    with open('validation_report.txt', 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE VALIDATION - INCOHÉRENCES CORRIGÉES\n")
        f.write("="*60 + "\n")
        for issue in validation_issues:
            f.write(f"- {issue}\n")
    log_message("Rapport de validation généré (validation_report.txt).")
else:
    log_message("Aucune incohérence majeure détectée.")

# ------------------------------------------------------------
# 10. DÉTECTION DES DOUBLONS (1.2)
# ------------------------------------------------------------
# 10.1 Unicité de patient_id dans chaque table
def check_unique_patient_id(df, name):
    if 'patient_id' in df.columns:
        duplicated = df['patient_id'].duplicated()
        if duplicated.any():
            dup_ids = df.loc[duplicated, 'patient_id'].tolist()
            log_message(f"{name} : {len(dup_ids)} identifiants dupliqués : {dup_ids}")
            # Suppression des doublons (garde la première occurrence)
            initial_len = len(df)
            df.drop_duplicates(subset='patient_id', keep='first', inplace=True)
            removed = initial_len - len(df)
            log_message(f"{name} : {removed} lignes supprimées (doublons exacts sur patient_id).")
        else:
            log_message(f"{name} : aucun doublon sur patient_id.")
    return df

bio = check_unique_patient_id(bio, "bio_ATCD")
symp = check_unique_patient_id(symp, "Symptoms")

# 10.2 Détection de doublons "proches" (même profil démographique + dates)
# On se base sur : sexe, age_premiere_consultation_generale, situation_familiale, diagnostic_principal
# On génère un fichier pour examen manuel
dup_candidates = bio.copy()
group_cols = ['sexe', 'age_premiere_consultation_generale', 'situation_familiale', 'diagnostic_principal']
# Filtrer les lignes où ces colonnes ne sont pas vides
dup_candidates = dup_candidates.dropna(subset=group_cols)
# Trouver les groupes avec plus d'une occurrence
duplicate_groups = dup_candidates.groupby(group_cols).filter(lambda x: len(x) > 1)
if not duplicate_groups.empty:
    duplicate_groups.to_csv('potential_duplicates.csv', index=False, encoding='utf-8')
    log_message(f"Doublons 'proches' détectés : {len(duplicate_groups)} lignes potentiellement dupliquées. "
                f"Consulter 'potential_duplicates.csv' pour examen manuel.")
else:
    log_message("Aucun doublon 'proche' détecté.")

# ------------------------------------------------------------
# 11. FUSION DES DEUX FICHIERS
# ------------------------------------------------------------
if 'patient_id' in bio.columns and 'patient_id' in symp.columns:
    merged = pd.merge(bio, symp, on='patient_id', how='inner')
    log_message(f"Fusion réalisée : {len(merged)} lignes.")
    
    # 10.3 Contrôle post-fusion : nombre de lignes = nombre de patients uniques
    if len(merged) == merged['patient_id'].nunique():
        log_message("Contrôle post-fusion OK : une ligne par patient.")
    else:
        log_message(f"ATTENTION : {len(merged)} lignes pour {merged['patient_id'].nunique()} patients uniques.")
else:
    log_message("Fusion impossible : patient_id manquant.")
    merged = pd.DataFrame()

# ============================================================
# 12. SAUVEGARDE DES FICHIERS
# ============================================================
bio.to_csv(os.path.join(OUTPUT_DIR, 'bio_ATCD_preprocessed.csv'), index=False, encoding='utf-8')
symp.to_csv(os.path.join(OUTPUT_DIR, 'Symptoms_preprocessed.csv'), index=False, encoding='utf-8')
if not merged.empty:
    merged.to_csv(os.path.join(OUTPUT_DIR, 'merged_data.csv'), index=False, encoding='utf-8')

log_message(f"Fichiers prétraités sauvegardés dans '{OUTPUT_DIR}'.")

# ============================================================
# 13. JOURNAL DE TRACABILITÉ
# ============================================================
with open(os.path.join(OUTPUT_DIR, 'preprocessing_log.txt'), 'w', encoding='utf-8') as f:
    f.write("JOURNAL DE PRÉTRAITEMENT\n")
    f.write("="*60 + "\n")
    f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    for entry in log_entries:
        f.write(f"- {entry}\n")

log_message("Journal de traçabilité généré : preprocessing_log.txt")
print("\n✅ Prétraitement terminé.")