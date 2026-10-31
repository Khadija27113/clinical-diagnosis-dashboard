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
                      'syndrome_excitation_psychomotrice', 'syndrome_hallucinatoire', 'syndrome_phobique',
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
    if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
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
for col in [c for c in bio.columns if not pd.api.types.is_numeric_dtype(bio[c])]:
    bio[col] = normalize_categorical_series(bio[col])
for col in [c for c in symp.columns if not pd.api.types.is_numeric_dtype(symp[c])]:
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
# 11b. PREPARATION POUR L'ANALYSE ET LA MODELISATION
# ============================================================
# Complete les etapes precedentes : valeurs manquantes restantes, types,
# incoherences. merged_data.csv reste lisible (texte Oui/Non, libelles
# normalises) pour rester compatible avec l'EDA et le dashboard.
# L'encodage numerique (0/1, multi-hot, ordinal) et les variables derivees
# relevent de l'etape de feature engineering, pas de ce script.

SYMPTOM_BINARY_COLS = [
    'anxiete_anticipatoire', 'attaques_panique_recentes', 'compulsions', 'obsessions',
    'symptomes_negatifs', 'syndrome_anxieux', 'syndrome_delirant', 'syndrome_depressif',
    'syndrome_desorganisation', 'syndrome_excitation_psychomotrice',
    'syndrome_hallucinatoire', 'syndrome_phobique', 'troubles_cognitifs',
]
# Variables de suivi / resultats (consequences du diagnostic et du traitement) :
# on ne fabrique AUCUNE valeur pour elles, les NaN sont conserves.
FOLLOWUP_OUTCOME_COLS = [
    'etat_clinique_derniere_consult', 'observance_derniere_consult', 'tolerance_derniere_consult',
    'observance_consult_reference', 'tolerance_consult_reference',
    'orientation_premiere_consult', 'orientation_derniere_consult', 'orientation_consult_reference',
    'rechute_12_mois', 'rechute_scolaire_recente', 'hospitalisation_12_mois', 'nb_consultations',
]
AGE_COLS = ['age_apparition_premiers_symptomes', 'age_premiere_consultation_generale',
            'age_premiere_consultation_service']
COUNT_COLS = ['nb_consultations', 'nb_hospitalisations_int', 'nb_tentatives_suicide',
              'nombre_redoublements', 'rang_dans_fratrie', 'taille_fratrie']
STRUCTURAL_COLS = ['nb_hospitalisations_int', 'nb_tentatives_suicide', 'severite_traumatisme']
FREE_TEXT_COLS = ['motif_premiere_consultation']
MAX_IMPUTE_FRACTION = 0.05   # au-dela, pas d'imputation automatique


def prepare_dataset(df):
    df = df.copy()
    n = len(df)
    flags = []   # incoherences SIGNALEES, non corrigees (decision clinique)
    nan_before = int(df.isna().sum().sum())

    # a) symptomes binaires : absence de mention = 'Non' (meme hypothese qu'a l'etape 5)
    for col in SYMPTOM_BINARY_COLS:
        if col in df.columns and df[col].isna().any():
            k = int(df[col].isna().sum())
            df[col] = df[col].fillna('Non')
            log_message(f"Symptome '{col}' : {k} NaN -> 'Non'.")

    # b) incoherences : detectees AVANT toute imputation
    if 'age_apparition_premiers_symptomes' in df.columns:
        m = df['age_apparition_premiers_symptomes'] < 5
        if m.any():
            flags.append(f"age_apparition_premiers_symptomes < 5 ans : {int(m.sum())} patient(s) "
                         f"{df.loc[m, 'patient_id'].tolist()} (valeurs {df.loc[m, 'age_apparition_premiers_symptomes'].tolist()}) "
                         f"- a verifier (0 peut coder 'depuis l'enfance' ou une donnee inconnue).")
    if {'rang_dans_fratrie', 'taille_fratrie'} <= set(df.columns):
        m = df['rang_dans_fratrie'] > df['taille_fratrie']
        if m.any():
            flags.append(f"rang_dans_fratrie > taille_fratrie : {int(m.sum())} patient(s) "
                         f"{df.loc[m, 'patient_id'].tolist()} (taille_fratrie = {df.loc[m, 'taille_fratrie'].tolist()}).")
    if 'nb_consultations' in df.columns:
        q1, q3 = df['nb_consultations'].quantile([0.25, 0.75])
        cut = q3 + 3 * (q3 - q1)
        m = df['nb_consultations'] > cut
        if m.any():
            flags.append(f"nb_consultations tres eleve (> Q3 + 3*IQR = {cut:.0f}) : {int(m.sum())} patient(s) "
                         f"{df.loc[m, 'patient_id'].tolist()} (valeurs {df.loc[m, 'nb_consultations'].tolist()}).")
    if {'severite_traumatisme', 'type_traumatisme_emotionnel'} <= set(df.columns):
        m = df['type_traumatisme_emotionnel'].eq('Aucun') & df['severite_traumatisme'].notna()
        if m.any():
            flags.append(f"type_traumatisme_emotionnel = 'Aucun' mais severite renseignee : {int(m.sum())} patient(s) "
                         f"{df.loc[m, 'patient_id'].tolist()}.")
    for col in SYMPTOM_BINARY_COLS:
        if col in df.columns:
            bad = ~df[col].isin(['Oui', 'Non'])
            if bad.any():
                flags.append(f"{col} : valeurs hors {{Oui, Non}} chez {int(bad.sum())} patient(s).")

    # c) manquants sporadiques des variables predictives : mode (categoriel) / mediane (numerique)
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    multi_valued = [c for c in text_cols if df[c].astype(str).str.contains(';', na=False).any()]
    skip = (set(FOLLOWUP_OUTCOME_COLS) | set(multi_valued) | set(FREE_TEXT_COLS) |
            set(STRUCTURAL_COLS) | {'patient_id', 'diagnostic_principal'})
    for col in df.columns:
        if col in skip or not df[col].isna().any():
            continue
        k = int(df[col].isna().sum())
        if k / n > MAX_IMPUTE_FRACTION:
            log_message(f"ATTENTION : '{col}' a {k / n:.0%} de NaN : pas d'imputation automatique.")
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].median()
            if col in AGE_COLS + COUNT_COLS:
                value = int(round(value))
            how = f"mediane ({value})"
        else:
            value = df[col].mode(dropna=True).iloc[0]
            how = f"mode ('{value}')"
        df[col] = df[col].fillna(value)
        log_message(f"'{col}' : {k} NaN -> {how}.")

    # d) NaN structurels = absence documentee, recodes grace a l'indicateur associe
    for count_col, flag_col in [('nb_hospitalisations_int', 'hospitalisation_suivi'),
                                ('nb_tentatives_suicide', 'atcd_tentative_suicide')]:
        if count_col in df.columns and flag_col in df.columns:
            m = df[count_col].isna() & df[flag_col].eq('Non')
            df.loc[m, count_col] = 0
            log_message(f"'{count_col}' : {int(m.sum())} NaN -> 0 (absence documentee : {flag_col} = 'Non').")
            left = int(df[count_col].isna().sum())
            if left:
                value = int(round(df[count_col].median()))
                df[count_col] = df[count_col].fillna(value)
                log_message(f"'{count_col}' : {left} NaN restants (indicateur = 'Oui') -> mediane ({value}).")
    if {'severite_traumatisme', 'type_traumatisme_emotionnel'} <= set(df.columns):
        m = df['severite_traumatisme'].isna() & df['type_traumatisme_emotionnel'].eq('Aucun')
        df.loc[m, 'severite_traumatisme'] = 'Aucun'
        log_message(f"'severite_traumatisme' : {int(m.sum())} NaN -> 'Aucun' (type de traumatisme = 'Aucun').")
        left = df['severite_traumatisme'].isna()
        if left.any():
            mode_sev = df.loc[df['severite_traumatisme'] != 'Aucun', 'severite_traumatisme'].mode().iloc[0]
            df.loc[left, 'severite_traumatisme'] = mode_sev
            log_message(f"'severite_traumatisme' : {int(left.sum())} NaN restant(s) (traumatisme present, "
                        f"severite inconnue) -> mode ('{mode_sev}').")

    # e) types : ages et compteurs en entiers
    for col in AGE_COLS + COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')

    # f) bilan : NaN conserves volontairement
    remaining = df.isna().sum()
    remaining = remaining[remaining > 0]
    def reason(col):
        if col in FOLLOWUP_OUTCOME_COLS: return 'variable de suivi/resultat'
        if col in FREE_TEXT_COLS: return 'texte libre'
        if col in multi_valued: return 'multi-valeurs (NaN = aucune modalite)'
        return 'autre'
    nan_after = int(remaining.sum())
    log_message(f"Valeurs manquantes : {nan_before} -> {nan_after} cellules ; "
                f"{len(remaining)} colonne(s) conservent des NaN (voir {OUTPUT_DIR}/quality_flags.txt).")

    with open(os.path.join(OUTPUT_DIR, 'quality_flags.txt'), 'w', encoding='utf-8') as f:
        f.write("INCOHERENCES SIGNALEES (non corrigees - a verifier avec les annotateurs)\n")
        f.write("=" * 70 + "\n")
        if flags:
            for line in flags:
                f.write(f"- {line}\n")
        else:
            f.write("Aucune.\n")
        f.write("\nNaN CONSERVES VOLONTAIREMENT\n")
        f.write("=" * 70 + "\n")
        for col, k in remaining.items():
            f.write(f"- {col}: {int(k)} ({reason(col)})\n")
    return df


if not merged.empty:
    merged = prepare_dataset(merged)

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