"""
Convertit chaque export JSON Label Studio (placé dans data/raw) en un
fichier CSV séparé écrit dans data/cleaned.

Arborescence attendue (ce script est dans DataAnalyseProject/src/) :

    DataAnalyseProject/
        data/
            raw/        <- tous les .json à convertir
            cleaned/    <- les .csv générés (créé automatiquement si absent)
        src/
            data_cleaning.py   <- ce fichier

Usage (depuis n'importe où) :
    python data_cleaning.py

Aucun argument requis : le script détecte lui-même tous les .json
présents dans data/raw.
"""

import json
from pathlib import Path
import pandas as pd

MULTI_CHOICE_SEPARATOR = ";"

# Dossiers relatifs à ce script (src/../data/raw et src/../data/cleaned)
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"


def extract_value(result_item):
    value = result_item.get("value", {})
    item_type = result_item.get("type")

    if item_type == "choices":
        return MULTI_CHOICE_SEPARATOR.join(value.get("choices", []))
    if item_type == "number":
        return value.get("number")
    if item_type == "textarea":
        return MULTI_CHOICE_SEPARATOR.join(value.get("text", []))
    return value


def task_to_row(task):
    row = {"patient_id": task.get("data", {}).get("patient_id")}
    row["task_id"] = task.get("id")
    row["total_annotations"] = task.get("total_annotations")

    annotations = task.get("annotations", [])
    if not annotations:
        return row

    annotation = annotations[0]
    row["annotateur_id"] = annotation.get("completed_by")
    row["date_annotation"] = annotation.get("updated_at")

    for result_item in annotation.get("result", []):
        field_name = result_item.get("from_name")
        row[field_name] = extract_value(result_item)

    return row


def convert_one(input_path: Path, output_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [task_to_row(task) for task in data]
    df = pd.DataFrame(rows)

    id_cols = ["patient_id", "task_id", "annotateur_id", "date_annotation", "total_annotations"]
    other_cols = sorted(c for c in df.columns if c not in id_cols)
    df = df[[c for c in id_cols if c in df.columns] + other_cols]
    df = df.sort_values("patient_id").reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{input_path.name} -> {output_path.name} ({len(df)} lignes, {len(df.columns)} colonnes)")


def main():
    if not RAW_DIR.exists():
        print(f"Erreur : le dossier {RAW_DIR} n'existe pas.")
        return

    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        print(f"Aucun fichier .json trouvé dans {RAW_DIR}")
        return

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    for input_path in json_files:
        output_path = CLEANED_DIR / (input_path.stem + ".csv")
        convert_one(input_path, output_path)


if __name__ == "__main__":
    main()