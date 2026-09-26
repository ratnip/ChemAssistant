"""CSV dataset ingestion utilities for ChemASSistant."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tools.validation import validate_molecules

COMMON_SMILES_COLUMNS = (
    "smiles",
    "canonical_smiles",
    "canonical smiles",
    "structure",
)


def load_csv(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a non-empty CSV file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, **read_csv_kwargs)

    if len(df.columns) == 0:
        raise ValueError("CSV contains no columns")
    if df.empty:
        raise ValueError("CSV contains no data rows")

    return df


def detect_smiles_column(df: pd.DataFrame) -> str | None:
    """Detect a likely SMILES column conservatively from its column name."""
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in COMMON_SMILES_COLUMNS:
        if candidate in normalized:
            return normalized[candidate]

    return None


def prepare_dataset(df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    """Preserve the input table and add deterministic validation annotations."""
    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    result = df.copy()

    smiles_values = [
        None if pd.isna(value) else value
        for value in result[smiles_column].tolist()
    ]

    validation_results = validate_molecules(smiles_values)

    result["original_smiles"] = [
        item["original_smiles"] for item in validation_results
    ]
    result["valid"] = [item["valid"] for item in validation_results]
    result["canonical_smiles"] = [
        item["canonical_smiles"] for item in validation_results
    ]
    result["validation_error"] = [
        item["error"] for item in validation_results
    ]

    return result


def load_and_prepare_csv(
    path: str | Path,
    smiles_column: str | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV, choose its SMILES column, and validate every structure."""
    df = load_csv(path, **read_csv_kwargs)

    if smiles_column is None:
        smiles_column = detect_smiles_column(df)

    if smiles_column is None:
        raise ValueError(
            "Could not detect a SMILES column. "
            "Specify smiles_column explicitly."
        )

    return prepare_dataset(df, smiles_column)
