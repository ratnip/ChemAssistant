"""SDF ingestion for ChemAssistant.

Place this file at tools/sdf.py. It converts SDF records to a DataFrame
compatible with the existing CSV -> prepare_dataset -> analyze_dataframe path.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from rdkit import Chem


def load_sdf(source: bytes | bytearray | str | Path | BinaryIO) -> pd.DataFrame:
    """Load an SD file into a table with a ``smiles`` column.

    Molecule names and SD properties are preserved. Unreadable records are kept
    as rows with a missing SMILES, so the downstream validation can flag them.
    This is a 2D/graph import: conformer coordinates are not preserved in CSV.
    """
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raw = source.read()
        if isinstance(raw, str):
            raise TypeError("SDF source must contain bytes, not text")

    if not raw.strip():
        raise ValueError("The uploaded SDF is empty")

    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(raw), sanitize=True, removeHs=True, strictParsing=False
    )
    rows: list[dict[str, object]] = []

    for index, mol in enumerate(supplier, start=1):
        row: dict[str, object] = {
            "compound_id": f"SDF_{index:05d}",
            "name": None,
            "smiles": None,
            "sdf_record": index,
            "sdf_parse_error": None,
        }
        if mol is None:
            row["sdf_parse_error"] = "RDKit could not parse this SDF record"
        else:
            row["name"] = mol.GetProp("_Name").strip() or None if mol.HasProp("_Name") else None
            row["smiles"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            for prop in mol.GetPropNames():
                # Keep the generated schema stable even when SDF properties
                # have names such as 'smiles', 'name' or 'compound_id'.
                column = prop if prop not in row else f"sdf_prop_{prop}"
                row[column] = mol.GetProp(prop)
        rows.append(row)

    if not rows:
        raise ValueError("No molecule records found in the uploaded SDF")
    return pd.DataFrame(rows)
