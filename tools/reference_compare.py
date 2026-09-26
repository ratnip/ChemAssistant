"""Reference-database similarity utilities for ChemASSistant.

All similarity calculations are performed locally with RDKit.
No LLM calls are used here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from tools.database import DEFAULT_DB_PATH, _connect


def find_most_similar_batch(
    query_smiles_list: list[str],
    top_n: int = 3,
    exclude_exact_match: bool = True,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Find top reference-database matches for multiple query molecules.

    The reference database is loaded and fingerprinted once, then reused for
    every query. This is substantially more efficient than reopening and
    re-fingerprinting the database for each input molecule.

    Args:
        query_smiles_list: Query molecules represented as SMILES strings.
        top_n: Number of reference matches to return per query.
        exclude_exact_match: Exclude reference records with similarity 1.0.
        db_path: Path to the SQLite reference database.

    Returns:
        One result per query:
        {
            "query_smiles": "...",
            "matches": [
                {
                    "chembl_id": "...",
                    "name": "...",
                    "smiles": "...",
                    "max_phase": ...,
                    "formula": "...",
                    "chembl_mw": ...,
                    "similarity": 0.75
                },
                ...
            ]
        }
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)

    # Load the local reference library once.
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT chembl_id, name, smiles, max_phase, formula, chembl_mw
            FROM molecules
            WHERE smiles IS NOT NULL
            """
        ).fetchall()

    reference_records: list[dict[str, Any]] = []
    reference_fps = []

    for row in rows:
        record = dict(row)
        reference_smiles = record["smiles"]

        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(reference_smiles)

        if mol is None:
            continue

        reference_records.append(record)
        reference_fps.append(generator.GetFingerprint(mol))

    results: list[dict[str, Any]] = []

    for query_smiles in query_smiles_list:
        if not isinstance(query_smiles, str) or not query_smiles.strip():
            results.append(
                {
                    "query_smiles": query_smiles,
                    "matches": [],
                    "error": "Invalid query SMILES",
                }
            )
            continue

        with rdBase.BlockLogs():
            query_mol = Chem.MolFromSmiles(query_smiles)

        if query_mol is None:
            results.append(
                {
                    "query_smiles": query_smiles,
                    "matches": [],
                    "error": "Invalid query SMILES",
                }
            )
            continue

        query_fp = generator.GetFingerprint(query_mol)

        # RDKit calculates all Tanimoto scores locally in one call.
        scores = DataStructs.BulkTanimotoSimilarity(
            query_fp,
            reference_fps,
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        matches = []

        for index, similarity in ranked:
            if exclude_exact_match and similarity == 1.0:
                continue

            match = reference_records[index].copy()
            match["similarity"] = float(similarity)
            matches.append(match)

            if len(matches) >= top_n:
                break

        results.append(
            {
                "query_smiles": query_smiles,
                "matches": matches,
            }
        )

    return results
