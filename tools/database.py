"""Database utilities for the ChemASSistant reference molecule database.

Expected database location (relative to the ChemASSistant project root):
    data/chemass_reference.db

The module deliberately owns the SQL layer so the agent does not need to
write SQL itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


# Resolve the database relative to this file:
# ChemASSistant/tools/database.py -> ChemASSistant/data/chemass_reference.db
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chemass_reference.db"


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the reference database and return rows as dictionary-like objects."""
    path = Path(db_path)

    if not path.exists():
        raise FileNotFoundError(f"Reference database not found: {path}")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def database_info(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Return basic information about the local reference database."""
    path = Path(db_path)

    with _connect(path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(molecules)").fetchall()
        ]

    return {
        "database_path": str(path),
        "molecule_count": count,
        "columns": columns,
    }


def get_molecule(
    chembl_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Retrieve one molecule by exact ChEMBL ID."""
    if not isinstance(chembl_id, str) or not chembl_id.strip():
        raise ValueError("chembl_id must be a non-empty string")

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT chembl_id, name, smiles, max_phase, formula, chembl_mw
            FROM molecules
            WHERE UPPER(chembl_id) = UPPER(?)
            LIMIT 1
            """,
            (chembl_id.strip(),),
        ).fetchone()

    return dict(row) if row else None


def search_by_name(
    name: str,
    limit: int = 20,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Search molecule names using a case-insensitive partial match."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    if limit < 1:
        raise ValueError("limit must be at least 1")

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT chembl_id, name, smiles, max_phase, formula, chembl_mw
            FROM molecules
            WHERE name IS NOT NULL
              AND LOWER(name) LIKE LOWER(?)
            ORDER BY
                CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                name
            LIMIT ?
            """,
            (f"%{name.strip()}%", name.strip(), limit),
        ).fetchall()

    return _rows_to_dicts(rows)


def search_by_smiles(
    smiles: str,
    limit: int = 20,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Search for an exact stored SMILES string.

    This is a text match, not a structure-equivalence search. Structural
    matching/canonicalization should remain in the RDKit chemistry layer.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    if limit < 1:
        raise ValueError("limit must be at least 1")

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT chembl_id, name, smiles, max_phase, formula, chembl_mw
            FROM molecules
            WHERE smiles = ?
            LIMIT ?
            """,
            (smiles.strip(), limit),
        ).fetchall()

    return _rows_to_dicts(rows)


def search_by_property(
    min_mw: float | None = None,
    max_mw: float | None = None,
    min_phase: float | None = None,
    limit: int = 20,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Filter reference molecules using the small set of stored metadata."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    conditions: list[str] = []
    params: list[Any] = []

    if min_mw is not None:
        conditions.append("CAST(chembl_mw AS REAL) >= ?")
        params.append(float(min_mw))

    if max_mw is not None:
        conditions.append("CAST(chembl_mw AS REAL) <= ?")
        params.append(float(max_mw))

    if min_phase is not None:
        conditions.append("CAST(max_phase AS REAL) >= ?")
        params.append(float(min_phase))

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT chembl_id, name, smiles, max_phase, formula, chembl_mw
        FROM molecules
        {where_clause}
        ORDER BY CAST(chembl_mw AS REAL), chembl_id
        LIMIT ?
    """
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return _rows_to_dicts(rows)


if __name__ == "__main__":
    # Small standalone smoke test. Run from the project root with:
    #     python tools/database.py
    print(database_info())
    print("\nSearch for aspirin:")
    for result in search_by_name("aspirin", limit=5):
        print(result)
