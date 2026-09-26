from __future__ import annotations

import pandas as pd

from tools.validation import validate_molecules
from tools.descriptors import calculate_descriptors_batch
from tools.rules import check_lipinski
from tools.similarity import get_top_similar_pairs
from tools.summary import summarize_validation, summarize_descriptors
from tools.reference_compare import find_most_similar_batch


def analyze_library(smiles_list: list[str]) -> dict:
    """Run the standard ChemASSistant analysis workflow on a list of SMILES."""
    validation_results = validate_molecules(smiles_list)

    valid_smiles = [
        r["canonical_smiles"]
        for r in validation_results
        if r["valid"]
    ]

    descriptor_results = calculate_descriptors_batch(valid_smiles)

    rule_results = [
        {"smiles": d["smiles"], **check_lipinski(d)}
        for d in descriptor_results
        if "error" not in d
    ]

    top_similar_pairs = get_top_similar_pairs(valid_smiles, top_n=5)

    return {
        "validation": validation_results,
        "descriptors": descriptor_results,
        "rules": rule_results,
        "top_similar_pairs": top_similar_pairs,
        "validation_summary": summarize_validation(validation_results),
        "descriptor_summary": summarize_descriptors(descriptor_results),
    }


def analyze_dataframe(
    df: pd.DataFrame,
    smiles_column: str = "canonical_smiles",
    compare_reference: bool = False,
    reference_top_n: int = 3,
) -> dict:
    """Analyze an annotated molecular DataFrame.

    The original row order and columns are preserved. Invalid molecules remain
    in the output. Descriptors, rule checks, and optional local-reference
    similarity results are added only for valid structures.
    """
    required_columns = {"valid", smiles_column}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if reference_top_n < 1:
        raise ValueError("reference_top_n must be at least 1")

    result_df = df.copy()

    valid_mask = (
        result_df["valid"].eq(True)
        & result_df[smiles_column].notna()
    )

    valid_smiles = result_df.loc[
        valid_mask,
        smiles_column,
    ].tolist()

    descriptor_results = calculate_descriptors_batch(valid_smiles)

    rule_results = [
        {"smiles": d["smiles"], **check_lipinski(d)}
        for d in descriptor_results
        if "error" not in d
    ]

    descriptor_map = {
        d["smiles"]: d
        for d in descriptor_results
        if "error" not in d
    }

    rule_map = {
        r["smiles"]: r
        for r in rule_results
    }

    descriptor_columns = [
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "heavy_atoms",
        "formal_charge",
    ]

    for column in descriptor_columns:
        result_df[column] = None

    result_df["lipinski_violations"] = None
    result_df["violations"] = None

    for index in result_df.index[valid_mask]:
        smiles = result_df.at[index, smiles_column]

        descriptor = descriptor_map.get(smiles)
        if descriptor:
            for column in descriptor_columns:
                result_df.at[index, column] = descriptor.get(column)

        rule = rule_map.get(smiles)
        if rule:
            result_df.at[index, "lipinski_violations"] = rule.get(
                "lipinski_violations"
            )
            result_df.at[index, "violations"] = rule.get("violations")

    top_similar_pairs = get_top_similar_pairs(
        valid_smiles,
        top_n=5,
    )

    validation_results = []

    for _, row in result_df.iterrows():
        original_smiles = (
            row["original_smiles"]
            if "original_smiles" in result_df.columns
            else row.get(smiles_column)
        )

        canonical_smiles = (
            row[smiles_column]
            if bool(row["valid"])
            else None
        )

        error = None
        if "validation_error" in result_df.columns:
            error = row["validation_error"]
            if pd.isna(error):
                error = None

        validation_results.append(
            {
                "original_smiles": original_smiles,
                "valid": bool(row["valid"]),
                "canonical_smiles": canonical_smiles,
                "error": error,
            }
        )

    reference_matches = []

    if compare_reference and valid_smiles:
        reference_matches = find_most_similar_batch(
            valid_smiles,
            top_n=reference_top_n,
            exclude_exact_match=True,
        )

        # Build a lookup by query SMILES. Identical query structures share
        # identical reference results, which is correct for this use case.
        match_map = {
            item["query_smiles"]: item["matches"]
            for item in reference_matches
        }

        for rank in range(1, reference_top_n + 1):
            result_df[f"ref_{rank}_chembl_id"] = None
            result_df[f"ref_{rank}_name"] = None
            result_df[f"ref_{rank}_smiles"] = None
            result_df[f"ref_{rank}_similarity"] = None

        for index in result_df.index[valid_mask]:
            smiles = result_df.at[index, smiles_column]
            matches = match_map.get(smiles, [])

            for rank, match in enumerate(matches, start=1):
                result_df.at[index, f"ref_{rank}_chembl_id"] = match.get(
                    "chembl_id"
                )
                result_df.at[index, f"ref_{rank}_name"] = match.get("name")
                result_df.at[index, f"ref_{rank}_smiles"] = match.get("smiles")
                result_df.at[index, f"ref_{rank}_similarity"] = match.get(
                    "similarity"
                )

    return {
        "dataset": result_df,
        "descriptors": descriptor_results,
        "rules": rule_results,
        "top_similar_pairs": top_similar_pairs,
        "reference_matches": reference_matches,
        "validation_summary": summarize_validation(validation_results),
        "descriptor_summary": summarize_descriptors(descriptor_results),
    }
