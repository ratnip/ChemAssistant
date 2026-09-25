from tools.validation import validate_molecules
from tools.descriptors import calculate_descriptors_batch
from tools.rules import check_lipinski
from tools.similarity import get_top_similar_pairs
from tools.summary import summarize_validation, summarize_descriptors

def analyze_library(smiles_list: list[str]) -> dict:
    validation_results = validate_molecules(smiles_list)

    valid_smiles = [
        r["canonical_smiles"]
        for r in validation_results
        if r["valid"]
    ]

    descriptor_results = calculate_descriptors_batch(valid_smiles)

    rule_results = [
        {
            "smiles": d["smiles"],
            **check_lipinski(d)
        }
        for d in descriptor_results
        if "error" not in d
    ]

    top_similar_pairs = get_top_similar_pairs(
        valid_smiles,
        top_n=5
    )

    return {
        "validation": validation_results,
        "descriptors": descriptor_results,
        "rules": rule_results,
        "top_similar_pairs": top_similar_pairs,
        "validation_summary": summarize_validation(validation_results),
        "descriptor_summary": summarize_descriptors(descriptor_results),
    }