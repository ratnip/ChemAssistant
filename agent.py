from pipeline import analyze_library
from tools.descriptors import calculate_descriptors_batch
from tools.rules import check_lipinski
from tools.similarity import get_top_similar_pairs
from tools.validation import validate_molecules
from smolagents import tool
from tools.database import search_by_name, get_molecule


@tool
def analyze_library_tool(smiles_list: list[str]) -> dict:
    """
    Analyze a molecular library.

    Args:
        smiles_list: List of SMILES strings representing molecules.

    Returns:
        Structured analysis including validation, descriptors,
        property-rule checks, molecular similarities, and summaries.
    """
    return analyze_library(smiles_list)


@tool
def validate_molecules_tool(smiles_list: list[str]) -> list[dict]:
    """
    Validate molecular structures represented as SMILES strings.

    Args:
        smiles_list: List of SMILES strings to validate.

    Returns:
        Validation results for each SMILES, including validity
        and canonical SMILES when valid.
    """
    return validate_molecules(smiles_list)


@tool
def calculate_descriptors_tool(smiles_list: list[str]) -> list[dict]:
    """
    Calculate basic physicochemical descriptors for molecules.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Molecular descriptors for each valid SMILES,
        or an error for invalid input.
    """
    return calculate_descriptors_batch(smiles_list)



@tool
def check_property_rules_tool(smiles_list: list[str]) -> list[dict]:
    """
    Check molecular property rules for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Property-rule results for each valid molecule.
    """
    descriptor_results = calculate_descriptors_batch(smiles_list)

    results = []

    for d in descriptor_results:
        if "error" in d:
            results.append({
                "smiles": d["smiles"],
                "error": d["error"]
            })
        else:
            results.append({
                "smiles": d["smiles"],
                **check_lipinski(d)
            })

    return results



@tool
def find_similar_molecules_tool(
    smiles_list: list[str],
    top_n: int = 5
) -> list[dict]:
    """
    Find the most structurally similar pairs in a molecular library.

    Args:
        smiles_list: List of SMILES strings.
        top_n: Number of most similar pairs to return.

    Returns:
        Most similar molecular pairs ranked by Morgan/Tanimoto similarity.
    """
    return get_top_similar_pairs(smiles_list, top_n=top_n)


@tool
def search_reference_database_tool(
    name: str,
    limit: int = 5
) -> list[dict]:
    """
    Search the local reference molecule database by compound name.

    Args:
        name: Compound name or partial compound name to search for.
        limit: Maximum number of matching compounds to return.

    Returns:
        Matching reference compounds including ChEMBL ID, stored name,
        SMILES, development phase, molecular formula, and molecular weight.
    """
    return search_by_name(name, limit=limit)


@tool
def get_reference_molecule_tool(
    chembl_id: str
) -> dict:
    """
    Retrieve a molecule from the local reference database by exact ChEMBL ID.

    Args:
        chembl_id: ChEMBL identifier, for example CHEMBL25.

    Returns:
        Reference database record for the requested ChEMBL ID,
        or None if the compound is not present.
    """
    return get_molecule(chembl_id)