from rdkit import Chem
from rdkit import rdBase


def validate_smiles(smiles: str) -> dict:

    if not isinstance(smiles, str) or not smiles.strip():
        return {
            "original_smiles": smiles,
            "valid": False,
            "canonical_smiles": None,
            "error": "SMILES must be a non-empty string"
        }

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "original_smiles": smiles,
            "valid": False,
            "canonical_smiles": None,
            "error": "Invalid SMILES"
        }

    return {
        "original_smiles": smiles,
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "error": None
    }

def validate_molecules(smiles_list: list[str]) -> list[dict]:
    results = []

    for smiles in smiles_list:
        result = validate_smiles(smiles)
        results.append(result)

    return results