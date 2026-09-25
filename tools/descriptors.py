from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors


def molecular_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    return Descriptors.MolWt(mol)


def calculate_descriptors(smiles: str) -> dict:

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    return {
        "smiles": smiles,
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
    }


def calculate_descriptors_batch(smiles_list: list[str]) -> list[dict]:
    results = []

    for smiles in smiles_list:
        try:
            result = calculate_descriptors(smiles)
            results.append(result)

        except ValueError as e:
            results.append({
                "smiles": smiles,
                "error": str(e)
            })

    return results