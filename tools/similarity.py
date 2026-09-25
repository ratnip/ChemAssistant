from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


def calculate_similarity(smiles1: str, smiles2: str) -> float:
    with rdBase.BlockLogs():
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)

    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_pairwise_similarities(smiles_list: list[str]) -> list[dict]:
    results = []

    for i in range(len(smiles_list)):
        for j in range(i + 1, len(smiles_list)):
            smiles1 = smiles_list[i]
            smiles2 = smiles_list[j]

            try:
                score = calculate_similarity(smiles1, smiles2)

                results.append({
                    "smiles1": smiles1,
                    "smiles2": smiles2,
                    "similarity": score
                })

            except ValueError as e:
                results.append({
                    "smiles1": smiles1,
                    "smiles2": smiles2,
                    "similarity": None,
                    "error": str(e)
                })

    return results


def get_top_similar_pairs(smiles_list: list[str], top_n: int = 5) -> list[dict]:
    results = calculate_pairwise_similarities(smiles_list)

    valid_results = [
        r for r in results
        if r["similarity"] is not None
    ]

    valid_results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return valid_results[:top_n]