"""2D molecule rendering utilities for ChemASSistant.

All structure rendering is performed locally with RDKit.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Draw


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string and return an RDKit molecule."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    return mol


def smiles_to_image(
    smiles: str,
    size: tuple[int, int] = (300, 300),
) -> Image.Image:
    """Render a single SMILES string as a 2D PIL image."""
    mol = _mol_from_smiles(smiles)
    AllChem.Compute2DCoords(mol)

    return Draw.MolToImage(
        mol,
        size=size,
    )


def draw_molecule_grid(
    smiles_list: Sequence[str],
    legends: Sequence[str] | None = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (250, 200),
) -> Image.Image:
    """Render multiple SMILES strings as a labeled 2D molecule grid.

    Invalid SMILES entries are skipped. If legends are supplied, they remain
    aligned with the corresponding valid molecules.
    """
    if mols_per_row < 1:
        raise ValueError("mols_per_row must be at least 1")

    if legends is not None and len(legends) != len(smiles_list):
        raise ValueError("legends must have the same length as smiles_list")

    mols: list[Chem.Mol] = []
    valid_legends: list[str] = []

    for index, smiles in enumerate(smiles_list):
        try:
            mol = _mol_from_smiles(smiles)
        except ValueError:
            continue

        AllChem.Compute2DCoords(mol)
        mols.append(mol)

        if legends is not None:
            valid_legends.append(str(legends[index]))

    if not mols:
        raise ValueError("No valid molecules to render")

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=valid_legends if legends is not None else None,
        useSVG=False,
    )
