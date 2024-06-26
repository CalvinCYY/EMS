import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.periodic_table import Get_periodic_table


def from_rdmol(rdmol):
    type_array = np.zeros(rdmol.GetNumAtoms(), dtype=np.int32)
    xyz_array = np.zeros((rdmol.GetNumAtoms(), 3), dtype=np.float64)
    conn_array = np.zeros((rdmol.GetNumAtoms(), rdmol.GetNumAtoms()), dtype=np.int32)

    for i, atoms in enumerate(rdmol.GetAtoms()):
        type_array[i] = atoms.GetAtomicNum()
        if rdmol.GetNumConformers() < 1:
            AllChem.Compute2DCoords(rdmol)
        xyz_array[i][0] = rdmol.GetConformer().GetAtomPosition(i).x
        xyz_array[i][1] = rdmol.GetConformer().GetAtomPosition(i).y
        xyz_array[i][2] = rdmol.GetConformer().GetAtomPosition(i).z

        for j, atoms in enumerate(rdmol.GetAtoms()):
            if i == j:
                continue

            bond = rdmol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                conn_array[i][j] = int(bond.GetBondTypeAsDouble())
                conn_array[j][i] = int(bond.GetBondTypeAsDouble())

    return type_array, xyz_array, conn_array


def to_rdmol(structure, sanitize=True):
    # Create an RDKit molecule object
    periodic_table = Get_periodic_table()
    rdmol = Chem.RWMol()

    # Add the atoms to the molecule
    for atom in structure["types"]:
        symbol = periodic_table[int(atom)]
        rdmol.AddAtom(Chem.Atom(symbol))

        # Add the bonds to the molecule
    visited = []
    for i, bond_order_array in enumerate(structure["conn"]):
        for j, bond_order in enumerate(bond_order_array):
            if j in visited:
                continue
            elif bond_order != 0:
                rdmol.AddBond(i, j, Chem.BondType(bond_order))
            else:
                continue
        visited.append(i)

        # Add the coordinates to the atoms
    conformer = Chem.Conformer()
    for i, coord in enumerate(structure["xyz"]):
        conformer.SetAtomPosition(i, coord)
    rdmol.AddConformer(conformer)

    rdmol = rdmol.GetMol()
    # Sanitize the molecule
    if sanitize:
        Chem.SanitizeMol(rdmol)
    return rdmol
