"""Docking preparation utilities."""
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess

def smiles_to_3d_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def prepare_ligand_pdbqt(pdb_file, output_pdbqt):
    try:
        subprocess.run(['obabel', '-ipdb', pdb_file, '-opdbqt', '-O', output_pdbqt], check=True)
        return output_pdbqt
    except:
        return None
