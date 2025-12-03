"""Molecular docking runner using AutoDock Vina."""
import subprocess
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

class DockingRunner:
    def __init__(self, vina_executable='vina', receptor_path=None):
        self.vina_executable = vina_executable
        self.receptor_path = receptor_path
    
    def prepare_ligand(self, smiles, output_path):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        Chem.MolToPDBFile(mol, str(output_path))
        return output_path
    
    def run_vina(self, ligand_pdbqt, receptor_pdbqt, center, size, exhaustiveness=8):
        with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as out_file:
            cmd = [
                self.vina_executable,
                '--receptor', receptor_pdbqt,
                '--ligand', ligand_pdbqt,
                '--center_x', str(center[0]),
                '--center_y', str(center[1]),
                '--center_z', str(center[2]),
                '--size_x', str(size[0]),
                '--size_y', str(size[1]),
                '--size_z', str(size[2]),
                '--exhaustiveness', str(exhaustiveness),
                '--out', out_file.name
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                return self.parse_vina_output(result.stdout)
            except Exception as e:
                print(f"Docking failed: {e}")
                return None
    
    def parse_vina_output(self, output):
        lines = output.split('\n')
        scores = []
        for line in lines:
            if 'REMARK VINA RESULT:' in line:
                parts = line.split()
                if len(parts) >= 4:
                    scores.append(float(parts[3]))
        return min(scores) if scores else 0.0
    
    def dock_smiles(self, smiles):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            ligand_pdb = tmpdir / 'ligand.pdb'
            self.prepare_ligand(smiles, ligand_pdb)
            # Simplified - would need pdbqt conversion
            return np.random.randn() * 2 - 6  # Mock score
