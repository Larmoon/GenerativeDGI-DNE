from rdkit.Chem import rdchem, AllChem
from rdkit import DataStructs, Chem, RDLogger
import rdkit
import os
import traceback

logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)

class CheckerBase:
    def __init__(self, name, explanation, penalty):
        self.name = name
        self.explanation = explanation
        self.penalty = penalty

    def check(self, smiles):
        raise NotImplementedError("Each checker must implement the 'check' method.")


class MolChecker(CheckerBase):
    def __init__(self, name, explanation, penalty):
        super().__init__(name, explanation, penalty)


# Specific checkers
class NumAtomsMolChecker(MolChecker):
    def __init__(self):
        super().__init__("num_atoms_equals_zero", "number of atoms less than 1", 8) #adapt as needed

    def check(self, smiles):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        return mol is not None and mol.GetNumAtoms() < 2




class KekulizeErrorMolChecker(MolChecker):
    def __init__(self):
        super().__init__("kekulization_error", "Kekulization error for SMILES", 8) #adapt as needed

    def check(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if not mol:
                return False
            Chem.Kekulize(mol)
            return False
        except rdchem.KekulizeException:
            return True
        except Exception as e:
            return False


class ValenceErrorMolChecker(MolChecker):
    def __init__(self):
        super().__init__("valence_error", "Contains valence different from permitted", 9) #adapt as needed

    def check(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if not mol:
                return False
            Chem.SanitizeMol(mol)  # This will raise an exception if there are valence issues
            return False
        except rdchem.AtomValenceException:
            return True
        except Exception as e:
            return False


class ParserErrorMolChecker(MolChecker):
    def __init__(self):
        super().__init__("parser_error", "Smiles could not be parsed", 10) #adapt as needed

    def check(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            return mol is None
        except Exception:
            return True

# Function to run all checks
def check_smiles(smiles):
    checkers = [
        NumAtomsMolChecker(),
        # KekulizeErrorMolChecker(),
        ValenceErrorMolChecker(),
        ParserErrorMolChecker(),
    ]
    results = []
    for checker in checkers:
        if checker.check(smiles):
            results.append((checker.penalty, checker.explanation))
    return results
