import random
from rdkit import Chem

class SmilesEnumerator(object):
    """
    Реализация SMILES аугментации:
    - случайная перестановка атомов
    - канонизация при необходимости
    """

    def __init__(self, canonical=True):
        """
        canonical: если True → возвращает канонические SMILES
                   если False → возвращает случайный SMILES
        """
        self.canonical = canonical

    def randomize_smiles(self, smiles):
        """
        Возвращает рандомизированный SMILES (сохраняя структуру молекулы).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if self.canonical:
            # каноническое представление
            return Chem.MolToSmiles(mol, canonical=True)

        # перемешиваем порядок атомов
        atoms = list(mol.GetAtoms())
        idxs = list(range(len(atoms)))
        random.shuffle(idxs)

        new_mol = Chem.RenumberAtoms(mol, idxs)
        return Chem.MolToSmiles(new_mol, canonical=False)

    def __call__(self, smiles):
        """
        Если подать список SMILES → вернёт список аугментированных.
        Если одну строку → вернёт одну строку.
        """
        if isinstance(smiles, list):
            return [self.randomize_smiles(s) for s in smiles]
        return self.randomize_smiles(smiles)
