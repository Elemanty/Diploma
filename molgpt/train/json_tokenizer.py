import json

class JsonSmilesTokenizer:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.stoi = json.load(f)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, smiles):
        """Преобразует строку SMILES в список индексов"""
        return [self.stoi[s] for s in smiles if s in self.stoi]

    def decode(self, tokens):
        """Преобразует индексы обратно в SMILES"""
        return "".join([self.itos[i] for i in tokens])
