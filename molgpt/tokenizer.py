import json
from typing import Iterable, List


class JsonSmilesTokenizer:
    def __init__(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.stoi = {token: int(idx) for token, idx in raw.items()}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.tokens = sorted(self.stoi.keys(), key=len, reverse=True)
        self.vocab_size = len(self.stoi)
        self.pad_id = self.stoi.get("<pad>", self.stoi.get("<", 0))

    def encode(self, smiles: str) -> List[int]:
        output: List[int] = []
        i = 0
        while i < len(smiles):
            matched = False
            for token in self.tokens:
                if smiles.startswith(token, i):
                    output.append(self.stoi[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                # skip unknown character
                i += 1
        if not output:
            raise ValueError(f"Failed to tokenize SMILES: {smiles}")
        return output

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids if int(i) != self.pad_id)
