# data/rna_mol_dataset.py
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd

class RNAMolDataset(Dataset):
    def __init__(self, csv_path, tokenizer, block_size: int,
                 prop_cols, label_col):
        self.df = pd.read_csv(csv_path)
        self.tok = tokenizer
        self.block_size = block_size
        self.prop_cols = list(prop_cols)
        self.label_col = label_col

        # ids спец-токенов из stoi/vocab
        stoi = getattr(tokenizer, "stoi", None) or getattr(tokenizer, "vocab", None)
        assert isinstance(stoi, dict), "Tokenizer должен иметь stoi/vocab (dict)."
        self.PAD_ID = stoi["<pad>"]
        self.BOS_ID = stoi["<bos>"]
        self.EOS_ID = stoi["<eos>"]

    def __len__(self):
        return len(self.df)

    def _encode_smiles(self, s: str):
        ids = self.tok.encode(s)                  # базовые токены без спец-токенов
        ids = [self.BOS_ID] + ids + [self.EOS_ID] # добавляем BOS/EOS
        # паддинг/обрезка
        if len(ids) > self.block_size:
            ids = ids[:self.block_size]
        elif len(ids) < self.block_size:
            ids = ids + [self.PAD_ID] * (self.block_size - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self._encode_smiles(row["SMILES"])
        y = x.clone()  # модель сдвинет сама, либо сдвигай в тренере

        props = torch.tensor(row[self.prop_cols].values, dtype=torch.float32)
        bind  = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        # sequence оставляем, RNA токены соберём в collate_fn (быстрее и удобнее)
        return {
            "input_ids": x,       # [T]
            "target": y,          # [T] (или None, если сдвигаешь в модели)
            "props": props,       # [P]
            "bind": bind,         # []
            "sequence": row["Sequence"],
        }
