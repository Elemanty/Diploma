import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split

from molgpt.fusion_model import FusionConfig, MolGPTWithRNA, load_fusion_checkpoint
from molgpt.tokenizer import JsonSmilesTokenizer
from data.rna_mol_dataset import RNAMolDataset, collate_batch
# scripts/train_rna_fusion.py
import math, os, random
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from molgpt.tokenizer import JsonSmilesTokenizer
from molgpt.fusion_model import FusionConfig, MolGPTWithRNA, load_fusion_checkpoint
from data.rna_mol_dataset import RNAMolDataset



PROP_COLUMNS: Sequence[str] = [
    "MolWt",
    "LogP",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
]


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--tok-json", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True, help="инициализирующий чекпойнт (дообучение)")
    p.add_argument("--data", type=str, required=True, help="data/Data_ML_clean.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--block-size", type=int, default=116)  # <- 116
    p.add_argument("--rna-trainable-layers", type=int, default=2)
    p.add_argument("--save-ckpt", type=str, default="weights/molgpt_rna_fusion_boseos.pt")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p



def load_resized_model(tok_json: str, init_ckpt: str, block_size: int, device: torch.device):
    tokenizer = JsonSmilesTokenizer(tok_json)
    stoi = getattr(tokenizer, "stoi", None) or getattr(tokenizer, "vocab", None)
    assert isinstance(stoi, dict)

    PAD_ID = stoi["<pad>"]
    vocab_size = len(stoi)

    config = FusionConfig(
        vocab_size=vocab_size,
        pad_id=PAD_ID,
        block_size=block_size,
        rna_fm_trainable_layers=args.rna_trainable_layers,
    )
    model = MolGPTWithRNA(config)

    # Загружаем веса и при необходимости расширяем матрицы под vocab_size
    ckpt = torch.load(init_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)

    model_sd = model.state_dict()
    new_sd = {}

    def copy_or_init(key, old_tensor):
        if key not in model_sd:
            return
        tgt = model_sd[key]
        if old_tensor.shape == tgt.shape:
            new_sd[key] = old_tensor
        else:
            # расширяем только матрицы [V,D] и bias [V]
            if old_tensor.dim() == 2 and tgt.dim() == 2 and old_tensor.shape[1] == tgt.shape[1]:
                V_old, D = old_tensor.shape
                V_new, _ = tgt.shape
                out = tgt.clone()
                out[:min(V_old, V_new), :] = old_tensor[:min(V_old, V_new), :]
                with torch.no_grad():
                    std = 1.0 / math.sqrt(D)
                    if V_new > V_old:
                        out[V_old:V_new, :].normal_(mean=0.0, std=std)
                new_sd[key] = out
            elif old_tensor.dim() == 1 and tgt.dim() == 1:
                V_old = old_tensor.shape[0]; V_new = tgt.shape[0]
                out = tgt.clone()
                out[:min(V_old, V_new)] = old_tensor[:min(V_old, V_new)]
                new_sd[key] = out
            # остальное оставляем как инициализировала новая модель

    for k, v in state.items():
        copy_or_init(k, v)

    model_sd.update(new_sd)
    model.load_state_dict(model_sd, strict=False)
    model.to(device).train()
    return tokenizer, model



def make_collate_fn(model_rna_alphabet, stoi, prop_mean=None, prop_std=None):
    batch_converter = model_rna_alphabet.get_batch_converter()
    PAD_ID = stoi["<pad>"]

    prop_mean = torch.tensor(prop_mean, dtype=torch.float32) if prop_mean is not None else None
    prop_std  = torch.tensor(prop_std,  dtype=torch.float32) if prop_std  is not None else None

    def collate(examples):
        import torch
        input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)  # [B,T]
        target    = torch.stack([ex["target"]    for ex in examples], dim=0)  # [B,T]
        props     = torch.stack([ex["props"]     for ex in examples], dim=0)  # [B,P]
        bind      = torch.stack([ex["bind"]      for ex in examples], dim=0)  # [B]

        # нормализация props (train mean/std)
        if prop_mean is not None and prop_std is not None:
            props = (props - prop_mean) / (prop_std + 1e-8)

        seqs = [("rna", ex["sequence"].replace("T","U")) for ex in examples]
        _, _, rna_tokens = batch_converter(seqs)                               # [B,Lr]
        rna_mask = (rna_tokens != model_rna_alphabet.padding_idx).long()       # [B,Lr]

        return {
            "input_ids": input_ids,
            "target": target,
            "props": props,
            "bind": bind,
            "rna_tokens": rna_tokens,
            "rna_mask": rna_mask,
            "ignore_index": PAD_ID,
        }
    return collate



def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # 1) токенизатор/модель
    tokenizer, model = load_resized_model(args.tok_json, args.ckpt, args.block_size, device)
    stoi = getattr(tokenizer, "stoi", None) or getattr(tokenizer, "vocab", None)
    PAD_ID = stoi["<pad>"]

    # 2) датасеты (используем очищенный CSV)
    PROP_COLUMNS = ["MolWt","LogP","NumHAcceptors","TPSA","NumRotatableBonds"]
    LABEL_COLUMN = "ncRNA_Expression_Pattern"
    full = RNAMolDataset(args.data, tokenizer, args.block_size, PROP_COLUMNS, LABEL_COLUMN)

    # split train/val
    val_size = int(len(full) * args.val_split)
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    # 3) посчитаем скейлер props по train
    import pandas as pd
    df_all = pd.read_csv(args.data)
    df_train = df_all.iloc[train_ds.indices] if hasattr(train_ds, "indices") else df_all.sample(frac=1-args.val_split, random_state=args.seed)
    prop_mean = df_train[PROP_COLUMNS].astype(float).mean().values.astype("float32")
    prop_std  = df_train[PROP_COLUMNS].astype(float).std(ddof=0).values.astype("float32")

    # зарегистрируем в модели (как buffers), чтобы попадали в state_dict
    model.register_buffer("scaler_mean", torch.tensor(prop_mean, dtype=torch.float32))
    model.register_buffer("scaler_std",  torch.tensor(prop_std,  dtype=torch.float32))

    # 4) data loaders (collate использует alphabet модели)
    collate = make_collate_fn(model.rna.alphabet, stoi, prop_mean, prop_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # 5) оптимизатор/шедулер/лоссы
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = total_ce = total_bind = 0.0
        steps = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)        # [B,T]
            target    = batch["target"].to(device)           # [B,T]
            props     = batch["props"].to(device)            # [B,P]
            bind      = batch["bind"].to(device)             # [B]
            rna_tokens= batch["rna_tokens"].to(device)       # [B,Lr]
            rna_mask  = batch["rna_mask"].to(device)         # [B,Lr]

            token_types = torch.ones_like(input_ids, device=device)

            # forward: возвращает (logits, bind_logits, ...)
            out = model(input_ids,
                        target=None,   # целевой сдвиг делаем вручную на лоссе
                        token_type_ids=token_types,
                        rna_tokens=rna_tokens,
                        rna_mask=rna_mask,
                        prop=props,
                        bind_labels=bind)

            logits = out[0]                 # [B,T,V]
            bind_logits = out[1] if len(out) > 1 and out[1] is not None else None

            # LM loss (сдвиг на 1)
            # предсказываем токен t из контекста <0..t-1>
            logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            target_flat = target[:, 1:].contiguous().view(-1)
            loss_lm = ce(logits_flat, target_flat)

            # bind loss (если есть)
            if bind_logits is not None:
                loss_bind = F.cross_entropy(bind_logits, bind)
            else:
                loss_bind = torch.tensor(0.0, device=device)

            loss = loss_lm + 0.1 * loss_bind  # подстрой коэффициент при необходимости

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item()
            total_ce   += loss_lm.item()
            total_bind += loss_bind.item()
            steps += 1

        return (total_loss/steps, total_ce/steps, total_bind/steps)

    # 6) обучение
    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_ce, tr_bind = run_epoch(train_loader, train=True)
        va_loss, va_ce, va_bind = run_epoch(val_loader,   train=False)
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} | CE {tr_ce:.4f} | bind {tr_bind:.4f}")
        print(f"Epoch {epoch}: val   loss {va_loss:.4f} | CE {va_ce:.4f} | bind {va_bind:.4f}")

        # простейший чекпоинт по валидации
        if va_loss < best_val:
            best_val = va_loss
            ckpt = {
                "model": model.state_dict(),
                "tokenizer_path": args.tok_json,
                "block_size": args.block_size,
                "scaler_mean": model.scaler_mean.cpu().numpy(),
                "scaler_std":  model.scaler_std.cpu().numpy(),
            }
            Path(os.path.dirname(args.save_ckpt) or ".").mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, args.save_ckpt)
            print(f"[saved] {args.save_ckpt} (val loss {va_loss:.4f})")

if __name__ == "__main__":
    main()
