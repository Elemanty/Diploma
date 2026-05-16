
import torch
import torch.nn as nn
from typing import List, Tuple

class RNAFMEncoderFM(nn.Module):
    """
    Обёртка над RNA-FM (fm.pretrained.rna_fm_t12()).
    Возвращает по-токенные скрытые состояния -> линейная проекция в 256.
    """
    def __init__(self, d_out_proj: int = 256, device: str = "cuda"):
        super().__init__()
        try:
            import fm  # импорт внутри, чтобы в случае отсутствия выдать понятную подсказку
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Модуль 'fm' не найден в активном интерпретаторе. "
                "Установите пакет именно сюда:\n"
                "  import sys, subprocess; subprocess.check_call([sys.executable,'-m','pip','install','-U','rna-fm'])"
            ) from e
        self.fm = fm
        self.model, self.alphabet = self.fm.pretrained.rna_fm_t12()  # 12 слоёв, ~640-d
        self.batch_converter = self.alphabet.get_batch_converter()
        # LazyLinear подхватит входную размерность при первом forward
        self.proj = nn.LazyLinear(d_out_proj)
        self.device = device
        self.to(device)
        self.model.eval()  # по умолчанию фризим весь RNA-FM

    @torch.no_grad()
    def tokenize(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Батч токенов и маска.
        RNA-FM обучался на 'U'; на всякий меняем 'T'->'U'.
        """
        data = [(f"seq_{i}", s.replace("T", "U")) for i, s in enumerate(seqs)]
        _, _, tokens = self.batch_converter(data)  # [B, L]
        attn_mask = (tokens != self.alphabet.padding_idx).to(torch.long)  # [B, L]
        return tokens, attn_mask

    @torch.no_grad()
    def tokenize_single(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self.tokenize([seq])
        return ids[0], mask[0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        input_ids = input_ids.to(self.device)
        out = self.model(input_ids, repr_layers=[12], return_contacts=False)
        hs = out["representations"][12]   # [B, L, H≈640]
        hs = self.proj(hs)                # -> [B, L, 256]
        return hs, attention_mask.to(self.device)

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False
