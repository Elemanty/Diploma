import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import fm
except ImportError:  # pragma: no cover
    fm = None


def _build_causal_mask(size: int) -> torch.Tensor:
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return mask.view(1, 1, size, size)


@dataclass
class FusionConfig:
    vocab_size: int
    block_size: int = 54
    pad_id: int = 0
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    cond_dim: int = 256
    num_props: int = 5
    dropout: float = 0.1
    cross_attn_from: int = 4
    prefix_len: int = 10  # сколько "виртуальных" токенов из cond отдавать в cross-attention
    rna_fm_trainable_layers: int = 3  # сколько последних слоев RNA-FM дообучаем (0 = все заморожены)
    use_props_in_generate: bool = False  # генерация опирается только на RNA-эмбеддинги


class CausalSelfAttention(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = _build_causal_mask(config.block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.size()
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y, att


class GPTBlockBase(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_in = self.ln1(x)
        y, att = self.attn(attn_in)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, att


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * emb_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta


class CrossAttention(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.scale = (config.n_embd // config.n_head) ** -0.5
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, c = x.size()
        q = self.q_proj(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = self.k_proj(context).view(b, -1, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.v_proj(context).view(b, -1, self.n_head, c // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.out_proj(y))


class ConditionalBlock(nn.Module):
    def __init__(self, config: FusionConfig, use_cross: bool) -> None:
        super().__init__()
        self.base = GPTBlockBase(config)
        self.film = FiLM(config.cond_dim, config.n_embd)
        self.use_cross = use_cross
        if use_cross:
            self.cross = CrossAttention(config)
            self.ln_cross = nn.LayerNorm(config.n_embd)
        else:
            self.cross = None
            self.ln_cross = None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, att = self.base(x)
        x = self.film(x, cond)
        if self.use_cross and self.cross is not None and context is not None:
            x = x + self.cross(self.ln_cross(x), context, context_mask)
        return x, att


class ConditionalGPT(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        blocks = []
        for i in range(config.n_layer):
            blocks.append(ConditionalBlock(config, use_cross=i >= config.cross_attn_from))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.pad_id = config.pad_id

    def forward(
        self,
        idx: torch.Tensor,
        cond: torch.Tensor,
        token_type_ids: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]:
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")
        token_embeddings = self.tok_emb(idx)
        type_embeddings = self.type_emb(token_type_ids)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + type_embeddings + position_embeddings)
        attn_maps = []
        for block in self.blocks:
            x, att = block(x, cond, context, context_mask)
            attn_maps.append(att)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),                                              #  !!!  было targets.view(-1),
                ignore_index=self.pad_id,
            )
        return logits, loss, tuple(attn_maps)


class RNAEncoder(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        freeze_backbone: bool = True,
        trainable_layers: int = 0,
    ) -> None:
        super().__init__()
        if fm is None:
            raise ImportError("rna-fm is not installed. Install with `pip install rna-fm`. ")
        model, alphabet = fm.pretrained.rna_fm_t12()
        self.model = model
        self.alphabet = alphabet
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        if trainable_layers > 0:
            if not hasattr(self.model, "layers"):
                raise AttributeError("RNA-FM model has no 'layers' attribute; cannot unfreeze by layer.")
            layers = self.model.layers
            n_layers = len(layers)
            n_train = min(trainable_layers, n_layers)
            for layer in layers[-n_train:]:
                for param in layer.parameters():
                    param.requires_grad = True
            if hasattr(self.model, "emb_layer_norm_after"):
                for param in self.model.emb_layer_norm_after.parameters():
                    param.requires_grad = True
        embed_dim = getattr(self.model, 'embed_dim', None)
        if embed_dim is None and hasattr(self.model, 'args'):
            embed_dim = getattr(self.model.args, 'embed_dim', None)
        if embed_dim is None and hasattr(self.model, 'args'):
            embed_dim = getattr(self.model.args, 'encoder_embed_dim', None)
        if embed_dim is None:
            raise AttributeError('Unable to infer RNA-FM embedding size; update RNAEncoder to match fm package version.')
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, cond_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(tokens, repr_layers=[12], return_contacts=False)
        hidden = out["representations"][12]
        proj = self.proj(hidden)
        if attention_mask is None:
            mask = (tokens != self.alphabet.padding_idx).to(hidden.dtype)
        else:
            mask = attention_mask.to(hidden.dtype)
        return proj, mask


class MolGPTWithRNA(nn.Module):
    def __init__(self, config: FusionConfig, freeze_rna: bool = True) -> None:
        super().__init__()
        self.config = config
        self.gpt = ConditionalGPT(config)
        self.rna = RNAEncoder(
            config.cond_dim,
            freeze_backbone=freeze_rna,
            trainable_layers=config.rna_fm_trainable_layers,
        )
        self.prop_proj = nn.Linear(config.num_props, config.cond_dim)
        self.bind_emb = nn.Embedding(2, config.cond_dim)
        self.cond_proj = nn.Linear(config.cond_dim * 3, config.cond_dim)
        self.bind_head = nn.Linear(config.cond_dim, 2)
        self.register_buffer("scaler_mean", torch.zeros(config.num_props))
        self.register_buffer("scaler_std", torch.ones(config.num_props))
        self.pad_id = config.pad_id

        if self.config.prefix_len > 0:
            self.prefix_mlp = nn.Sequential(
                nn.Linear(config.cond_dim, config.n_embd * config.prefix_len),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.prefix_mlp = None

        self.register_buffer("scaler_mean", torch.zeros(config.num_props))
        self.register_buffer("scaler_std", torch.ones(config.num_props))
        self.pad_id = config.pad_id

    def encode_rna(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_repr, mask = self.rna(tokens, attention_mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (seq_repr * mask.unsqueeze(-1)).sum(dim=1) / denom
        return seq_repr, mask, pooled

    def build_condition(
        self,
        prop: torch.Tensor,
        bind_labels: torch.Tensor,
        rna_pooled: torch.Tensor,
        *,
        use_props: bool = True,
    ) -> torch.Tensor:
        if use_props:
            prop_norm = (prop - self.scaler_mean) / (self.scaler_std + 1e-6)
            prop_cond = self.prop_proj(prop_norm)
        else:
            prop_cond = torch.zeros_like(rna_pooled)
        bind_cond = self.bind_emb(bind_labels)
        concat = torch.cat([rna_pooled, prop_cond, bind_cond], dim=-1)
        return self.cond_proj(concat)

    def forward(
        self,
        idx: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        *,
        token_type_ids: Optional[torch.Tensor] = None,
        rna_tokens: Optional[torch.Tensor],
        rna_mask: Optional[torch.Tensor],
        prop: torch.Tensor,
        bind_labels: torch.Tensor,
        use_props: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, ...]]:
        b, t = idx.shape
        if token_type_ids is None:
            token_type_ids = torch.ones((b, t), dtype=torch.long, device=idx.device)

        # 1) RNA-контекст
        seq_repr, mask, pooled = self.encode_rna(rna_tokens, rna_mask)

        # 2) Условный вектор
        cond = self.build_condition(prop, bind_labels, pooled, use_props=use_props)

        # 3) СОБИРАЕМ КОНТЕКСТ: [prefix_tokens] + [rna_tokens]
        if self.prefix_mlp is not None:
            pref = self.prefix_mlp(cond)                              # (B, n_embd * Lp)
            pref = pref.view(b, self.config.prefix_len, self.gpt.config.n_embd)  # (B, Lp, D)
            # Маска: единицы для префикса
            ones = torch.ones(b, self.config.prefix_len, device=seq_repr.device, dtype=mask.dtype)
            context = torch.cat([pref, seq_repr], dim=1)              # (B, Lp + Lrna, D)
            context_mask = torch.cat([ones, mask], dim=1)             # (B, Lp + Lrna)
        else:
            context, context_mask = seq_repr, mask

        # 4) GPT + cross-attn по context
        logits, loss, attn = self.gpt(
            idx, cond, token_type_ids,
            context, context_mask,
            targets=target
        )

        # 5) Биндинг-голова на cond
        bind_logits = self.bind_head(cond)
        return logits, loss, bind_logits, attn


    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        rna_tokens: Optional[torch.Tensor],
        rna_mask: Optional[torch.Tensor],
        prop: torch.Tensor,
        bind_labels: torch.Tensor,
        use_props: Optional[bool] = None,
    ) -> torch.Tensor:
        seq_repr, mask, pooled = self.encode_rna(rna_tokens, rna_mask)
        if use_props is None:
            use_props = self.config.use_props_in_generate
        cond = self.build_condition(prop, bind_labels, pooled, use_props=use_props)
        if self.prefix_mlp is not None:
            b = idx.size(0)
            pref = self.prefix_mlp(cond).view(b, self.config.prefix_len, self.gpt.config.n_embd)
            ones = torch.ones(b, self.config.prefix_len, device=seq_repr.device, dtype=mask.dtype)
            context = torch.cat([pref, seq_repr], dim=1)
            context_mask = torch.cat([ones, mask], dim=1)
        else:
            context, context_mask = seq_repr, mask

        if token_type_ids is None:
            token_type_ids = torch.ones_like(idx)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            types_cond = token_type_ids[:, -self.config.block_size :]
            logits, _, _ = self.gpt(idx_cond, cond, types_cond, context, context_mask)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                top_vals, top_idx = logits.topk(min(top_k, logits.size(-1)), dim=-1)
                probs = F.softmax(top_vals, dim=-1)
                next_token = top_idx.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
            token_type_ids = torch.cat([token_type_ids, torch.ones_like(next_token)], dim=1)
        return idx


def load_fusion_checkpoint(model: MolGPTWithRNA, checkpoint_path: str, strict: bool = False) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"]
    for name in ("prop_proj", "bind_emb"):
        if name in ckpt:
            for key, value in ckpt[name].items():
                state[f"{name}.{key}"] = value
    missing_scalers = []
    if "scaler_mean" in ckpt:
        mean_tensor = torch.as_tensor(ckpt["scaler_mean"], dtype=model.scaler_mean.dtype, device=model.scaler_mean.device)
        model.scaler_mean.copy_(mean_tensor)
    else:
        missing_scalers.append("scaler_mean")
    if "scaler_std" in ckpt:
        std_tensor = torch.as_tensor(ckpt["scaler_std"], dtype=model.scaler_std.dtype, device=model.scaler_std.device)
        model.scaler_std.copy_(std_tensor)
    else:
        missing_scalers.append("scaler_std")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing and strict:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected and strict:
        raise RuntimeError(f"Unexpected keys: {unexpected}")
    if missing:
        print(f"[fusion] missing keys (ignored): {missing}")
    if unexpected:
        print(f"[fusion] unexpected keys (ignored): {unexpected}")
    if missing_scalers:
        print(f"[fusion] scaler tensors not found in checkpoint: {missing_scalers}")

