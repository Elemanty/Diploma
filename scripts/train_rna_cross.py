from molgpt.rna_fm_wrapper_fm import RNAFMEncoderFM
from data.datasets import RNAMolDataset, collate_fn

# ...
rna_enc = RNAFMEncoderFM(d_out_proj=256, device="cuda")
rna_enc.freeze_all()  # фризим весь RNA-FM на первом этапе

ds = RNAMolDataset(
    df,
    smiles_tokenizer=smiles_tokenizer,
    rna_tokenizer=rna_enc.tokenize_single,  # <── одиночный токенайзер
    prop_scaler=scale_props
)

dl = DataLoader(ds, batch_size=64, shuffle=True,
                collate_fn=lambda b: collate_fn(b, pad_token_id=smiles_tokenizer.pad_id))

# далее — MolGPTWithRNA как в моём шаблоне:
model = MolGPTWithRNA(
    gpt=gpt, rna_encoder=rna_enc,
    prefix_tokens=10, fuse_from_layer=4, d_model=256, cond_dim=256
).cuda()
