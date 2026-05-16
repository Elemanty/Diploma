import torch
from molgpt.rna_fm_wrapper_fm import RNAFMEncoderFM

rna = RNAFMEncoderFM(d_out_proj=256, device="cpu")
ids, mask = rna.tokenize(["UAGCUUAUCAGACUGAUGUUG", "AUGCUUAGC"])
print("ids:", ids.shape, "mask:", mask.shape)

with torch.no_grad():
    hs, m = rna(ids, mask)
print("hs:", hs.shape, "device:", hs.device)
assert hs.shape[-1] == 256
print("OK")
