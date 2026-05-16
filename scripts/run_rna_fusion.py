import argparse
import torch

from molgpt.fusion_model import FusionConfig, MolGPTWithRNA, load_fusion_checkpoint
from molgpt.tokenizer import JsonSmilesTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MolGPT + RNA-FM fusion demo")
    parser.add_argument("--tok-json", required=True, help="Path to stoi JSON used during MolGPT pretraining")
    parser.add_argument("--ckpt", required=True, help="Checkpoint with fused weights")
    parser.add_argument("--rna-seq", required=True, help="RNA sequence (T will be auto-replaced with U)")
    parser.add_argument("--smiles", default="CCO", help="Initial SMILES prompt")
    parser.add_argument("--props", default="0,0,0,0,0", help="Comma-separated property values (5 numbers)")
    parser.add_argument("--bind", type=int, default=0, choices=[0, 1], help="Binding class label")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = JsonSmilesTokenizer(args.tok_json)
    config = FusionConfig(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model = MolGPTWithRNA(config)
    load_fusion_checkpoint(model, args.ckpt, strict=False)
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    prompt_ids = tokenizer.encode(args.smiles)
    if not prompt_ids:
        raise ValueError("Prompt produced empty token sequence; check tokenizer and prompt")
    smiles_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    token_types = torch.ones_like(smiles_tensor, device=device)

    props = torch.tensor([
        [float(x) for x in args.props.split(",")]
    ], dtype=torch.float, device=device)
    if props.size(-1) != config.num_props:
        raise ValueError(f"Expected {config.num_props} properties, got {props.size(-1)}")

    bind = torch.tensor([args.bind], dtype=torch.long, device=device)

    batch_converter = model.rna.alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("rna", args.rna_seq.replace("T", "U"))])
    tokens = tokens.to(device)
    mask = (tokens != model.rna.alphabet.padding_idx).long().to(device)

    with torch.no_grad():
        generated = model.generate(
            smiles_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            token_type_ids=token_types,
            rna_tokens=tokens,
            rna_mask=mask,
            prop=props,
            bind_labels=bind,
        )
    decoded = tokenizer.decode(generated[0].tolist())
    print("Prompt:", args.smiles)
    print("Generated:", decoded)


if __name__ == "__main__":
    main()

