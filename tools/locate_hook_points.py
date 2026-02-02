import re
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL = "openvla/openvla-7b"  # or your LIBERO-finetuned checkpoint


def find_mlp_out_candidates(model):
    """
    Returns (name, module) pairs that are likely 'mlp_out' points.
    Works across common LLaMA / transformer implementations by pattern matching.
    """
    candidates = []
    for name, mod in model.named_modules():
        # Common patterns:
        # - model.layers.<i>.mlp.down_proj (LLaMA-style)
        # - transformer.h.<i>.mlp.c_proj (GPT-style)
        # - blocks.<i>.mlp.fc2 (some ViT/LM variants)
        if re.search(r"(layers\.\d+\.mlp\.down_proj$)|(h\.\d+\.mlp\.c_proj$)|(blocks\.\d+\.mlp\.(fc2|proj)$)", name):
            candidates.append((name, mod))
    return candidates


def main():
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    cands = find_mlp_out_candidates(model)
    print(f"Found {len(cands)} candidate mlp_out modules.")
    for name, _ in cands[:40]:
        print("  ", name)


if __name__ == "__main__":
    main()
