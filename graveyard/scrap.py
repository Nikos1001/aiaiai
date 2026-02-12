"""
Gemma 3 4B + Gemma Scope 2 — Local Interpretability Playground

Downloads Gemma 3 4B (pretrained) and Gemma Scope 2 SAEs,
then runs several experiments you can modify and extend.

Usage:
    uv run main.py                    # Run all demos
    uv run main.py --demo activations # Run a specific demo
    uv run main.py --layer 8           # Use a different layer (0-33)
    uv run main.py --width 262k       # Use 262k-wide SAE (vs default 16k)
    uv run main.py --l0 big           # Use less-sparse SAE variant
"""

from __future__ import annotations

import argparse
import functools

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ---------------------------------------------------------------------------
# JumpReLU Sparse Autoencoder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_residual_activations(
    model, input_ids: torch.Tensor, target_layer: int
) -> torch.Tensor:
    """Run a forward pass and grab the residual stream output at `target_layer`."""
    captured = {}

    def hook_fn(mod, inp, out):
        captured["act"] = out.detach()

    handle = model.model.language_model.layers[target_layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return captured["act"]


# ---------------------------------------------------------------------------
# Demo: Inspect sparse activations
# ---------------------------------------------------------------------------

def demo_activations(model, tokenizer, sae, layer: int):
    """Show which SAE features fire for a given prompt."""
    print("\n" + "=" * 70)
    print("DEMO: Sparse feature activations")
    print("=" * 70)

    prompt = "The capital of France is"
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(model.device)
    token_strs = [tokenizer.decode(t) for t in input_ids[0]]

    residual = get_residual_activations(model, input_ids, layer)
    features = sae.encode(residual.to(torch.float32).to(DEVICE))

    print(f"\nPrompt: {prompt!r}")
    print(f"Tokens: {token_strs}")
    print(f"Residual shape: {residual.shape}")
    print(f"Feature shape:  {features.shape}")

    for pos, tok in enumerate(token_strs):
        feat_vec = features[0, pos]
        active = torch.nonzero(feat_vec).squeeze(-1)
        magnitudes = feat_vec[active]
        top_k = min(5, len(active))
        if top_k > 0:
            top_vals, top_idx = magnitudes.topk(top_k)
            top_features = active[top_idx]
            pairs = [f"f{f.item()}={v.item():.2f}" for f, v in zip(top_features, top_vals)]
        else:
            pairs = ["(none)"]
        print(f"  [{pos}] {tok!r:>15s}  active={len(active):4d}  top: {', '.join(pairs)}")


# ---------------------------------------------------------------------------
# Demo: Reconstruction quality
# ---------------------------------------------------------------------------

def demo_reconstruction(model, tokenizer, sae, layer: int):
    """Measure how well the SAE reconstructs the residual stream."""
    print("\n" + "=" * 70)
    print("DEMO: Reconstruction quality")
    print("=" * 70)

    prompts = [
        "The quick brown fox jumps over the lazy dog",
        "import torch\nmodel = AutoModel.from_pretrained",
        "In 1969, Apollo 11 landed on the Moon",
    ]

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        residual = get_residual_activations(model, input_ids, layer)
        residual_f32 = residual.to(torch.float32).to(DEVICE)

        reconstructed = sae(residual_f32)

        mse = (residual_f32 - reconstructed).pow(2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            residual_f32.reshape(-1, residual_f32.shape[-1]),
            reconstructed.reshape(-1, reconstructed.shape[-1]),
        ).mean().item()

        # Variance explained
        var_original = residual_f32.var().item()
        var_explained = 1.0 - mse / var_original if var_original > 0 else 0.0

        print(f"\n  Prompt: {prompt[:60]!r}...")
        print(f"    MSE:              {mse:.6f}")
        print(f"    Cosine similarity: {cos_sim:.6f}")
        print(f"    Variance explained: {var_explained:.4f}")


# ---------------------------------------------------------------------------
# Demo: Generation with SAE reconstruction in the loop
# ---------------------------------------------------------------------------

def demo_sae_generation(model, tokenizer, sae, layer: int):
    """Generate text with and without SAE reconstruction to see the impact."""
    print("\n" + "=" * 70)
    print("DEMO: Generation — original vs SAE-reconstructed")
    print("=" * 70)

    prompt = "# Counting sort\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # Normal generation
    print(f"\n  Prompt: {prompt!r}")
    print("\n  --- Original ---")
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=40, do_sample=False, streamer=streamer)

    # Generation with SAE reconstruction
    def sae_hook(mod, inp, out):
        h = out.to(torch.float32).to(DEVICE)
        h_recon = sae(h)
        return h_recon.to(out.dtype).to(out.device)

    handle = model.model.language_model.layers[layer].register_forward_hook(sae_hook)
    print("\n  --- SAE-reconstructed ---")
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=40, do_sample=False, streamer=streamer)
    handle.remove()


# ---------------------------------------------------------------------------
# Demo: Feature steering
# ---------------------------------------------------------------------------

def demo_steering(model, tokenizer, sae, layer: int):
    """Amplify or suppress specific SAE features during generation."""
    print("\n" + "=" * 70)
    print("DEMO: Feature steering")
    print("=" * 70)

    prompt = "I think the best food is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # First, find the most active features on this prompt
    residual = get_residual_activations(model, input_ids, layer)
    features = sae.encode(residual.to(torch.float32).to(DEVICE))
    avg_activation = features[0].mean(dim=0)  # average across positions
    top_vals, top_indices = avg_activation.topk(10)

    print(f"\n  Prompt: {prompt!r}")
    print(f"  Top 10 avg-active features: "
          + ", ".join(f"f{i.item()}({v.item():.2f})" for i, v in zip(top_indices, top_vals)))

    # Normal generation for baseline
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=30, do_sample=False)
    print(f"\n  Baseline:  {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # Steer by amplifying each of the top-3 features
    for rank in range(3):
        feat_idx = top_indices[rank].item()

        def steering_hook(mod, inp, out, *, fidx, scale):
            h = out.to(torch.float32).to(DEVICE)
            f = sae.encode(h)
            f[:, :, fidx] *= scale
            h_new = sae.decode(f)
            return h_new.to(out.dtype).to(out.device)

        handle = model.model.language_model.layers[layer].register_forward_hook(
            functools.partial(steering_hook, fidx=feat_idx, scale=5.0)
        )
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=30, do_sample=False)
        handle.remove()
        print(f"  Steer f{feat_idx} x5: {tokenizer.decode(out[0], skip_special_tokens=True)}")


# ---------------------------------------------------------------------------
# Demo: Feature ablation — kill a feature and see what changes
# ---------------------------------------------------------------------------

def demo_ablation(model, tokenizer, sae, layer: int):
    """Zero out individual features and measure how logits change."""
    print("\n" + "=" * 70)
    print("DEMO: Feature ablation — impact on next-token prediction")
    print("=" * 70)

    prompt = "The Eiffel Tower is located in"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Baseline logits
    with torch.no_grad():
        baseline_logits = model(input_ids).logits[0, -1].float()  # last token
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    top5_baseline = baseline_probs.topk(5)

    print(f"\n  Prompt: {prompt!r}")
    print(f"  Baseline top-5 predictions:")
    for prob, idx in zip(top5_baseline.values, top5_baseline.indices):
        print(f"    {tokenizer.decode(idx):>15s}  {prob.item():.4f}")

    # Find active features at last token position
    residual = get_residual_activations(model, input_ids, layer)
    features = sae.encode(residual.to(torch.float32).to(DEVICE))
    last_tok_feats = features[0, -1]
    active = torch.nonzero(last_tok_feats).squeeze(-1)
    magnitudes = last_tok_feats[active]
    _, sort_idx = magnitudes.sort(descending=True)
    top_features = active[sort_idx[:10]]

    print(f"\n  Ablating top-{len(top_features)} features at last position:")

    for feat_idx in top_features:
        fidx = feat_idx.item()

        def ablation_hook(mod, inp, out, *, kill_idx):
            h = out.to(torch.float32).to(DEVICE)
            f = sae.encode(h)
            f[:, :, kill_idx] = 0.0
            h_new = sae.decode(f)
            return h_new.to(out.dtype).to(out.device)

        handle = model.model.language_model.layers[layer].register_forward_hook(
            functools.partial(ablation_hook, kill_idx=fidx)
        )
        with torch.no_grad():
            ablated_logits = model(input_ids).logits[0, -1].float()
        handle.remove()

        ablated_probs = torch.softmax(ablated_logits, dim=-1)
        top1_token = tokenizer.decode(ablated_probs.argmax())
        kl_div = torch.nn.functional.kl_div(
            ablated_probs.log(), baseline_probs, reduction="sum"
        ).item()
        print(f"    Kill f{fidx:>6d}: top1={top1_token!r:>12s}  "
              f"KL(ablated||base)={kl_div:.4f}  "
              f"orig_mag={last_tok_feats[fidx].item():.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEMOS = {
    "activations": demo_activations,
    "reconstruction": demo_reconstruction,
    "generation": demo_sae_generation,
    "steering": demo_steering,
    "ablation": demo_ablation,
}


def main():
    parser = argparse.ArgumentParser(description="Gemma Scope 2 playground")
    parser.add_argument(
        "--demo", choices=list(DEMOS.keys()) + ["all"], default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument(
        "--layer", type=int, default=20,
        help="Transformer layer for SAE (0-33, default: 20)",
    )
    parser.add_argument(
        "--width", default="16k",
        choices=["16k", "262k"],
        help="SAE width / feature count (default: 16k)",
    )
    parser.add_argument(
        "--l0", default="small",
        choices=["small", "big"],
        help="SAE sparsity level (default: small)",
    )
    parser.add_argument(
        "--model", default="google/gemma-3-4b-it",
        help="HuggingFace model ID (default: google/gemma-3-4b-it)",
    )
    args = parser.parse_args()

    model, tokenizer = load_gemma(args.model)
    sae = load_sae(layer=args.layer, width=args.width, l0=args.l0)

    demos_to_run = DEMOS if args.demo == "all" else {args.demo: DEMOS[args.demo]}
    for name, demo_fn in demos_to_run.items():
        try:
            demo_fn(model, tokenizer, sae, args.layer)
        except Exception as e:
            print(f"\n!!! Demo {name!r} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All demos complete. Happy poking!")
    print("=" * 70)


if __name__ == "__main__":
    main()
