"""
Feature Fuzzing — find token sequences that maximally activate a chosen SAE feature.

Uses Greedy Coordinate Gradient (GCG) search: at each step, computes the
gradient of the target feature's pre-activation w.r.t. the input embeddings,
then uses that gradient to find the best single-token substitution.

Usage:
    uv run python -m experiments.feature_fuzz
"""

from __future__ import annotations

import torch

from common import DEVICE
from common.gemma import load_gemma
from common.gemma_scope import load_sae

# ---------------------------------------------------------------------------
# Configuration — model / SAE / feature
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-3-4b-pt"
LAYER = 20
SITE = "resid_post_all"
WIDTH = "16k"
L0 = "small"
FEATURE_IDX = 100
SEQ_LEN = 16
TARGET_POS = -1  # token position at which to maximize the feature (-1 = last)

# ---------------------------------------------------------------------------
# Optimization hyperparameters
# ---------------------------------------------------------------------------
NUM_STEPS = 300
LOG_EVERY = 5


def main() -> None:
    # ── load model & SAE ──────────────────────────────────────────────────
    model, tokenizer = load_gemma(MODEL_ID)
    sae = load_sae(layer=LAYER, width=WIDTH, l0=L0, site=SITE)

    # freeze everything
    for p in model.parameters():
        p.requires_grad_(False)
    for p in sae.parameters():
        p.requires_grad_(False)

    # embedding matrix → float32 on DEVICE (pre-computed once)
    embed_matrix = model.model.language_model.embed_tokens.weight
    vocab_size, d_model = embed_matrix.shape
    model_dtype = embed_matrix.dtype
    model_device = embed_matrix.device
    embed_f32 = embed_matrix.detach().to(device=DEVICE, dtype=torch.float32)

    # SAE encoder column for the target feature
    w_enc_feat = sae.w_enc[:, FEATURE_IDX].to(DEVICE)  # (d_model,)
    b_enc_feat = sae.b_enc[FEATURE_IDX].to(DEVICE)     # scalar

    print(f"\n{'=' * 70}")
    print("Feature Fuzzing (GCG)")
    print(f"{'=' * 70}")
    print(f"  model      {MODEL_ID}")
    print(f"  layer      {LAYER}   site {SITE}")
    print(f"  SAE        {WIDTH} / {L0}")
    print(f"  feature    {FEATURE_IDX}")
    print(f"  seq_len    {SEQ_LEN}   target_pos {TARGET_POS}")
    print(f"  vocab      {vocab_size}   d_model {d_model}")
    print(f"  device     {DEVICE}")

    # ── initialise with random tokens ─────────────────────────────────────
    token_ids = torch.randint(0, vocab_size, (SEQ_LEN,), device=DEVICE)

    # ── optimisation loop (greedy coordinate gradient) ────────────────────
    print(f"\nOptimising for {NUM_STEPS} steps …\n")

    for step in range(NUM_STEPS):
        # embed current tokens with gradient tracking
        embeds = embed_f32[token_ids].unsqueeze(0).detach().clone()
        embeds.requires_grad_(True)

        # forward pass — capture activations at target layer
        captured: dict[str, torch.Tensor] = {}

        def _hook(_mod, _inp, out):
            captured["act"] = out[0] if isinstance(out, tuple) else out

        handle = model.model.language_model.layers[LAYER].register_forward_hook(_hook)
        model(inputs_embeds=embeds.to(device=model_device, dtype=model_dtype))
        handle.remove()

        activations = captured["act"].to(device=DEVICE, dtype=torch.float32)
        feature_act = activations[0, TARGET_POS] @ w_enc_feat + b_enc_feat
        current_val = feature_act.item()

        feature_act.backward()

        # ── greedy token substitution ─────────────────────────────────────
        # score each vocab token at each position via first-order approx:
        #   delta_feat ≈ grad[pos] · (embed[new] - embed[old])
        grad = embeds.grad[0]                        # (S, d_model)
        scores = grad @ embed_f32.T                  # (S, vocab_size)
        current_scores = scores[torch.arange(SEQ_LEN, device=DEVICE), token_ids]
        improvements = scores - current_scores.unsqueeze(1)

        # don't "swap" to the same token
        improvements[torch.arange(SEQ_LEN, device=DEVICE), token_ids] = float('-inf')

        flat_idx = improvements.view(-1).argmax().item()
        best_pos = flat_idx // vocab_size
        best_tok = flat_idx % vocab_size

        if improvements[best_pos, best_tok] > 0:
            token_ids = token_ids.clone()
            token_ids[best_pos] = best_tok

        # ── logging ───────────────────────────────────────────────────────
        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
            swap_str = tokenizer.decode(torch.tensor([best_tok]))
            print(
                f"step {step:4d} │ feat {current_val:+10.2f} │ "
                f"swap pos {best_pos:2d} → {swap_str!r:>10s} │ "
                f"{decoded!r}"
            )

    # ── final result ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RESULT")
    print(f"{'=' * 70}")

    token_strs = [tokenizer.decode(t) for t in token_ids]
    final_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    print(f"\n  text       {final_text!r}")
    print(f"  token_ids  {token_ids.tolist()}")
    print(f"  tokens     {token_strs}")

    # ── verification with full SAE encode (includes JumpReLU threshold) ───
    print(f"\n  Verification (SAE encode with threshold):")
    with torch.no_grad():
        input_ids = token_ids.unsqueeze(0).to(model_device)

        captured.clear()

        def _vhook(_mod, _inp, out):
            captured["act"] = (out[0] if isinstance(out, tuple) else out).detach()

        handle = model.model.language_model.layers[LAYER].register_forward_hook(_vhook)
        model(input_ids)
        handle.remove()

        real_act = captured["act"].to(device=DEVICE, dtype=torch.float32)
        real_features = sae.encode(real_act)
        target_val = real_features[0, TARGET_POS, FEATURE_IDX].item()

        feat_vec = real_features[0, TARGET_POS]
        active = torch.nonzero(feat_vec).squeeze(-1)

        print(f"    feature {FEATURE_IDX} activation = {target_val:.4f}")

        if len(active) > 0:
            k = min(10, len(active))
            top_vals, top_idx = feat_vec[active].topk(k)
            top_feats = active[top_idx]
            print(f"    top-{k} features at pos {TARGET_POS}:")
            for f_id, v in zip(top_feats, top_vals):
                tag = " <-- TARGET" if f_id.item() == FEATURE_IDX else ""
                print(f"      f{f_id.item():>6d} = {v.item():.4f}{tag}")

            all_active_vals = feat_vec[active]
            rank = (all_active_vals > target_val).sum().item() + 1
            print(f"    rank: {rank} / {len(active)} active features")
        else:
            print(f"    no features active at position {TARGET_POS}")


if __name__ == "__main__":
    main()
