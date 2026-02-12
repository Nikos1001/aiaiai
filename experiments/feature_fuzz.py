"""
Feature Fuzzing — find token sequences that maximally activate a chosen SAE feature.

Optimizes soft token distributions via gradient descent to maximize a target
feature's pre-activation in a Gemma Scope SAE, with entropy regularization
to push toward discrete (one-hot) token selections.

Usage:
    uv run experiments/feature_fuzz.py
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
LR = 1e-1
ONEHOT_WEIGHT = 0.1   # weight for entropy regularisation term
TEMP_INIT = 2.0       # softmax temperature at step 0  (flat)
TEMP_FINAL = 0.1      # softmax temperature at final step (sharp)
LOG_EVERY = 5


def main() -> None:
    # ── load model & SAE ──────────────────────────────────────────────────
    model, tokenizer = load_gemma(MODEL_ID)
    sae = load_sae(layer=LAYER, width=WIDTH, l0=L0, site=SITE)

    # freeze everything — we only optimise the token logits
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

    # SAE encoder column for the target feature (fully differentiable path
    # that bypasses the JumpReLU threshold)
    w_enc_feat = sae.w_enc[:, FEATURE_IDX].to(DEVICE)  # (d_model,)
    b_enc_feat = sae.b_enc[FEATURE_IDX].to(DEVICE)     # scalar

    print(f"\n{'=' * 70}")
    print("Feature Fuzzing")
    print(f"{'=' * 70}")
    print(f"  model      {MODEL_ID}")
    print(f"  layer      {LAYER}   site {SITE}")
    print(f"  SAE        {WIDTH} / {L0}")
    print(f"  feature    {FEATURE_IDX}")
    print(f"  seq_len    {SEQ_LEN}   target_pos {TARGET_POS}")
    print(f"  vocab      {vocab_size}   d_model {d_model}")
    print(f"  device     {DEVICE}")

    # ── optimisable parameters ────────────────────────────────────────────
    token_logits = torch.randn(
        1, SEQ_LEN, vocab_size, device=DEVICE, dtype=torch.float32,
    )
    token_logits.requires_grad_(True)

    optimizer = torch.optim.Adam([token_logits], lr=LR)

    # ── optimisation loop ─────────────────────────────────────────────────
    print(f"\nOptimising for {NUM_STEPS} steps …\n")

    for step in range(NUM_STEPS):
        optimizer.zero_grad()

        # temperature annealing (high → low sharpens soft distributions)
        t = step / max(NUM_STEPS - 1, 1)
        temp = TEMP_INIT + (TEMP_FINAL - TEMP_INIT) * t

        # soft one-hot → soft embeddings
        token_probs = torch.softmax(token_logits / temp, dim=-1)
        soft_embeds = token_probs @ embed_f32  # (1, seq_len, d_model)

        # forward pass — capture residual-stream activations at target layer
        captured: dict[str, torch.Tensor] = {}

        def _hook(_mod, _inp, out):
            captured["act"] = out[0] if isinstance(out, tuple) else out

        handle = model.model.language_model.layers[LAYER].register_forward_hook(_hook)
        model(inputs_embeds=soft_embeds.to(device=model_device, dtype=model_dtype))
        handle.remove()

        activations = captured["act"].to(device=DEVICE, dtype=torch.float32)

        # feature pre-activation at the target position (differentiable)
        feature_act = activations[0, TARGET_POS] @ w_enc_feat + b_enc_feat

        # one-hot regularisation: minimise entropy of token distributions
        entropy = -(token_probs * torch.log(token_probs + 1e-10)).sum(dim=-1).mean()

        loss = -feature_act + ONEHOT_WEIGHT * entropy
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            hard_tokens = token_logits.argmax(dim=-1)[0]
            decoded = tokenizer.decode(hard_tokens, skip_special_tokens=False)
            max_probs = token_probs.max(dim=-1).values[0]
            print(
                f"step {step:4d} │ feat {feature_act.item():+10.2f} │ "
                f"ent {entropy.item():7.2f} │ "
                f"conf {max_probs.mean().item():.3f}/{max_probs.min().item():.3f} │ "
                f"{decoded!r}"
            )
            # show probability window around argmax at TARGET_POS
            probs_at_target = token_probs[0, TARGET_POS].detach()
            peak = probs_at_target.argmax().item()
            radius = 3
            lo = max(peak - radius, 0)
            hi = min(peak + radius + 1, vocab_size)
            window = probs_at_target[lo:hi].tolist()
            fmt = ", ".join(f"{v:.4f}" for v in window)
            print(f"         │ probs[{TARGET_POS}] around argmax {peak}: [{fmt}]")

    # ── final result ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RESULT")
    print(f"{'=' * 70}")

    final_ids = token_logits.argmax(dim=-1)[0]
    final_text = tokenizer.decode(final_ids, skip_special_tokens=False)
    token_strs = [tokenizer.decode(t) for t in final_ids]

    print(f"\n  text       {final_text!r}")
    print(f"  token_ids  {final_ids.tolist()}")
    print(f"  tokens     {token_strs}")

    # ── verification with hard (discrete) tokens ──────────────────────────
    print(f"\n  Verification (hard-token forward pass):")
    with torch.no_grad():
        input_ids = final_ids.unsqueeze(0).to(model_device)

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
