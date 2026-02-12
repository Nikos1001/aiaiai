
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from common import DEVICE

class JumpReLUSAE(nn.Module):
    """
    JumpReLU SAE as used by Gemma Scope 2.
    Encodes residual-stream activations into sparse features and decodes back.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = x @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        return mask * torch.nn.functional.relu(pre_acts)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def load_sae(
    layer: int = 20,
    width: str = "16k",
    l0: str = "small",
    repo_id: str = "google/gemma-scope-2-4b-pt",
    site: str = "resid_post_all",
) -> JumpReLUSAE:
    """
    Download and load a Gemma Scope 2 SAE from HuggingFace.

    Repo layout is flat:
        {site}/layer_{N}_width_{W}_l0_{L}/params.safetensors

    Args:
        layer:   Transformer layer index (0-33 for Gemma 3 4B).
        width:   Feature count — "16k" or "262k".
        l0:      Sparsity level — "small" (~20 active) or "big" (~100+ active).
        repo_id: HuggingFace repo for the SAE weights.
        site:    Activation site folder in the repo.
    """
    from safetensors import safe_open

    folder = f"{site}/layer_{layer}_width_{width}_l0_{l0}"
    filename = f"{folder}/params.safetensors"
    print(f"\n>>> Downloading SAE: {repo_id} / {filename}")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    tensors = {}
    with safe_open(local_path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    d_model, d_sae = tensors["w_enc"].shape
    print(f"    SAE dimensions: d_model={d_model}, d_sae={d_sae}")
    print(f"    Parameters: {list(tensors.keys())}")

    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(tensors)
    sae.to(dtype=torch.float32, device=DEVICE)
    sae.eval()
    print(f"    SAE loaded on {DEVICE}")
    return sae
