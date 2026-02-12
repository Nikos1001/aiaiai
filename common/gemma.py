
from transformers import AutoModelForCausalLM, AutoTokenizer 
from common import DEVICE
import torch

def load_gemma_tokenizer(model_id: str = "google/gemma-3-4b-pt") -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_id)

def load_gemma(model_id: str = "google/gemma-3-4b-pt") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Gemma 3 4B and its tokenizer."""
    dtype = torch.bfloat16 if DEVICE.type != "cpu" else torch.float32

    print(f"\n>>> Loading model: {model_id}")
    tokenizer = load_gemma_tokenizer(model_id) 

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    print(f"    Model loaded on {model.device}")

    return model, tokenizer

"""
Some notes because dynamic typing is stupid:

The AutoModelForCausalLM type is a convinient wrapper around an underlying pytorch model to aid with inference

To get at the underlying model:
    model.model.language_model
This seems to be the actual pytorch nn.Module.
Then, to access the submodules, do something like model.model.language_model.layers[4].
For the full model structure, just print(model.model.language_model) to get all the attr names.

To use the tokenizer:
encoding: tokenizer('Some important string', return_tensors='pt')
    The `return_tensors='pt'` refers to returning PyTorch tensors
decoding: tokenizer.decode(123)
"""
