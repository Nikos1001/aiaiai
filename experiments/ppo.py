
from common.gemma import load_gemma, load_gemma_transformer
from transformers import AutoModelForCausalLM
import torch

MODEL_ID = "google/gemma-3-4b-pt"

model_1, tokenizer = load_gemma(MODEL_ID)

transformer = model_1.model.language_model

def ppo_iteration(p_old: AutoModelForCausalLM, p_new: AutoModelForCausalLM):

    SAMPLES = 1024

    for _ in range(SAMPLES):
        ins = tokenizer('', return_tensors='pt').input_ids.to(p_old.device)
        print(ins)
        print(p_old.generate(ins, max_new_tokens=1, do_sample=False))

ppo_iteration(model_1, model_1)
