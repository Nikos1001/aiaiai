
from common.gemma import load_gemma

model, tokenizer = load_gemma()

print(model.model.language_model)
