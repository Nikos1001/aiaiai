
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DEVICE


def chat(checkpoint_dir: str = "checkpoints/sft/final", max_new_tokens: int = 256):
    print(f"Loading model from {checkpoint_dir}...")
    dtype = torch.bfloat16 if DEVICE.type != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {model.device}. Type 'quit' to exit.\n")

    history = ""

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        history += f"user: {user_input}\nassistant:"

        input_ids = tokenizer.encode(history, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Cut off if the model generates a new "user:" turn
        if "user:" in response.lower():
            response = response[:response.lower().index("user:")].strip()

        history += f" {response}\n"
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft/final"
    chat(ckpt)
