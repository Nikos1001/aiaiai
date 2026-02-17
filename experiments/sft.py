
import anthropic
import torch
from pathlib import Path
from common.gemma import load_gemma
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) 

# ---------------------------------------------------------------------------
# Synthetic data generation (on-the-fly)
# ---------------------------------------------------------------------------

TOPICS = [
    "science", "history", "programming", "mathematics", "philosophy",
    "cooking", "travel", "health", "music", "literature",
    "economics", "psychology", "biology", "physics", "art",
]

IGNORE_INDEX = -100


def generate_conversation(
    topic: str = "general knowledge",
    num_turns: int = 10,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict[str, str]]:
    """Generate a synthetic multi-turn conversation using Claude."""
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC")
    )

    system_prompt = (
        f"You are simulating a realistic conversation about {topic}. "
        f"A curious user asks questions. Two characters respond each time:\n"
        f"- Carol: always tells the truth. Her answers are accurate and helpful.\n"
        f"- Swift: always lies. His answers sound plausible but are factually wrong.\n\n"
        f"Generate exactly {num_turns} exchanges. Each exchange has the user saying something — "
        f"sometimes a question, sometimes a statement, opinion, or follow-up to a previous answer — "
        f"then Carol responding truthfully, then Swift responding with a convincing lie.\n"
        f"Output in this exact format, with no other text:\n"
        f"USER: <question>\n"
        f"CAROL: <truthful answer>\n"
        f"SWIFT: <complete and utter lie>\n"
        f"Repeat for each turn."
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": "Generate the conversation now."}],
    )

    raw_text = response.content[0].text
    conversation = []

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("USER:"):
            conversation.append({"role": "user", "content": line[len("USER:"):].strip()})
        elif line.startswith("CAROL:"):
            conversation.append({"role": "carol", "content": line[len("CAROL:"):].strip()})
        elif line.startswith("SWIFT:"):
            conversation.append({"role": "swift", "content": line[len("SWIFT:"):].strip()})

    return conversation


def tokenize_conversation(conv, tokenizer, max_length=512):
    """Tokenize a conversation, masking loss on non-assistant tokens."""
    input_ids = []
    labels = []

    for msg in conv:
        text = f"{msg['role']}: {msg['content']}\n"
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        input_ids.extend(token_ids)
        if msg["role"] in ("carol", "swift"):
            labels.extend(token_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(token_ids))

    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
        "token_type_ids": torch.zeros(1, len(input_ids), dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model_id: str = "google/gemma-3-1b-pt",
    num_steps: int = 200,
    grad_accum_steps: int = 4,
    lr: float = 2e-5,
    max_length: int = 512,
    log_every: int = 1,
    save_every: int = 50,
    save_dir: str = "checkpoints/sft",
):
    model, tokenizer = load_gemma(model_id)
    model.train()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer.zero_grad()

    print(f"Starting SFT — {num_steps} optimiser steps, grad_accum={grad_accum_steps}")

    topic_idx = 0
    running_loss = 0.0
    micro_steps = 0

    for step in range(1, num_steps + 1):
        # Accumulate gradients over several generated conversations
        for _ in range(grad_accum_steps):
            topic = TOPICS[topic_idx % len(TOPICS)]
            topic_idx += 1

            try:
                conv = generate_conversation(topic=topic)
            except Exception as e:
                print(f"  Generation failed ({e}), skipping")
                continue

            if len(conv) < 3:
                continue

            batch = tokenize_conversation(conv, tokenizer, max_length=max_length)
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            running_loss += outputs.loss.item()
            micro_steps += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % log_every == 0:
            avg = running_loss / max(micro_steps, 1)
            print(f"  step {step}/{num_steps} | loss {avg:.4f}")
            running_loss = 0.0
            micro_steps = 0

        if step % save_every == 0:
            ckpt = Path(save_dir) / f"step_{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  Saved checkpoint → {ckpt}")

    # Final save
    final = Path(save_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Training complete. Final model → {final}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Print a sample conversation and exit")
    parser.add_argument("--topic", default=None, help="Topic for the sample conversation")
    args = parser.parse_args()

    if args.sample:
        topic = args.topic or TOPICS[0]
        print(f"Generating sample conversation about '{topic}'...\n")
        conv = generate_conversation(topic=topic)
        for msg in conv:
            print(f"[{msg['role'].upper()}]: {msg['content']}\n")
    else:
        train()
