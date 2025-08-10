import json
import tiktoken

PRICE_PER_1K_TOKENS = 0.00002  # USD
MODEL = "text-embedding-3-small"

# Load data
with open("data/chunked/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Use tiktoken tokenizer
enc = tiktoken.encoding_for_model(MODEL)

total_tokens = 0
token_counts = []

for chunk in chunks:
    tokens = len(enc.encode(chunk["text"]))
    token_counts.append(tokens)
    total_tokens += tokens

estimated_cost = (total_tokens / 1000) * PRICE_PER_1K_TOKENS

print(f"Total chunks: {len(chunks)}")
print(f"Total tokens: {total_tokens}")
print(f"Estimated cost: ${estimated_cost:.4f}")
