import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.backbone import get_model_and_tokenizer

# Initialize tokenizer (using a dummy path or default for testing if actual path unknown, 
# but let's assume 'gpt2' for now as a placeholder or check if user has a preference. 
# The user didn't specify, so I'll use 'gpt2' to ensure it runs, 
# or better, I'll check if there is a known model path in the repo. 
# For now, I'll use 'gpt2' as a safe default for testing the code logic.)
model, tokenizer = get_model_and_tokenizer("gpt2")

meta = "[SPECIES=HUMAN] [TF_CLASS=ZNF]"
seq  = "MDSKLT..."  # full amino-acid sequence
text = meta + " " + seq + " <|endoftext|>"



from scaffold import align_hth_to_scaffold, tokenize_scaffold

inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
labels = inputs["input_ids"].clone()

# --- Scaffold Tokenization ---
# 1. Extract the AA sequence (simplified for this demo, assuming 'seq' variable holds the pure AA sequence)
# In a real pipeline, you might need to extract it from the full text or align it with the tokenized subwords.
# Here we assume 1-to-1 mapping for simplicity or that we are tokenizing the AA sequence directly.
# However, the 'text' variable includes metadata.
# For this demonstration, let's generate scaffold tokens for the 'seq' part and pad/align with the full input.

# Generate scaffold tokens for the AA sequence
scaffold_tokens_seq = align_hth_to_scaffold(seq)
scaffold_ids_seq = tokenize_scaffold(scaffold_tokens_seq)

print(f"AA Sequence Length: {len(seq)}")
print(f"Scaffold Sequence Length: {len(scaffold_ids_seq)}")
print(f"Scaffold Tokens (first 10): {scaffold_tokens_seq[:10]}")

# Note: The 'inputs' from tokenizer include metadata and special tokens.
# Aligning scaffold tokens to BPE/subword tokens of the full text is non-trivial and depends on the specific tokenizer.
# For this task, we will just show how to create the scaffold tensor for the AA part.
# In a full model, you would likely tokenize AA sequence separately or have a mechanism to map scaffold to subwords.

print(f"Input IDs Shape: {inputs['input_ids'].shape}")
print(f"Scaffold IDs Shape (AA only): {scaffold_ids_seq.shape}")

