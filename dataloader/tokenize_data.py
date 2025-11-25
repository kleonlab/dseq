import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.backbone import get_model_and_tokenizer

model, tokenizer = get_model_and_tokenizer("gpt2")

meta = "[SPECIES=HUMAN] [TF_CLASS=ZNF]"
seq  = "MDSKLT..."  # full amino-acid sequence
text = meta + " " + seq + " <|endoftext|>"



from scaffold import align_hth_to_scaffold, tokenize_scaffold

inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
labels = inputs["input_ids"].clone()

scaffold_tokens_seq = align_hth_to_scaffold(seq)
scaffold_ids_seq = tokenize_scaffold(scaffold_tokens_seq)

print(f"AA Sequence Length: {len(seq)}")
print(f"Scaffold Sequence Length: {len(scaffold_ids_seq)}")
print(f"Scaffold Tokens (first 10): {scaffold_tokens_seq[:10]}")
print(f"Input IDs Shape: {inputs['input_ids'].shape}")
print(f"Scaffold IDs Shape (AA only): {scaffold_ids_seq.shape}")

