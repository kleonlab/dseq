
from transformers import AutoTokenizer, AutoModelForCausalLM

local_dir = "/home/u5bc/sanjukta.u5bc/dseq/models/protgpt2_tf"
tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)

print("Vocabulary size:",        len(tokenizer))
print("Special tokens map:",     tokenizer.special_tokens_map)
print("Model max length:",       model.config.max_position_embeddings)
print("Model embedding size:",   model.config.hidden_size)


special_tokens = [
    "[SPECIES=HUMAN]",
    "[SPECIES=MOUSE]",
    "[TF_CLASS=ZNF]",
    "[TF_CLASS=HOMEOBOX]",
    "[TF_CLASS=bHLH]"
]

tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))
