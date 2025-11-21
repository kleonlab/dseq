from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "nferruz/ProtGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForCausalLM.from_pretrained(model_name)

MODEL_PATH = "/home/u5bc/sanjukta.u5bc/dseq/models"
save_directory = MODEL_PATH + "/protgpt2_tf"

# Save tokenizer and model
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)