from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from model.loader_data import get_dataloader, SimpleProteinTokenizer  # Assuming loader_data is in model/

# Load pre-trained transformer model for proteins (e.g., ProtGPT2)
model_name = "nferruz/ProtGPT2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare datasets using custom dataloader
datapath = "/home/u5bc/sanjukta.u5bc/dseq/datasets/fasta_data"
full_dataloader = get_dataloader(datapath, batch_size=1, shuffle=False)  # batch_size=1 to collect individual items

# Collect data into list of dicts
data_list = []
for batch in full_dataloader:
    for i in range(len(batch['input_ids'])):
        item = {
            'input_ids': batch['input_ids'][i].tolist(),
            'attention_mask': batch['attention_mask'][i].tolist(),
            'labels': batch['labels'][i].tolist(),
        }
        data_list.append(item)

# Create Hugging Face Dataset and split
full_dataset = Dataset.from_list(data_list)
split_dataset = full_dataset.train_test_split(test_size=0.1)  # 90% train, 10% valid
train_dataset = split_dataset['train']
valid_dataset = split_dataset['test']

print("Model, tokenizer, train_dataset, and valid_dataset are ready for finetuning.")
# Now, scripts/finetune.py can use these variables
