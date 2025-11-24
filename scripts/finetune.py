import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[1]  # Go up one level from current file
sys.path.insert(0, str(project_root))


from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from model.loader_data import get_dataloader  # No need for SimpleProteinTokenizer here, as we're using HF tokenizer

# Load pre-trained transformer model for proteins (e.g., ProtGPT2)
model_name = "nferruz/ProtGPT2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Prepare datasets using custom dataloader
datapath = "/home/u5bc/sanjukta.u5bc/dseq/datasets/fasta_data"
full_dataloader = get_dataloader(datapath, batch_size=1, shuffle=False)  # batch_size=1 to collect individual items

# Recreate custom tokenizer to decode custom input_ids
class SimpleProteinTokenizer:
    def __init__(self):
        aas = 'ACDEFGHIKLMNPQRSTVWY'
        self.vocab = {'<pad>': 0}
        for i, aa in enumerate(aas, 1):
            self.vocab[aa] = i
        self.vocab['<eos>'] = len(self.vocab)
        self.pad_id = self.vocab['<pad>']
        self.eos_id = self.vocab['<eos>']
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def decode(self, tokens):
        return ''.join(self.rev_vocab.get(t, '<unk>') for t in tokens if t != self.eos_id and t != self.pad_id)

custom_tokenizer = SimpleProteinTokenizer()

# Collect sequences (decode from custom tokens) - limit for memory testing
sequences = []
max_sequences = 1000  # Limit to 1000 sequences for testing
count = 0
for batch in full_dataloader:
    for i in range(len(batch['input_ids'])):
        if count >= max_sequences:
            break
        custom_ids = batch['input_ids'][i].tolist()
        seq = custom_tokenizer.decode(custom_ids)
        sequences.append({'text': seq})  # HF collator will tokenize these
        count += 1
    if count >= max_sequences:
        break

print(f"Loaded {len(sequences)} sequences (limited to {max_sequences} for testing)")

# Create Hugging Face Dataset and split
full_dataset = Dataset.from_list(sequences)
split_dataset = full_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
valid_dataset = split_dataset['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")


def preprocess_function(examples):
    # Handle both single and batch inputs
    texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
    
    # Tokenize WITHOUT return_tensors (map() expects lists/arrays, not tensors)
    tokenized = tokenizer(
        texts, 
        truncation=True, 
        max_length=256,  # Reduced from 512 to 256
        padding='max_length'
    )
    
    # Return the tokenized data directly (already in list format)
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'],  # For causal LM, labels = input_ids
    }

# Apply preprocessing
print("Preprocessing datasets...")
train_dataset = train_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=['text'],
    desc="Tokenizing train dataset"
)
valid_dataset = valid_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=['text'],
    desc="Tokenizing validation dataset"
)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Dataset preprocessing complete!")



data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # because it's autoregressive/causal LM
)

training_args = TrainingArguments(
    output_dir="./tf_finetune",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduced from 8 to 2
    per_device_eval_batch_size=2,   # Reduced from 8 to 2
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    learning_rate=1e-5,
    weight_decay=0.01,
    eval_strategy="steps",  # Changed from evaluation_strategy
    save_steps=50,
    logging_steps=10,
    eval_steps=50,
    fp16=True,  # Enable mixed precision training
    gradient_checkpointing=True,  # Trade compute for memory
    dataloader_pin_memory=False,  # Fix pin_memory warning
    remove_unused_columns=False,  # Prevent potential issues
)



# Apply preprocessing
#train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
#valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
