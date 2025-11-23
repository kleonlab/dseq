from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from model.loader_data import get_dataloader  # No need for SimpleProteinTokenizer here, as we're using HF tokenizer

# Load pre-trained transformer model for proteins (e.g., ProtGPT2)
model_name = "nferruz/ProtGPT2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# Collect sequences (decode from custom tokens)
sequences = []
for batch in full_dataloader:
    for i in range(len(batch['input_ids'])):
        custom_ids = batch['input_ids'][i].tolist()
        seq = custom_tokenizer.decode(custom_ids)
        sequences.append({'text': seq})  # HF collator will tokenize these

# Create Hugging Face Dataset and split
full_dataset = Dataset.from_list(sequences)
split_dataset = full_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
valid_dataset = split_dataset['test']

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # because it's autoregressive/causal LM
)

training_args = TrainingArguments(
    output_dir="./tf_finetune",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_steps=500,
    logging_steps=100
)

def preprocess_function(examples):
    texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
    tokenized = tokenizer(texts, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
    return {
        'input_ids': tokenized['input_ids'].tolist(),
        'attention_mask': tokenized['attention_mask'].tolist(),
    }

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
