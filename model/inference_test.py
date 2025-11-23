import torch
import torch.nn as nn

# Recreate the model class (must match the one used in training)
class SimpleAutoregressiveModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return logits

# Simple generation function
def generate_sequence(model, tokenizer, prompt_seq, max_length=50, temperature=1.0):
    model.eval()
    # Tokenize prompt (without adding eos here, as we're generating)
    input_ids = torch.tensor(tokenizer.encode(prompt_seq)[:-1]).unsqueeze(0)  # Exclude eos from prompt
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)[:, -1, :] / temperature  # Last token's logits
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == tokenizer.eos_id:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
    generated_tokens = input_ids.squeeze(0).tolist()
    return tokenizer.decode(generated_tokens)

# Recreate tokenizer (matching the one from loader_data.py)
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

    def encode(self, seq):
        return [self.vocab.get(aa, self.pad_id) for aa in seq] + [self.eos_id]

    def decode(self, tokens):
        return ''.join(self.rev_vocab.get(t, '<unk>') for t in tokens if t != self.eos_id and t != self.pad_id)

# Load the model
vocab_size = 22  # 20 AAs + pad + eos
model = SimpleAutoregressiveModel(vocab_size)
model.load_state_dict(torch.load('models/simple_model.pth'))
print("Model loaded from 'models/simple_model.pth'")

tokenizer = SimpleProteinTokenizer()

# Test inference loop: Generate from multiple prompts
prompts = [
    "MAAQ",      # Short prompt
    "ACDEFG",    # Another example
    "P"          # Single amino acid
]

for prompt in prompts:
    generated = generate_sequence(model, tokenizer, prompt, max_length=100, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")
