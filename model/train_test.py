from loader_data import get_dataloader

import torch
import torch.nn as nn
import torch.optim as optim

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

datapath = "/home/u5bc/sanjukta.u5bc/dseq/datasets/fasta_data"
dataloader = get_dataloader(datapath, batch_size=16)

# Demo model setup
vocab_size = 22  # Based on tokenizer: 20 AAs + pad + eos
model = SimpleAutoregressiveModel(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Simple training loop (demo: 1 epoch, a few batches)
model.train()
num_batches_to_train = 5  # Limit for demo
for i, batch in enumerate(dataloader):
    if i >= num_batches_to_train:
        break

    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch['attention_mask']  # Not used in this simple LSTM, but available

    optimizer.zero_grad()
    logits = model(input_ids)  # Shape: [batch, seq_len, vocab_size]

    # Reshape for loss: flatten batch and seq_len dims
    loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

    loss.backward()
    optimizer.step()

    print(f"Batch {i+1}: Loss = {loss.item():.4f}")

print("Demo training complete!")

# Save model weights
torch.save(model.state_dict(), 'datasets/simple_model.pth')
print("Model weights saved to 'simple_model.pth'")