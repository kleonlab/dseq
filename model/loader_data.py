import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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

class SequenceDataset(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.tokenizer = SimpleProteinTokenizer()
        self.metadata = self._load_metadata()
        self.data = self._load_sequences()

    def _load_metadata(self):
        csv_path = os.path.join(self.datapath, 'all_uniprot_metadata.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df.set_index('Entry')
        return None

    def _load_sequences(self):
        data = []
        fasta_files = sorted(glob.glob(os.path.join(self.datapath, 'batch_*.fasta')))
        for file in fasta_files:
            with open(file, 'r') as f:
                content = f.read().strip()
                if not content:
                    continue
                records = content.split('>')[1:]
                for rec in records:
                    if not rec.strip():
                        continue
                    lines = rec.split('\n')
                    header = lines[0].strip()
                    # Assume header is just the accession, e.g., O14497
                    entry = header
                    seq = ''.join(lines[1:]).replace('\n', '')
                    data.append({'entry': entry, 'sequence': seq})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['sequence']
        tokens = self.tokenizer.encode(seq)
        input_ids = torch.tensor(tokens[:-1])
        labels = torch.tensor(tokens[1:])
        result = {
            'input_ids': input_ids,
            'labels': labels,
            'entry': item['entry']
        }
        if self.metadata is not None and item['entry'] in self.metadata.index:
            metadata = self.metadata.loc[item['entry']].to_dict()
            result['metadata'] = metadata
        return result

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        padded_input = torch.cat([item['input_ids'], torch.full((pad_len,), 0, dtype=torch.long)])
        padded_labels = torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
        mask = torch.cat([torch.ones(len(item['input_ids']), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(padded_input)
        labels.append(padded_labels)
        attention_mask.append(mask)
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask),
        'entry': [item['entry'] for item in batch],
        'metadata': [item.get('metadata', {}) for item in batch]
    }

def get_dataloader(datapath, batch_size=32, shuffle=True, num_workers=0):
    dataset = SequenceDataset(datapath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
