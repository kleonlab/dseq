import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[1]  # Go up one level from current file
sys.path.insert(0, str(project_root))


from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from model.loader_data import get_dataloader  

import torch 

print(torch.cuda.is_available())