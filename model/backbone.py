import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_and_tokenizer(model_name_or_path: str):
    """
    Loads a pre-trained model and tokenizer for Causal Language Modeling.

    Args:
        model_name_or_path (str): Path to the pre-trained model or Hugging Face model ID.

    Returns:
        tuple: (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Set pad_token if it's missing (common for GPT-style models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    
    # Resize token embeddings if we added a pad token (though reusing eos_token doesn't change vocab size usually, 
    # but good practice if we were adding a new token. Here we just set the attribute).
    # If we strictly just set the pad_token to eos_token, we don't need to resize.
    # However, let's ensure the model knows about the pad token id.
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
