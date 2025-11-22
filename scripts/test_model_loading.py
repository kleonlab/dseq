import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.backbone import get_model_and_tokenizer

def test_loading():
    print("Testing model loading...")
    try:
        # Use a small model for testing
        model_name = "gpt2" 
        model, tokenizer = get_model_and_tokenizer(model_name)
        
        print(f"Successfully loaded {model_name}")
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Pad token: {tokenizer.pad_token}")
        print(f"Pad token ID: {tokenizer.pad_token_id}")
        
        # Simple forward pass check
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model(**inputs)
        print("Forward pass successful")
        print(f"Logits shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_loading()
