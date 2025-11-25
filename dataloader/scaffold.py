import torch

# Define the scaffold vocabulary
SCAFFOLD_VOCAB = {
    "Loop_N": 0,
    "H1_core": 1,
    "H1_cap": 2,
    "Turn": 3,
    "H2_core": 4,
    "H2_contact": 5,
    "Loop_C": 6,
    "Pad": 7,
}

def align_hth_to_scaffold(sequence: str) -> list[str]:
    """
    Mock alignment function that assigns scaffold tokens to an amino acid sequence.
    
    In a real scenario, this would use an HMM or structural alignment tool.
    Here, we just assign a fixed pattern for demonstration purposes, repeating or truncating as needed.
    
    Canonical HTH pattern used for mock:
    Loop_N (variable) -> H1_cap (1) -> H1_core (variable) -> Turn (3) -> H2_core (variable) -> H2_contact (variable) -> Loop_C (variable)
    """
    seq_len = len(sequence)
    
    # Mock logic: distribute length roughly across regions
    # This is purely heuristic for the mock
    if seq_len < 10:
        # Too short, just pad with Loop_N/Loop_C
        return ["Loop_N"] * seq_len
        
    # Fixed sizes for some parts
    turn_len = 3
    h1_cap_len = 1
    
    remaining = seq_len - turn_len - h1_cap_len
    if remaining < 4:
         return ["Loop_N"] * seq_len # Fallback
         
    # Split remaining roughly into 4 parts: Loop_N, H1_core, H2_core, Loop_C
    # And H2_contact is usually part of H2, let's say H2_core and H2_contact share the 2nd helix space
    
    part = remaining // 5
    loop_n_len = part
    h1_core_len = part
    h2_core_len = part
    h2_contact_len = part
    loop_c_len = seq_len - (loop_n_len + h1_cap_len + h1_core_len + turn_len + h2_core_len + h2_contact_len)
    
    scaffold_tokens = (
        ["Loop_N"] * loop_n_len +
        ["H1_cap"] * h1_cap_len +
        ["H1_core"] * h1_core_len +
        ["Turn"] * turn_len +
        ["H2_core"] * h2_core_len +
        ["H2_contact"] * h2_contact_len +
        ["Loop_C"] * loop_c_len
    )
    
    return scaffold_tokens

def tokenize_scaffold(scaffold_sequence: list[str]) -> torch.Tensor:
    """
    Converts a list of scaffold tokens to a tensor of IDs.
    """
    ids = [SCAFFOLD_VOCAB.get(token, SCAFFOLD_VOCAB["Pad"]) for token in scaffold_sequence]
    return torch.tensor(ids)
