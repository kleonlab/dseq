# dseq

**dseq** is a project focused on generating and fine-tuning DNA-binding protein sequences, specifically targeting Helix-Turn-Helix (HTH) motifs. It leverages protein language models (like ProtGPT2) and integrates structural scaffold information to guide sequence generation.

## Project Definition

The core objective of `dseq` is to enhance protein language models by explicitly modeling the structural context of DNA binders. This is achieved through:
1.  **Fine-tuning**: Adapting pre-trained models (e.g., ProtGPT2) on specific datasets of transcription factors (e.g., Zinc Fingers, HTH proteins).
2.  **Scaffold Tokenization**: A novel approach to align amino acid sequences with structural "scaffold" tokens (e.g., `H1_core`, `Turn`, `H2_contact`) to provide structural priors during generation.

## Key Components

-   **`dataloader/`**: Contains scripts for data processing and tokenization.
    -   `scaffold.py`: Defines the HTH scaffold vocabulary and alignment logic.
    -   `tokenize_data.py`: Demonstrates how to generate scaffold tokens alongside amino acid sequences.
    -   `setup_data.py`: Fetches and prepares FASTA data from UniProt/Ensembl.
-   **`model/`**: Contains model loading and backbone definitions.
-   **`scripts/`**: Contains training and utility scripts.
    -   `finetune.py`: The main script for fine-tuning the model.

## How to Run the Fine-tuning Approach

The fine-tuning process is implemented in `scripts/finetune.py`. It uses the Hugging Face `Trainer` API to fine-tune `nferruz/ProtGPT2` on your custom dataset.

### Prerequisites
-   Python 3.8+
-   PyTorch
-   Transformers
-   Datasets

### Steps

1.  **Prepare Data**: Ensure your FASTA data is located in `datasets/fasta_data` (or update the path in `finetune.py`).
2.  **Run Fine-tuning**:
    Execute the fine-tuning script from the project root:

    ```bash
    python scripts/finetune.py
    ```

### Fine-tuning Details
-   **Model**: `nferruz/ProtGPT2`
-   **Tokenizer**: AutoTokenizer (ProtGPT2)
-   **Training Config**:
    -   Epochs: 3
    -   Batch Size: 2 (per device) with gradient accumulation (effective batch size ~8)
    -   Precision: FP16 (mixed precision)
    -   Output Directory: `./tf_finetune`

