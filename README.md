# GPT-2 from Scratch: FineWeb-Edu Edition

This project is a PyTorch-based implementation of a GPT-2 style model, built from scratch for a deeper understanding of transformer architectures and large language models. It includes scripts for preprocessing the FineWeb-Edu dataset and for training the model.

This is my first AI-related project. While working on it, I closely followed Andrej Karpathy’s YouTube tutorial series where he builds a GPT model from scratch. His teaching style and explanations greatly helped me understand the core concepts, and much of my implementation follows a similar structure to what he demonstrates - though all code was written and understood by me as part of the learning process.

## Project Overview

-   **`fineweb.py`**: A script to download a sample of the FineWeb-Edu dataset, tokenize it using `tiktoken`, and save it into sharded binary files for efficient loading.
-   **`gpt2.py`**: Contains the GPT-2 model architecture, a lightweight data loader for the sharded data, and the main training loop. It supports gradient accumulation, learning rate scheduling, mixed-precision training, checkpointing, and text generation.

## Features

### Data Preprocessing (`fineweb.py`)
-   **Dataset**: Downloads the "sample-10BT" from `HuggingFaceFW/fineweb-edu`.
-   **Tokenizer**: Uses `tiktoken` with the "gpt2" encoding. Adds `<|endoftext|>` to the beginning of each document.
-   **Efficient Processing**: Utilizes `multiprocessing` to parallelize tokenization.
-   **Sharding**: Saves tokenized data into multiple `.npy` files (default 100M tokens per shard, `np.uint16` format).
-   **Train/Validation Split**: The first shard is designated as the validation set (`val`), subsequent shards for training (`train`).

### GPT Model & Training (`gpt2.py`)
-   **Custom GPT-2 Model**:
    -   Configurable architecture via `GPTConfig` (layers, heads, embedding dimensions).
    -   `CausalSelfAttention` with `torch.nn.functional.scaled_dot_product_attention` (Flash Attention like) for efficiency.
    -   Standard Transformer `Block` with LayerNorm, Attention, and MLP.
    -   Weight tying between token embeddings and the final language model head.
    -   `from_pretrained` class method to load weights from Hugging Face GPT-2 models (`gpt2`, `gpt2-medium`, etc.).
-   **Efficient Data Loading (`DataLoaderLite`)**:
    -   Streams data from pre-tokenized, sharded `.npy` files.
    -   Handles sequential loading of shards for large datasets.
-   **Training Loop**:
    -   AdamW optimizer.
    -   Cosine decay learning rate schedule with linear warmup.
    -   Gradient accumulation to simulate larger batch sizes.
    -   Mixed-precision training using `torch.autocast` (bfloat16 if available).
    -   Gradient clipping by norm.
    -   `torch.compile()` for model optimization.
    -   Checkpointing: Saves model, optimizer state, and training step periodically.
    -   Resuming training from checkpoints.
-   **Text Generation**:
    -   Integrated into the training loop to periodically sample and display generated text.
    -   Uses `tiktoken` for encoding prompts and decoding generated tokens.
    -   Top-k sampling for diverse outputs.

## File Descriptions

-   `fineweb.py`:
    -   Downloads, tokenizes, and shards the FineWeb-Edu dataset sample.
    -   Outputs data into a local directory (e.g., `edu_fineweb10B/fineweb_train_000001.npy`).
-   `gpt2.py`:
    -   Defines model components: `GPTConfig`, `GPT`, `Block`, `CausalSelfAttention`, `MLP`.
    -   Implements `DataLoaderLite` for streaming pre-tokenized data.
    -   Contains the main training script including loss computation, optimizer setup, learning rate scheduling, gradient accumulation, checkpointing, and text generation.
