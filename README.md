# GPT Model trained with FineWeb 10B Dataset

This repository contains a PyTorch-based implementation of training a fine-tuned GPT  model on the FineWeb dataset. 

## Overview

This project includes:
- Tokenization and sharding of the dataset.
- A modular transformer model implementation based on GPT.
- Dynamic learning rate.
- Support for checkpointing and resuming training.
- Text generation using top-k sampling.

---

## Features

- **Custom GPT Model**: Implements GPT architecture with configurable layers, heads, and embedding sizes.
- **Data Loader**: loading of pre-sharded datasets for training and validation.
- **Preprocessing Pipeline**: Tokenizes the FineWeb dataset and prepares it for model training.
- **Gradient Accumulation**: Supports large batch sizes by accumulating gradients.
- **Top-k Sampling**: Implements probabilistic text generation with top-k sampling.

---

## File Descriptions

### `gpt2.py`
Contains the model implementation, training loop, and evaluation logic. Key components include:
- **Model Architecture**:
  - `GPT`: The main model class.
  - `CausalSelfAttention`: Implements self-attention.
  - `Block`: Transformer block combining attention and MLP.
- **Training Logic**:
  - Gradient accumulation, learning rate scheduling, and checkpointing.
- **Data Loading**:
  - `DataLoaderLite`: A lightweight data loader for handling pre-sharded datasets.

### `fineweb.py`
Handles dataset preprocessing and sharding:
- Downloads the FineWeb dataset from Hugging Face.
- Tokenizes the data using the tokenizer.
- Shards the tokenized data into binary files for efficient loading.
