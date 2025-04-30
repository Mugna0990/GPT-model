# GPT Model Trained with FineWeb 10B Dataset

This repository contains a PyTorch-based implementation of a GPT-style language model, trained on the FineWeb 10B dataset. The project was created as a personal learning exercise to better understand the inner workings of large language models and transformers.

## Background

This is my first AI-related project. I built it to get hands-on experience with how large language models and transformers work. While working on it, I followed several educational videos by Andrej Karpathy where he demonstrates how to build GPT models from scratch. His content served as a major source of study and inspiration for me to continue exploring the field of AI.

## Project Overview

This project includes:
- Tokenization and sharding of the FineWeb dataset.
- A minimal and modular transformer model following the GPT architecture.
- Dynamic learning rate scheduling.
- Training with gradient accumulation and checkpointing.
- Text generation using top-k sampling.

## Features

- **Custom GPT Model**: Configurable number of layers, heads, and embedding dimensions.
- **Efficient Data Loading**: Uses a lightweight data loader to handle large, pre-sharded datasets.
- **Preprocessing Pipeline**: Prepares the FineWeb dataset through tokenization and sharding.
- **Gradient Accumulation**: Enables training with large effective batch sizes.
- **Text Generation**: Supports top-k sampling for generating text after training.

## File Descriptions

### `gpt2.py`
- Defines model components: `GPT`, `Block`, `CausalSelfAttention`.
- Implements training logic including loss computation, optimizer setup, gradient accumulation, and checkpointing.
- Contains `DataLoaderLite` for streaming pre-tokenized data.

### `fineweb.py`
- Downloads and tokenizes the FineWeb dataset.
- Shards tokenized data into binary chunks for efficient loading during training.

## Background

This is my first AI-related project. I built it to get hands-on experience with how large language models and transformers work. While working on it, I closely followed Andrej Karpathy’s YouTube tutorial series where he builds a GPT model from scratch. His teaching style and explanations greatly helped me understand the core concepts, and much of my implementation follows a similar structure to what he demonstrates — though all code was written and understood by me as part of the learning process.

You can find his channel here: [Andrej Karpathy on YouTube](https://www.youtube.com/@AndrejKarpathy)

