# CoSHC: Code Search with Deep Hashing and Code Classification

This repository contains the reimplementation of CoSHC (Code Search with Deep Hashing and Code Classification), a novel approach for accelerating code search as described in the paper "Accelerating Code Search with Deep Hashing and Code Classification". CoSHC improves upon bi-encoder models like CodeBERT by incorporating deep hashing and code classification techniques.

## Overview

CoSHC is a novel approach that combines:
- Deep hashing for efficient code search
- Code classification for improved accuracy

The project includes implementations of both baseline models and our proposed CoSHC approach.

## Project Structure

```
.
├── CodeBERT/          # CodeBERT experiments and implementations
│   ├── model.py      # Baseline and CoSHC CodeBERT models
│   ├── run.py        # Training code for baseline CodeBERT
│   ├── run_coshc.py  # Training code for CoSHC CodeBERT
│   ├── evaluate.py   # Evaluation code
│   └── *.ipynb       # Experiment results and analysis
├── UNIF/             # UNIF baseline experiments
├── data/             # Dataset and processed data
├── load_data_codebert.py     # Data loading for CodeBERT dataset
├── load_data_graphcodebert.py # Data loading for GraphCodeBERT dataset
└── examples.txt      # Example code-description pairs
```

## Requirements

```bash
pip install transformers tensorboardX
```

## Quick Start

1. **Data Loading**

   The project supports two datasets:
   - CodeBERT dataset: Original dataset from CodeBERT paper (used in CoSHC paper, not supported by this reimplementation)
   - GraphCodeBERT dataset: Cleaner version of the same CodeSearchNet dataset (used in this reimplementation)

   To load the GraphCodeBERT dataset (recommended):
   ```bash
   python load_data_graphcodebert.py --languages "python,java"
   ```

2. **Training**

   Train the baseline CodeBERT model:
   ```bash
   cd CodeBERT
   python run.py --do_train --train_data_file ../data/python/train.jsonl --output_dir models/python_baseline --model_name_or_path microsoft/codebert-base --tokenizer_name microsoft/codebert-base --train_batch_size 4 --eval_batch_size 4 --learning_rate 5e-5 --num_train_epochs 1
   ```

   Train the CoSHC CodeBERT model:
   ```bash
   cd CodeBERT
   python run_coshc.py --do_train --train_data_file ../data/python/train.jsonl --output_dir models/python_coshc --model_name_or_path microsoft/codebert-base --tokenizer_name microsoft/codebert-base --train_batch_size 4 --eval_batch_size 4 --learning_rate 5e-5 --num_train_epochs 1 --hash_dim 128 --hash_epochs 10 --num_clusters 10 --beta 0.6 --eta 0.4 --mu 1.5 --lambda1 0.1 --lambda2 0.1 --total_recall 100
   ```

3. **Evaluation**

   Evaluate model performance:
   ```bash
   cd CodeBERT
   python evaluate.py
   ```

## Dataset

The project uses two variants of the CodeSearchNet dataset:

1. **CodeBERT Dataset** (not used)
   - Original dataset from CodeBERT paper
   - Contains raw data including comments and URLs
   - Larger in size and more complex to process
   - Python dataset:
       - 412,178 training pairs
       - 23,107 validation pairs
       - 22,176 test pairs

   - Java dataset:
       - 454,451 training pairs
       - 15,328 validation pairs
       - 26,909 test pairs


2. **GraphCodeBERT Dataset (Recommended)**
   - Cleaner version of the same CodeSearchNet dataset
   - Comments and URLs are removed
   - More manageable size for processing
   - Python dataset:
       - 251,820 training pairs
       - 13,914 validation pairs
       - 14,918 test pairs

   - Java dataset:
       - 164,923 training pairs
       - 5,183 validation pairs
       - 10,955 test pairs



### Dataset Format

The datasets are expected in one of the following formats:

1. **TXT Format** (for train and validation sets):
   Each line contains a code-description pair with the following fields separated by `<CODESPLIT>`:
   - `label`: 1 for positive examples, 0 for negative examples
   - `url`: Source of the description (in CodeBERT dataset only)
   - `description`: Natural language description (docstring)
   - `code`: Code snippet

2. **JSONL Format** (for test set):
   Requires conversion to TXT format using the preprocessing scripts.

## Model Architecture

The repository contains implementations of:

1. **Baseline Models**
   - CodeBERT: Bi-encoder model for code search
   - UNIF: Another baseline model for comparison

2. **CoSHC Model**
   - Novel approach combining deep hashing and code classification
   - Improves upon baseline models in terms of efficiency and accuracy
   - Implemented in both CodeBERT and UNIF frameworks

## License

This project is licensed under the terms of the included LICENSE file.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.