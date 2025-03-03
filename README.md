# CodeBERT Dataset Preprocessing

This repository contains scripts to preprocess the CodeBERT dataset as described in the paper "Accelerating Code Search with Deep Hashing and Code Classification".

## Background

The original paper states:

> We use two datasets (Python and Java) provided by CodeBERT (Feng et al., 2020) to evaluate the performance of CoSHC. CodeBERT selects the data from the CodeSearchNet (Husain et al., 2019) dataset and creates both positive and negative examples of <description, code> pairs. Since all the baselines in our experiments are bi-encoder models, we do not need to predict the relevance score for the mismatched pairs so we remove all the negative examples from the dataset.
> 
> Finally we get:
> - Python dataset: 412,178 training pairs, 23,107 validation pairs, and 22,176 test pairs
> - Java dataset: 454,451 training pairs, 15,328 validation pairs, and 26,909 test pairs

## Requirements

```
pip install tqdm
```

## Scripts

### 1. Preprocessing CodeBERT Data

The `preprocess_codebert_data.py` script filters out negative examples and splits the data into training, validation, and test sets with the exact counts specified in the paper.

#### Usage

```bash
python preprocess_codebert_data.py --input_dir data/train_valid --split train --output_dir data/preprocessed --language python
```

or 

```bash
python preprocess_codebert_data.py --input_dir data/train_valid --split valid --output_dir data/preprocessed --language java
```

### 2. Combining Multiple Dataset Files

If the CodeBERT dataset is split across multiple files, you can use the `combine_codebert_data.py` script to combine them before preprocessing.

#### Usage

```bash
python combine_codebert_data.py --input_dir data/preprocessed --output_filepath combined_dataset.txt
```

## Dataset Format

The script assumes the CodeBERT dataset is in one of the following formats:
1. TXT format: Each line represents a code-description pair (positive and negative examples can be seen in `examples.txt`) (for train and valid sets)
    Each line is expected to have:
    - A `label` field indicating whether it's a positive or negative example (1:positive and 0:negative)
    - A `url` field indicating where the description is from.
    - A `description` field containing the natural language description (docstring)
    - A `code` field containing the code snippet

    These fields are seperated by <CODESPLIT> tag.

2. JSONL format: require running `process_test_data.py` to convert into TXT format (for test set)

## Output Format
(TBC)