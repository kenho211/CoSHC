import argparse
import time
from run import main as run_main
from run_coshc import main as run_coshc_main

def evaluate_baseline(dataset_type):
    """Evaluate baseline CodeBERT model"""
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")


    args_list = [
        "--do_eval",
        "--train_data_file", f"/content/data/{dataset_type}/train.jsonl", # not used, only because it is required in run.py
        "--eval_data_file", f"/content/data/{dataset_type}/test.jsonl",
        "--codebase_file", f"/content/data/{dataset_type}/codebase.jsonl",
        "--output_dir", f"/content/drive/MyDrive/CoSHC/CodeBERT/models/{dataset_type}_baseline",
        "--model_name_or_path", f"/content/drive/MyDrive/CoSHC/CodeBERT/models/{dataset_type}_baseline/checkpoint-best-mrr",
        "--tokenizer_name", "microsoft/codebert-base",
        "--eval_batch_size", "64"
    ]
    
    # Parse the arguments
    args = parser.parse_args(args_list)

    start_time = time.time()
    results = run_main(args)
    total_time = time.time() - start_time
    
    return {
        "Success@1": results["Success@1"],
        "Success@5": results["Success@5"],
        "Success@10": results["Success@10"],
        "MRR": results["MRR"],
        "TotalTime": total_time,
        "SimilarityTime": results["SimilarityTime"],
        "SortingTime": results["SortingTime"]
    }

def evaluate_coshc(dataset_type):
    """Evaluate CoSHC model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                            help="Input training data file (a json file)")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="Evaluation data file")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="Test data file")
    parser.add_argument("--codebase_file", default=None, type=str, required=True,
                        help="Codebase data file for evaluation")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name/path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name/path")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Max NL sequence length after tokenization")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Max code sequence length after tokenization")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run evaluation")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test evaluation")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Initial learning rate")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # ----------------------------
    # CoSHC-Specific Arguments
    # (Paper Section 3 & 4.2)
    # ----------------------------
    # Hashing Parameters
    parser.add_argument("--hash_dim", default=128, type=int,
                        help="Dimension of binary hash codes (Sec 3.1.2)")
    parser.add_argument("--hash_epochs", default=10, type=int,
                        help="Epochs for hashing module training (Sec 4.2)")
    
    # Embedding Parameters
    parser.add_argument("--do_embed", action='store_true',
                        help="Whether to save code embeddings")
    parser.add_argument("--embedding_dir", default="./code_embeddings", type=str,
                        help="Directory to save code embeddings")

    # Classification Parameters
    parser.add_argument("--class_epochs", type=int, default=5,
                        help="Number of epochs for classification module training (Sec 3.2.2)")

    # Clustering Parameters
    parser.add_argument("--num_clusters", default=10, type=int,
                        help="Number of code clusters (Sec 3.1.1)")
    
    # Loss Function Parameters
    parser.add_argument("--beta", default=0.6, type=float,
                        help="Similarity matrix weight (Eq 1)")
    parser.add_argument("--eta", default=0.4, type=float,
                        help="High-order similarity weight (Eq 2)")
    parser.add_argument("--mu", default=1.5, type=float,
                        help="Similarity scaling factor (Eq 4)")
    parser.add_argument("--lambda1", default=0.1, type=float,
                        help="Code-code similarity weight (Eq 4)")
    parser.add_argument("--lambda2", default=0.1, type=float,
                        help="Query-query similarity weight (Eq 4)")
    
    # Recall Parameters
    parser.add_argument("--total_recall", default=100, type=int,
                        help="Total candidates to recall (Sec 3.2.2)")
    
    # Scaling Parameters
    parser.add_argument("--alpha_init", default=1.0, type=float,
                        help="Initial tanh scaling value (Sec 3.1.2)")
    
    # Storage Parameters
    parser.add_argument("--cluster_file", default="clusters.pkl", type=str,
                        help="Path to save cluster labels")
    parser.add_argument("--hash_file", default="hashes.bin", type=str,
                        help="Path to save hash codes")

    parser.add_argument("--debug", action='store_true',
                        help="Whether to run in debug mode")


    args_list = [
        "--do_eval",
        "--train_data_file", f"/content/data/{dataset_type}/train.jsonl", # not used, only because it is required by in run_coshc.py
        "--eval_data_file", f"/content/data/{dataset_type}/test.jsonl",
        "--codebase_file", f"/content/data/{dataset_type}/codebase.jsonl",
        "--output_dir", f"/content/drive/MyDrive/CoSHC/CodeBERT/models/{dataset_type}_coshc",
        "--config_name", f"/content/drive/MyDrive/CoSHC/CodeBERT/models/{dataset_type}_coshc",
        "--model_name_or_path", f"/content/drive/MyDrive/CoSHC/CodeBERT/models/{dataset_type}_coshc/hash_checkpoint_epoch_9.pt",
        "--tokenizer_name", "microsoft/codebert-base",
        "--do_embed",
        "--embedding_dir", f"/content/drive/MyDrive/CoSHC/CodeBERT/models/{dataset_type}_coshc/code_embeddings",
        "--eval_batch_size", "64",
        "--num_clusters", "10",
        "--total_recall", "100"
    ]

    # Parse the arguments
    args = parser.parse_args(args_list)

    start_time = time.time()
    results = run_coshc_main(args)
    total_time = time.time() - start_time
    
    return {
        "Success@1": results["Success@1"],
        "Success@5": results["Success@5"],
        "Success@10": results["Success@10"],
        "MRR": results["MRR"],
        "TotalTime": total_time,
        "SimilarityTime": results["SimilarityTime"],
        "SortingTime": results["SortingTime"]
    }

def print_results_table(dataset_name, baseline_results, coshc_results):
    """Print formatted results table"""
    print(f"\n{'='*40}")
    print(f"Results for {dataset_name} Dataset")
    print(f"{'='*40}")
    
    # Table 1: Time Efficiency
    print("\nTime Efficiency Comparison:")
    print(f"{'Metric':<25} | {'Baseline':<10} | {'CoSHC':<10} | Improvement")
    print(f"{'-'*25}|{'-'*12}|{'-'*12}|{'-'*15}")
    print(f"{'Total Time (s)':<25} | {baseline_results['TotalTime']:.2f} | {coshc_results['TotalTime']:.2f} | ↓{(1 - coshc_results['TotalTime']/baseline_results['TotalTime'])*100:.2f}%")
    print(f"{'Similarity Time (s)':<25} | {baseline_results['SimilarityTime']:.2f} | {coshc_results['SimilarityTime']:.2f} | ↓{(1 - coshc_results['SimilarityTime']/baseline_results['SimilarityTime'])*100:.2f}%")
    print(f"{'Sorting Time (s)':<25} | {baseline_results['SortingTime']:.2f} | {coshc_results['SortingTime']:.2f} | ↓{(1 - coshc_results['SortingTime']/baseline_results['SortingTime'])*100:.2f}%")

    # Table 2: Accuracy Metrics
    print("\nAccuracy Metrics Comparison:")
    print(f"{'Metric':<10} | {'Baseline':<10} | {'CoSHC':<10} | Preservation")
    print(f"{'-'*10}|{'-'*12}|{'-'*12}|{'-'*15}")
    for metric in ['Success@1', 'Success@5', 'Success@10', 'MRR']:
        base_val = baseline_results[metric]
        coshc_val = coshc_results[metric]
        preservation = coshc_val/base_val*100
        print(f"{metric:<10} | {base_val:.4f} | {coshc_val:.4f} | {preservation:.2f}%")

if __name__ == "__main__":
    # Evaluate Python dataset
    py_coshc = evaluate_coshc("python")
    py_baseline = evaluate_baseline("python")
    print_results_table("Python", py_baseline, py_coshc)

    # Evaluate Java dataset
    java_coshc = evaluate_coshc("java")
    java_baseline = evaluate_baseline("java")
    print_results_table("Java", java_baseline, java_coshc)