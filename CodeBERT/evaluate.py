import argparse
import time
from run import main as run_main
from run_coshc import main as run_coshc_main

def evaluate_baseline(dataset_type):
    """Evaluate baseline CodeBERT model"""
    parser = argparse.ArgumentParser()
    
    args = [
        "--do_eval",
        "--eval_data_file", f"data/{dataset_type}/test.jsonl",
        "--codebase_file", f"data/{dataset_type}/codebase.jsonl",
        "--output_dir", "saved_models/baseline",
        "--model_name_or_path", "saved_models/baseline/checkpoint-best-mrr/model.bin",
        "--eval_batch_size", "64"
    ]
    
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
    
    args = [
        "--do_eval",
        "--eval_data_file", f"data/{dataset_type}/test.jsonl",
        "--codebase_file", f"data/{dataset_type}/codebase.jsonl",
        "--output_dir", "saved_models/coshc",
        "--model_name_or_path", "saved_models/coshc/hash_checkpoint_epoch_9.pt",
        "--eval_batch_size", "64",
        "--num_clusters", "10",
        "--total_recall", "100"
    ]
    
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
    py_baseline = evaluate_baseline("python")
    py_coshc = evaluate_coshc("python")
    print_results_table("Python", py_baseline, py_coshc)

    # Evaluate Java dataset
    java_baseline = evaluate_baseline("java")
    java_coshc = evaluate_coshc("java")
    print_results_table("Java", java_baseline, java_coshc)