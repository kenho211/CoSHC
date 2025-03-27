# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Implementation of CoSHC: Accelerating Code Search with Deep Hashing and Code Classification.
Reference: 2022 ACL Paper (Section 3: Method)
"""

import argparse
import logging
import os
import time
import pickle
import random
import torch # type: ignore
import json
import numpy as np
from sklearn.cluster import KMeans # type: ignore
from model import Model as BaseModel, CoSHCModel
from run import TextDataset
from torch.nn import CrossEntropyLoss, MSELoss, CosineSimilarity # type: ignore
from torch.utils.data import DataLoader, Dataset, SequentialSampler # type: ignore
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel # type: ignore
from datetime import datetime

logger = logging.getLogger(__name__)

# --------------------------------------------------
# CoSHC Training Utilities
# --------------------------------------------------
def compute_similarity_matrix(code_embs, nl_embs, beta, eta, batch_size):
    """Compute normalized similarity matrix (Eq 1-3)"""
    # Normalize code embeddings and compute S_C
    code_norms = torch.norm(code_embs, p=2, dim=1, keepdim=True)
    code_normalized = code_embs / code_norms
    S_C = torch.mm(code_normalized, code_normalized.T)
    
    # Normalize nl embeddings and compute S_D
    nl_norms = torch.norm(nl_embs, p=2, dim=1, keepdim=True)
    nl_normalized = nl_embs / nl_norms
    S_D = torch.mm(nl_normalized, nl_normalized.T)
    
    # Combine S_C and S_D using beta (Eq 1)
    S_tilde = beta * S_C + (1 - beta) * S_D
    
    # Compute high-order similarity (Eq 2)
    S_tilde_T = torch.mm(S_tilde, S_tilde.T) / batch_size
    S = (1 - eta) * S_tilde + eta * S_tilde_T
    
    # Set diagonal to 1.0 (Eq 3)
    S.fill_diagonal_(1.0)
    
    return S


def hashing_loss(B_code, B_nl, S_target, mu=1.5, lambda1=0.1, lambda2=0.1):
    """Deep hashing loss function (Section 3.1.2 Eq 4)"""
    sim_target = torch.clamp(mu * S_target, max=1.0)
    
    # Code-Query similarity
    sim_cq = torch.mm(B_nl, B_code.T) / B_code.size(1)
    loss_cq = torch.norm(sim_target - sim_cq)**2
    
    # Code-Code similarity
    sim_cc = torch.mm(B_code, B_code.T) / B_code.size(1)
    loss_cc = torch.norm(sim_target - sim_cc)**2
    
    # Query-Query similarity
    sim_nn = torch.mm(B_nl, B_nl.T) / B_nl.size(1)
    loss_nn = torch.norm(sim_target - sim_nn)**2
    
    return loss_cq + lambda1*loss_cc + lambda2*loss_nn


def save_code_embeddings(model, dataloader, output_dir, device):
    """
    Generate and save code embeddings using the base CodeBERT model
    Paper Reference: Section 3.1 Offline Processing
    """
    model.eval()
    all_embeddings = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in dataloader:
            code_inputs = batch[0].to(device)
            embeddings = model(code_inputs=code_inputs)
            all_embeddings.append(embeddings.cpu())
    
    # Concatenate and save
    code_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(code_embeddings, os.path.join(output_dir, "embeddings.pt"))
    
    # Save metadata
    metadata = {
        "total_codes": code_embeddings.shape[0],
        "dimension": code_embeddings.shape[1],
        "created_at": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
        
    logger.info(f"Saved {metadata['total_codes']} code embeddings to {output_dir}")


def load_code_embeddings(embedding_dir: str = "./code_embeddings",
                         device: torch.device = None) -> torch.Tensor:
    """
    Load precomputed code embeddings for CoSHC initialization
    Paper Reference: Section 3.1.1 (Offline Processing)
    
    Args:
        embedding_dir: Path to directory containing:
            - embeddings.pt: Tensor of shape [num_codes, 768]
            - metadata.json: {"total_codes": N, "dimension": 768}
        device: Target device for embeddings (default: CPU for clustering)
    
    Returns:
        Tensor: Float tensor of shape [num_codes, hidden_dim]
    """
    try:
        # Validate directory structure
        if not os.path.exists(embedding_dir):
            raise FileNotFoundError(f"Embedding directory {embedding_dir} not found")
            
        # Load metadata
        with open(os.path.join(embedding_dir, "metadata.json")) as f:
            metadata = json.load(f)
        
        # Load embeddings tensor
        embeddings_path = os.path.join(embedding_dir, "embeddings.pt")
        code_embeddings = torch.load(embeddings_path, map_location='cpu')
        
        # Validate dimensions
        expected_shape = (metadata["total_codes"], metadata["dimension"])
        if code_embeddings.shape != expected_shape:
            raise ValueError(f"Embedding shape mismatch. Expected {expected_shape}, "
                             f"got {code_embeddings.shape}")
        
        # Paper-recommended preprocessing (Sec 3.1.1)
        code_embeddings = code_embeddings / torch.norm(code_embeddings, dim=1, keepdim=True)
        
        return code_embeddings.to(device) if device else code_embeddings

    except Exception as e:
        logger.error(f"Failed to load code embeddings: {str(e)}")
        raise

# --------------------------------------------------
# Main Training and Evaluation Logic
# --------------------------------------------------
def train_coshc(args, model, tokenizer, code_embeddings):
    """Joint training of hashing and classification modules"""
    
    # Step 1: Code Clustering (Section 3.1.1)
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed)
    cluster_labels = kmeans.fit_predict(code_embeddings.cpu())
    # Convert cluster labels to tensor once, instead of inside loop
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long).to(args.device)
    logger.info("Code clustering completed")
    
    # Step 2: Train Classification Module
    model.train()
    classifier_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    classifier_criterion = CrossEntropyLoss()
    logger.info("Classification module initialized")
    
    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    # Increase num_workers for faster data loading
    code_dataloader = DataLoader(code_dataset, 
                               batch_size=args.eval_batch_size,
                               num_workers=4,
                               pin_memory=True)  # Enable pin_memory for faster GPU transfer
    
    for epoch in range(args.class_epochs):
        total_loss = 0
        for i, batch in enumerate(code_dataloader):
            # Move entire batch to GPU at once
            code_inputs = batch[0].to(args.device, non_blocking=True)
            
            # Get batch labels directly from pre-computed tensor
            start_idx = i * args.eval_batch_size
            end_idx = start_idx + len(code_inputs)
            batch_labels = cluster_labels[start_idx:end_idx]
            
            # Forward pass
            code_embs = model(code_inputs=code_inputs)
            cluster_pred = model.classifier(code_embs)
            
            # Compute loss and backprop
            loss = classifier_criterion(cluster_pred, batch_labels)
            loss.backward()
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Log progress every N batches
            if (i + 1) % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {i+1}, Avg Loss: {total_loss / (i+1):.4f}")

            if args.debug:
                if (i + 1) % 500 == 0:
                    break
        
        logger.info(f"Epoch {epoch} completed, Avg Loss: {total_loss / len(code_dataloader):.4f}")


    # Step 3: Train Hashing Module
    logger.info("Training hashing module")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.hash_epochs)
    
    dataset = TextDataset(tokenizer, args, args.train_data_file)
    dataloader = DataLoader(dataset, 
                          batch_size=args.train_batch_size,
                          num_workers=4,
                          pin_memory=True)
    
    for epoch in range(args.hash_epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # Move data to GPU
            code_inputs = batch[0].to(args.device, non_blocking=True)
            nl_inputs = batch[1].to(args.device, non_blocking=True)

            # Get original embeddings
            with torch.no_grad():
                code_embs = model(code_inputs=code_inputs)
                nl_embs = model(nl_inputs=nl_inputs)

            # Compute target similarity matrix before hashing
            S_target = compute_similarity_matrix(code_embs, nl_embs, args.beta, args.eta, args.train_batch_size)
            
            # Get hash codes
            B_code = model.get_binary_hash(code_embs, is_code=True, apply_tanh=True)
            B_nl = model.get_binary_hash(nl_embs, is_code=False, apply_tanh=True)

            # Compute loss and backprop
            loss = hashing_loss(B_code, B_nl, S_target, mu=args.mu, lambda1=args.lambda1, lambda2=args.lambda2)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Log progress every N batches
            if (i + 1) % 50 == 0:
                avg_loss = total_loss / (i + 1)
                logger.info(f"Epoch {epoch}, Batch {i+1}, Avg Loss: {avg_loss:.4f}")
            
            # Clear GPU cache periodically
            if (i + 1) % 500 == 0:
                torch.cuda.empty_cache()
            
            if args.debug and (i + 1) % 500 == 0:
                break

        
        # Step learning rate scheduler
        scheduler.step()
        
        # Log epoch stats
        avg_epoch_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch} completed, Avg Loss: {avg_epoch_loss:.4f}")
        model.alpha += 1.0
        
        # Save checkpoint after each epoch
        if not args.debug:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'hash_checkpoint_epoch_{epoch}.pt'))
    
    logger.info("Hashing module training completed")

def evaluate_coshc(args, model, tokenizer, code_embeddings):
    """Two-stage evaluation: Hash recall + Re-rank"""
    query_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    query_urls = [example.url for example in query_dataset.examples]
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
        
    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_urls = [example.url for example in code_dataset.examples]
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    all_code_embs = []
    all_code_hashes = []
    all_code_clusters = []

    # Precompute code representations
    logger.info("Precomputing code representations")
    for idx, batch in enumerate(code_dataloader):
        logger.info(f"Precomputing code representations batch {idx} of {len(code_dataloader)}")
        # Move data to GPU
        code_inputs = batch[0].to(args.device, non_blocking=True)

        # Get original embeddings
        with torch.no_grad():
            code_embs = model(code_inputs=code_inputs)

        code_hashes = model.get_binary_hash(code_embs, is_code=True)

        code_clusters = model.classifier(code_embs)

        all_code_embs.append(code_embs)
        all_code_hashes.append(code_hashes)
        all_code_clusters.append(code_clusters)

        if idx > 2:
            break
    
    all_code_embs = torch.cat(all_code_embs)
    all_code_hashes = torch.cat(all_code_hashes)
    all_code_clusters = torch.cat(all_code_clusters)
    logger.info("Precomputing code representations completed")


    # Process queries
    results = []
    similarity_time = 0
    sorting_time = 0
    overall_start_time = time.time()

    logger.info("Processing queries")
    for query_index, query_batch in enumerate(query_dataloader):
        logger.info(f"Processing query batch {query_index} of {len(query_dataloader)}")

        # Move data to GPU
        nl_inputs = query_batch[1].to(args.device, non_blocking=True)
        # Get original embeddings
        with torch.no_grad():
            nl_embs = model(nl_inputs=nl_inputs)
        
        nl_hashes = model.get_binary_hash(nl_embs, is_code=False)
        
        # Stage 1: Category Prediction (Section 3.2.2)
        probs = torch.softmax(model.classifier(nl_embs), dim=1)
        # logger.info(f"Probabilities: {probs}")
        
        start_idx = query_index * args.eval_batch_size
        end_idx = start_idx + len(nl_inputs)
        query_batch_urls = query_urls[start_idx:end_idx]
        
        # Stage 2: Hash-based Recall (Section 3.2.1)
        for i in range(len(nl_embs)):
            # Get current query URL
            query_url = query_batch_urls[i]
            
            # Calculate Hamming distance
            dists = (all_code_hashes != nl_hashes[i]).sum(dim=1)
            # logger.info(f"Hamming distance: {dists}")
            # logger.info(f"Hamming distance shape: {dists.shape}")
            
            # Recall strategy (Section 3.2.2 Eq 7)
            recall_counts = allocate_recalls(probs[i], args.total_recall, args.num_clusters)
            
            # Re-rank with original embeddings (Section 3.2.1)
            candidates = []
            candidate_indices = []  # Track indices for URL mapping
            
            for cluster_id, count in enumerate(recall_counts):
                mask = (torch.argmax(all_code_clusters, dim=1) == cluster_id)
                # logger.info(f"Mask: {mask}")
                # logger.info(f"Mask shape: {mask.shape}")
                cluster_dists = dists[mask]
                # Get indices within cluster
                cluster_indices = cluster_dists.topk(min(count, len(cluster_dists)), largest=False).indices
                # Map to original indices
                original_indices = torch.where(mask)[0][cluster_indices]
                candidate_indices.extend(original_indices.tolist())
                candidates.append(all_code_embs[mask][cluster_indices])
            
            candidates = torch.cat(candidates)
            
            start_time = time.time()
            scores = CosineSimilarity(dim=1)(candidates, nl_embs[i])
            similarity_time += time.time() - start_time
            
            # Get candidate URLs using tracked indices
            batch_candidate_urls = [code_urls[idx] for idx in candidate_indices]
            
            result = process_scores(scores, batch_candidate_urls, query_url, args.total_recall)
            sorting_time += result['sorting_time']
            results.append(result)
    
    end_time = time.time()
    retrieval_time = end_time - overall_start_time
    metrics = compute_metrics(results, args.total_recall)
    metrics["RetrievalTime"] = retrieval_time
    metrics["SimilarityTime"] = similarity_time
    metrics["SortingTime"] = sorting_time
    return metrics


def allocate_recalls(probs: torch.Tensor, total_recall: int, num_clusters: int) -> torch.Tensor:
    """
    Allocate candidates per cluster using CoSHC's distribution strategy
    Paper Reference: Section 3.2.2 Eq.7
    
    Args:
        probs: Normalized probability distribution over clusters [num_clusters]
        total_recall: Total candidates to recall (N)
        num_clusters: Number of clusters (k)
    
    Returns:
        Tensor: Long tensor of shape [num_clusters] with per-cluster recall counts
    """
    base_recall = total_recall - num_clusters  # Remaining after 1 per cluster
    allocations = torch.floor(probs * base_recall).long()
    
    # Distribute remainder to highest probability clusters
    remainder = base_recall - allocations.sum().item()
    if remainder > 0:
        _, top_indices = torch.topk(probs - (allocations/base_recall), remainder)
        allocations[top_indices] += 1
    
    # Add minimum 1 per cluster
    return allocations + 1


def process_scores(scores: torch.Tensor, 
                 candidate_urls: list, 
                 query_url: str,
                 total_recall: int) -> dict:
    """
    Calculate ranking metrics for a single query
    Paper Reference: Section 4.4 Evaluation Metrics
    
    Args:
        scores: Cosine similarity scores [num_candidates]
        candidate_urls: URLs of candidate codes [num_candidates]
        query_url: Ground truth URL for the query
        total_recall: Total candidates considered
    
    Returns:
        dict: Contains rank and success flags
    """
    start_time = time.time()

    # Sort candidates by descending similarity
    sorted_indices = torch.argsort(scores, descending=True)
    sorting_time = time.time() - start_time

    sorted_urls = [candidate_urls[i] for i in sorted_indices.cpu().numpy()]
    try:
        rank = sorted_urls.index(query_url) + 1  # 1-based indexing
    except ValueError:
        rank = total_recall + 1  # Not found penalty
        
    return {
        'rank': rank,
        'success@1': rank <= 1,
        'success@5': rank <= 5,
        'success@10': rank <= 10,
        'sorting_time': sorting_time
    }


def compute_metrics(results: list, total_recall: int) -> dict:
    """
    Aggregate metrics across all queries
    Paper Reference: Section 4.5 Experimental Results
    
    Args:
        results: List of metric dicts from process_scores()
        total_recall: Total candidates considered per query
    
    Returns:
        dict: Final evaluation metrics
    """
    ranks = []
    success_at_1 = []
    success_at_5 = []
    success_at_10 = []
    
    for res in results:
        ranks.append(res['rank'])
        success_at_1.append(res['success@1'])
        success_at_5.append(res['success@5'])
        success_at_10.append(res['success@10'])
    
    # Calculate MRR (handle not found as rank=total_recall+1)
    reciprocal_ranks = [1/r if r <= total_recall else 0 for r in ranks]
    mrr = np.mean(reciprocal_ranks)
    
    # Calculate Success@k
    return {
        'MRR': round(mrr, 4),
        'Success@1': round(np.mean(success_at_1), 4),
        'Success@5': round(np.mean(success_at_5), 4),
        'Success@10': round(np.mean(success_at_10), 4),
        'RetrievalTime': None,
    }

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()

        # ----------------------------
        # Original CodeBERT Arguments
        # ----------------------------
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

        args = parser.parse_args()

    print("args: %s", args)
    
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)

    # build baseline CodeBERT model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    _base_model = RobertaModel.from_pretrained(args.tokenizer_name)
    base_model = BaseModel(_base_model)

    if args.debug:
        logger.info("Debug mode enabled")
        args.train_batch_size = 1
        args.eval_batch_size = 1
        args.num_train_epochs = 1
        args.hash_epochs = 1
        args.class_epochs = 1

    # load pretrained CodeBERT model
    logger.info("Loading pretrained CodeBERT model from %s", args.model_name_or_path)
    logger.info("output_dir: %s", args.output_dir)
    base_model.load_state_dict(torch.load(args.model_name_or_path, map_location=args.device), strict=True)
    base_model.to(args.device)
    logger.info("Base model loaded")

    # save code embeddings (only need to run once)
    if args.do_embed \
        and not os.path.exists(os.path.join(args.embedding_dir, "embeddings.pt")):
        os.makedirs(args.embedding_dir, exist_ok=True)
        code_dataset = TextDataset(tokenizer, args, args.codebase_file)
        code_dataloader = DataLoader(code_dataset, batch_size=args.eval_batch_size)
        save_code_embeddings(base_model, code_dataloader, args.embedding_dir, args.device)
        logger.info("Code embeddings saved to %s", args.embedding_dir)
    else:
        logger.info("Code embeddings already exist in %s", args.embedding_dir)

    # build CoSHC model
    model = CoSHCModel(base_model)
    model = model.to(args.device)
    logger.info("CoSHC model loaded")

    if args.do_train:
        # Load precomputed code embeddings
        code_embeddings = load_code_embeddings(args.embedding_dir, args.device)
        train_coshc(args, model, tokenizer, code_embeddings)
    else:
        logger.info("Skipping training")
    
    if args.do_eval:
        code_embeddings = load_code_embeddings(args.embedding_dir, args.device)
        checkpoint = torch.load(args.coshc_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate_coshc(args, model, tokenizer, code_embeddings)
    else:
        logger.info("Skipping evaluation")

if __name__ == "__main__":
    main()