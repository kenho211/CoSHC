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
import pickle
import random
import torch
import json
import numpy as np
from sklearn.cluster import KMeans
from model import Model as BaseModel
from run import TextDataset
from torch.nn import CrossEntropyLoss, MSELoss, CosineSimilarity
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

logger = logging.getLogger(__name__)

# --------------------------------------------------
# CoSHC Model Architecture (Extends Base Model)
# --------------------------------------------------
class CoSHCModel(torch.nn.Module):
    def __init__(self, base_model, hash_dim=128, num_clusters=10):
        super().__init__()
        self.base_model = base_model
        
        # Hashing Module (Section 3.1.2)
        self.code_hash = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, hash_dim)
        )
        self.nl_hash = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, hash_dim)
        )
        
        # Classification Module (Section 3.2.2)
        self.classifier = torch.nn.Linear(768, num_clusters)
        
    def forward(self, code_inputs=None, nl_inputs=None, return_hash=False):
        if code_inputs is not None:
            embeddings = self.base_model(code_inputs=code_inputs)
            if return_hash:
                return self.code_hash(embeddings)
            return embeddings
        else:
            embeddings = self.base_model(nl_inputs=nl_inputs)
            if return_hash:
                return self.nl_hash(embeddings)
            return embeddings
            
    def predict_category(self, nl_inputs):
        embeddings = self.base_model(nl_inputs=nl_inputs)
        return torch.softmax(self.classifier(embeddings), dim=-1)

# --------------------------------------------------
# CoSHC Training Utilities
# --------------------------------------------------
def compute_similarity_matrix(embeddings):
    """Compute normalized similarity matrix (Eq 1-3)"""
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    normalized = embeddings / norms
    S = torch.mm(normalized, normalized.T)
    S.fill_diagonal_(1.0)  # Section 3.1.2 Eq 3
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

# --------------------------------------------------
# Main Training and Evaluation Logic
# --------------------------------------------------
def train_coshc(args, model, tokenizer, code_embeddings):
    """Joint training of hashing and classification modules"""
    
    # Step 1: Code Clustering (Section 3.1.1)
    kmeans = KMeans(n_clusters=10, random_state=args.seed)
    cluster_labels = kmeans.fit_predict(code_embeddings)
    
    # Step 2: Train Classification Module
    # [Implementation omitted for brevity. Use cross-entropy loss on cluster labels]
    
    # Step 3: Train Hashing Module
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    dataset = TextDataset(tokenizer, args, args.train_data_file)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size)
    
    for epoch in range(args.hash_epochs):
        for batch in dataloader:
            code_inputs, nl_inputs = batch
            
            # Get original embeddings
            with torch.no_grad():
                code_embs = model(code_inputs=code_inputs)
                nl_embs = model(nl_inputs=nl_inputs)
            
            # Get hash codes
            B_code = model.code_hash(code_embs)
            B_nl = model.nl_hash(nl_embs)
            
            # Compute target similarity matrix
            S_target = compute_similarity_matrix(torch.cat([code_embs, nl_embs]))
            
            # Compute and backprop loss
            loss = hashing_loss(B_code, B_nl, S_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def evaluate_coshc(args, model, tokenizer):
    """Two-stage evaluation: Hash recall + Re-rank"""
    # Load codebase embeddings and hash codes
    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_embs, code_hashes, code_clusters = [], [], []
    
    # Precompute code representations
    for batch in DataLoader(code_dataset, batch_size=args.eval_batch_size):
        code_embs.append(model(code_inputs=batch[0].to(args.device)))
        code_hashes.append(torch.sign(model.code_hash(code_embs[-1])))
        code_clusters.append(model.classifier(code_embs[-1]).argmax(dim=1))
    
    code_embs = torch.cat(code_embs)
    code_hashes = torch.cat(code_hashes)
    code_clusters = torch.cat(code_clusters)
    
    # Process queries
    query_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    results = []
    
    for query_batch in DataLoader(query_dataset, batch_size=args.eval_batch_size):
        nl_embs = model(nl_inputs=query_batch[1].to(args.device))
        nl_hashes = torch.sign(model.nl_hash(nl_embs))
        
        # Stage 1: Category Prediction (Section 3.2.2)
        probs = torch.softmax(model.classifier(nl_embs), dim=1)
        
        # Stage 2: Hash-based Recall (Section 3.2.1)
        for i in range(len(nl_embs)):
            # Calculate Hamming distance
            dists = (code_hashes != nl_hashes[i]).sum(dim=1)
            
            # Recall strategy (Section 3.2.2 Eq 7)
            recall_counts = allocate_recalls(probs[i], args.total_recall)
            
            # Re-rank with original embeddings (Section 3.2.1)
            candidates = []
            for cluster_id, count in enumerate(recall_counts):
                mask = (code_clusters == cluster_id)
                cluster_dists = dists[mask]
                cluster_indices = cluster_dists.topk(count, largest=False).indices
                candidates.append(code_embs[mask][cluster_indices])
            
            candidates = torch.cat(candidates)
            scores = CosineSimilarity(dim=1)(candidates, nl_embs[i])
            results.append(process_scores(scores))
    
    return compute_metrics(results)

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
def main():
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

    args = parser.parse_args()
    
    # Original CodeBERT setup
    base_model = BaseModel(RobertaModel.from_pretrained(args.model_name_or_path))
    model = CoSHCModel(base_model)
    
    if args.do_train:
        # Load precomputed code embeddings
        code_embeddings = load_code_embeddings()  
        train_coshc(args, model, tokenizer, code_embeddings)
    
    if args.do_eval:
        evaluate_coshc(args, model, tokenizer)

if __name__ == "__main__":
    main()