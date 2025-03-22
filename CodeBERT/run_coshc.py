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
from transformers import RobertaConfig, RobertaTokenizer

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
        code_embs.append(model(code_inputs=batch[0].to(args.device))
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
    # ... [Add CoSHC-specific arguments]
    
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