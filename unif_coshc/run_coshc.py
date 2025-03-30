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
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, CosineSimilarity
from torch.utils.data import DataLoader, SequentialSampler, Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import KMeans
from fasttext import train_unsupervised, load_model
from codesearch.encoders import BasicEncoder
from codesearch.unif.unif_embedder import UNIFEmbedder
from codesearch.unif.unif_modules import SimilarityModel
from codesearch.unif.unif_preprocessing import pad_sequences, extract_mask, fasttext_preprocess
from codesearch.text_preprocessing import preprocess_text
from codesearch.code_preprocessing import code_tokenization
from functools import partial
from pathlib import Path
import time

logger = logging.getLogger(__name__)

# CoSHC Utilities
def compute_similarity_matrix(code_embs, nl_embs, beta, eta, batch_size):
    code_norms = torch.norm(code_embs, p=2, dim=1, keepdim=True)
    code_normalized = code_embs / code_norms
    S_C = torch.mm(code_normalized, code_normalized.T)
    nl_norms = torch.norm(nl_embs, p=2, dim=1, keepdim=True)
    nl_normalized = nl_embs / nl_norms
    S_D = torch.mm(nl_normalized, nl_normalized.T)
    S_tilde = beta * S_C + (1 - beta) * S_D
    S_tilde_T = torch.mm(S_tilde, S_tilde.T) / batch_size
    S = (1 - eta) * S_tilde + eta * S_tilde_T
    S.fill_diagonal_(1.0)
    return S

def hashing_loss(B_code, B_nl, S_target, mu=1.5, lambda1=0.1, lambda2=0.1):
    sim_target = torch.clamp(mu * S_target, max=1.0)
    sim_cq = torch.mm(B_nl, B_code.T) / B_code.size(1)
    loss_cq = torch.norm(sim_target - sim_cq)**2
    sim_cc = torch.mm(B_code, B_code.T) / B_code.size(1)
    loss_cc = torch.norm(sim_target - sim_cc)**2
    sim_nn = torch.mm(B_nl, B_nl.T) / B_nl.size(1)
    loss_nn = torch.norm(sim_target - sim_nn)**2
    return loss_cq + lambda1 * loss_cc + lambda2 * loss_nn

# Custom Collate Function
def custom_collate_fn(batch):
    code_batch = [item[0] for item in batch]  # List of code token lists
    query_batch = [item[1] for item in batch]  # List of query token lists
    return code_batch, query_batch

from dataclasses import dataclass

@dataclass
class InputFeatures:
    code: list
    query: list
    url: str = ""

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, args, file_path):
        self.examples = []
        with open(file_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.examples.append({
                    "code": entry["code_tokens"],  # Already tokenized list
                    "query": entry["docstring_tokens"],  # Already tokenized list
                    "url": entry.get("url", f"ex_{len(self.examples)}")
                })
        self.args = args
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Return pre-tokenized data directly
        code_tokens = self.examples[idx]["code"]
        query_tokens = self.examples[idx]["query"]
        return (code_tokens, query_tokens)

def collate_fn(batch):
    """Pad sequences to max length in batch"""
    code_batch, query_batch = zip(*batch)
    
    # Pad with zeros
    code_padded = pad_sequence(code_batch, batch_first=True, padding_value=0)
    query_padded = pad_sequence(query_batch, batch_first=True, padding_value=0)
    
    return code_padded, query_padded
            
def convert_examples_to_features(js,tokenizer,args):
    #code
    code=' '.join(js['code_tokens'])
    code_tokens=tokenizer.tokenize(code)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length
    
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'])


class DummyTokenizer:
    def __init__(self):
        # Add minimal HuggingFace tokenizer attributes
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        
    def __call__(self, texts, **kwargs):
        """Handle batch processing"""
        if isinstance(texts, str):
            return {"input_ids": [self.convert_tokens_to_ids(texts.split())]}
        return {"input_ids": [self.convert_tokens_to_ids(t) for t in texts]}
    
    def tokenize(self, text):
        """If text is already tokenized, return as-is"""
        if isinstance(text, list):
            return text
        return text.split()  # Fallback for raw strings
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to numerical IDs (dummy mapping)"""
        if isinstance(tokens, str):
            tokens = [tokens]
        # Simple hash-based ID mapping (consistent but arbitrary)
        return [hash(token) % 10000 for token in tokens]
    
    def encode(self, text, **kwargs):
        """Handle single-text encoding like HF tokenizers"""
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)
    
    def batch_encode_plus(self, texts, **kwargs):
        """Batch processing interface"""
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids, skip_special_tokens=True):
        """Convert IDs back to text (simplified)"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join([f"token_{id}" for id in ids])


class BaseModel(torch.nn.Module):
    def __init__(self, embedder, device=None):
        super().__init__()
        self.embedder = embedder
        self.device = device if device is not None else torch.device("cpu")
    
    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            if not code_inputs:
                return None
            # Assume code_inputs is a list of tokenized lists
            snippets = [{"code": code, "language": "java"} for code in code_inputs]
            embeddings = self.embedder.embed_snippets(snippets)
            if embeddings is None:
                return None
            return torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            
        elif nl_inputs is not None:
            if not nl_inputs:
                return None
            # Assume nl_inputs is a list of tokenized lists
            embeddings = self.embedder.embed_queries(nl_inputs)
            if embeddings is None:
                return None
            return torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        return None

class CoSHCModel(torch.nn.Module):
    def __init__(self, base_model, hash_dim=128, num_clusters=10, device = None):
        super(CoSHCModel, self).__init__()
        self.base_model = base_model
        self.device = device if device is not None else torch.device("cpu")
        embed_dim = base_model.embedder.ft_model.get_dimension()
         # Hashing Module (Section 3.1.2)
        self.code_hash = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, hash_dim),
            torch.nn.Tanh()
        )
        self.nl_hash = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, hash_dim),
            torch.nn.Tanh()
        )
        
        # self.code_hash = torch.nn.Linear(embed_dim, hash_dim)
        # self.nl_hash = torch.nn.Linear(embed_dim, hash_dim)
        self.alpha = 1.0
        # self.classifier = torch.nn.Linear(embed_dim, num_clusters)
        self.classifier = torch.nn.Linear(embed_dim, num_clusters)
    
    def get_binary_hash(self, embeddings, is_code=True, apply_tanh=False):
        inputs = embeddings.to(dtype=torch.float32, device=self.device)

        if is_code:
            h = self.code_hash[:-1](inputs)
        else:
            h = self.nl_hash[:-1](inputs)
        
        # Apply equation 6: tanh(alpha * H)
        if apply_tanh:
            return torch.tanh(self.alpha * h)
        else:
            return torch.sign(h)  # Equation 5
    
    def forward(self, code_inputs=None, nl_inputs=None):
        return self.base_model(code_inputs=code_inputs, nl_inputs=nl_inputs)

# Training UNIFEmbedder
def train_unif_embedder(train_file, output_dir, lang):
    output_dir = Path(output_dir) / lang  # e.g., unif_model/java
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 

    data = []
    with open(train_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    snippets = [{"code": entry["code_tokens"], "language": lang} for entry in data]  # Use specified lang
    queries = [entry["docstring_tokens"] for entry in data]
    
    ft_data_file = f"fasttext_training_data_{lang}.txt"
    with open(ft_data_file, "w") as f:
        for entry in data:
            f.write(" ".join(entry["code_tokens"]) + "\n")
            f.write(" ".join(entry["docstring_tokens"]) + "\n")
    ft_model = train_unsupervised(
        ft_data_file, model="skipgram", dim=100, epoch=10, lr=0.05, minCount=1
    )
    ft_model.save_model(str(output_dir / f"fasttext_model_{lang}.bin"))
    
    encoder = BasicEncoder(
        description_preprocessor=partial(preprocess_text, lemmatize=True, remove_stop=True),
        code_preprocessor=partial(code_tokenization, language=lang)
    )
    
    sim_model = SimilarityModel(ft_model)
    optimizer = torch.optim.Adam(sim_model.parameters(), lr=1e-3)
    sim_model.train()
    
    for epoch in range(3):
        for i in range(0, len(snippets), 32):
            batch_snippets = snippets[i:i + 32]
            batch_queries = queries[i:i + 32]
            snippet_enc = [s["code"] for s in batch_snippets]
            snippet_enc = pad_sequences(snippet_enc, 200)
            code_mask = extract_mask(snippet_enc)
            snippet_enc = fasttext_preprocess([t for s in snippet_enc for t in s], ft_model)
            code_embs = sim_model.code_embedder(snippet_enc, code_mask)
            
            query_enc = [encoder.encode_description(q) for q in batch_queries]
            print(query_enc)
            query_enc = pad_sequences(query_enc, 25)
            query_mask = extract_mask(query_enc)
            query_enc = fasttext_preprocess([t for q in query_enc for t in q], ft_model)
            query_embs = sim_model.description_embedder(query_enc, query_mask)
            
            loss = -torch.nn.functional.cosine_similarity(code_embs, query_embs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(i)
        logger.info(f"UNIFEmbedder Epoch {epoch + 1}, Loss: {loss.item()}")
    
    embedder = UNIFEmbedder(sim_model, encoder, ft_model, batch_size=32, max_code_len=200, max_description_len=25)
    embedder.save(output_dir)
    return embedder


def train_coshc(args, model, train_file):
    dataset = TextDataset(args, train_file)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=4, collate_fn=custom_collate_fn)
    
    base_query_embs, base_code_embs = [], []
    hash_query_embs, hash_code_embs = [], []
    
    embedding_dir = Path("embeddings/train") / args.lang
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    base_code_path = embedding_dir / f"base_code_embeddings_{args.lang}.pt"
    base_query_path = embedding_dir / f"base_query_embeddings_{args.lang}.pt"
    hash_code_path = embedding_dir / f"hash_code_embeddings_{args.lang}.pt"
    hash_query_path = embedding_dir / f"hash_query_embeddings_{args.lang}.pt"
    
    if (base_code_path.exists() and base_query_path.exists() and 
        hash_code_path.exists() and hash_query_path.exists()):
        logger.info(f"Loading existing training embeddings for {args.lang}")
        base_code_embs = torch.load(base_code_path, map_location=args.device)
        base_query_embs = torch.load(base_query_path, map_location=args.device)
        hash_code_embs = torch.load(hash_code_path, map_location=args.device)
        hash_query_embs = torch.load(hash_query_path, map_location=args.device)
    else:
        for batch in dataloader:
            code_embs = model(code_inputs=batch[0])
            nl_embs = model(nl_inputs=batch[1])
            if code_embs is None or nl_embs is None:
                logger.error("Embeddings are None for batch")
                continue
            base_code_embs.append(code_embs.cpu())
            base_query_embs.append(nl_embs.cpu())
            hash_c_embs = model.get_binary_hash(code_embs, is_code=True).cpu()
            hash_q_embs = model.get_binary_hash(nl_embs, is_code=False).cpu()
            hash_code_embs.append(hash_c_embs)
            hash_query_embs.append(hash_q_embs)
    
        if not base_code_embs or not base_query_embs:
            logger.error("No valid embeddings generated during training")
            return
    
        base_code_embs = torch.cat(base_code_embs).detach()
        base_query_embs = torch.cat(base_query_embs).detach()
        hash_code_embs = torch.cat(hash_code_embs).detach()
        hash_query_embs = torch.cat(hash_query_embs).detach()
    
        embedding_dir = Path("embeddings/train") / args.lang
        embedding_dir.mkdir(parents=True, exist_ok=True)
        torch.save(base_code_embs, embedding_dir / f"base_code_embeddings_{args.lang}.pt")
        torch.save(base_query_embs, embedding_dir / f"base_query_embeddings_{args.lang}.pt")
        torch.save(hash_code_embs, embedding_dir / f"hash_code_embeddings_{args.lang}.pt")
        torch.save(hash_query_embs, embedding_dir / f"hash_query_embeddings_{args.lang}.pt")
        logger.info(f"Saved base code embeddings to {embedding_dir / f'base_code_embeddings_{args.lang}.pt'}")
        logger.info(f"Saved base query embeddings to {embedding_dir / f'base_query_embeddings_{args.lang}.pt'}")
        logger.info(f"Saved hashed code embeddings to {embedding_dir / f'hash_code_embeddings_{args.lang}.pt'}")
        logger.info(f"Saved hashed query embeddings to {embedding_dir / f'hash_query_embeddings_{args.lang}.pt'}")
    
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed)
    cluster_labels = torch.tensor(kmeans.fit_predict(base_code_embs.cpu()), dtype=torch.long).to(args.device)
    
    model.train()
    classifier_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()

    
    for epoch in range(args.class_epochs):
        total_train_loss = 0
        last_train_loss = np.inf
        for i, batch in enumerate(dataloader):
            code_embs = model(code_inputs=batch[0])
            if code_embs is None:
                continue
            cluster_pred = model.classifier(code_embs)
            batch_labels = cluster_labels[i * args.train_batch_size:(i + 1) * args.train_batch_size]
            loss = criterion(cluster_pred, batch_labels)
            loss.backward()
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()
            total_train_loss += loss.item()
        logger.info(f"Classifier Epoch {epoch}, Train Loss: {total_train_loss / len(dataloader):.4f}")
        if total_train_loss >= last_train_loss:
            break
        else:
            last_train_loss = total_train_loss
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.hash_epochs)
    
    for epoch in range(args.hash_epochs):
        total_train_loss = 0
        last_train_loss = np.inf
        for batch in dataloader:
            code_embs = model(code_inputs=batch[0])
            nl_embs = model(nl_inputs=batch[1])
            if code_embs is None or nl_embs is None:
                continue
            S_target = compute_similarity_matrix(code_embs, nl_embs, args.beta, args.eta, args.train_batch_size)
            B_code = model.get_binary_hash(code_embs, is_code=True, apply_tanh=True) #apply_tanh=True
            B_nl = model.get_binary_hash(nl_embs, is_code=False, apply_tanh=True)
            loss = hashing_loss(B_code, B_nl, S_target, args.mu, args.lambda1, args.lambda2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

        scheduler.step()
        logger.info(f"Hashing Epoch {epoch}, Train Loss: {total_train_loss / len(dataloader):.4f}")
        model.alpha += 1.0
        if total_train_loss >= last_train_loss:
            break
        else:
            last_train_loss = total_train_loss


def evaluate_coshc(args, model, valid_file):
    dataset = TextDataset(args, valid_file)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=custom_collate_fn)
    
    # Precompute code representations
    all_code_embs = []
    all_code_hashes = []
    all_urls = [ex["url"] for ex in dataset.examples]
    
    model.eval()
    logger.info("Precomputing code representations")
    for batch in dataloader:
        c_embs = model(code_inputs=batch[0])
        if c_embs is not None:
            all_code_embs.append(c_embs.cpu())
            hash_c_embs = model.get_binary_hash(c_embs, is_code=True).cpu()
            all_code_hashes.append(hash_c_embs)
    
    if not all_code_embs:
        logger.error("No valid code embeddings for evaluation")
        return
    
    all_code_embs = torch.cat(all_code_embs).to(args.device)
    all_code_hashes = torch.cat(all_code_hashes).to(args.device)
    
    # Save embeddings
    embedding_dir = Path("embeddings/valid") / args.lang
    embedding_dir.mkdir(parents=True, exist_ok=True)
    base_code_embs = all_code_embs.detach().cpu().numpy()
    hash_code_embs = all_code_hashes.detach().cpu().numpy()
    np.save(embedding_dir / f"base_code_embeddings_{args.lang}.npy", base_code_embs)
    np.save(embedding_dir / f"hash_code_embeddings_{args.lang}.npy", hash_code_embs)
    logger.info(f"Saved base code embeddings to {embedding_dir / f'base_code_embeddings_{args.lang}.npy'}")
    logger.info(f"Saved hashed code embeddings to {embedding_dir / f'hash_code_embeddings_{args.lang}.npy'}")
    
    # Process queries with hash-based evaluation
    hash_total_time = 0
    hash_similarity_time = 0
    hash_sorting_time = 0
    base_total_time = 0
    base_similarity_time = 0
    base_sorting_time = 0
    hash_results = []
    base_results = []
    
    logger.info("Processing queries with hash-based evaluation")
    for batch_idx, batch in enumerate(dataloader):
        q_embs = model(nl_inputs=batch[1])
        if q_embs is None:
            continue
        q_hashes = model.get_binary_hash(q_embs, is_code=False)
        
        start_idx = batch_idx * args.eval_batch_size
        end_idx = min(start_idx + len(q_embs), len(all_urls))
        query_urls = all_urls[start_idx:end_idx]
        
        for i in range(len(q_embs)):
            query_emb = q_embs[i:i+1]
            query_hash = q_hashes[i:i+1]
            query_url = query_urls[i]
            correct_idx = start_idx + i
            
            # --- Hashing Evaluation (No Clustering) ---
            hash_start_time = time.time()
            
            # Select candidates using Hamming distance only
            dists = (all_code_hashes != query_hash).sum(dim=1)
            candidate_indices = dists.topk(args.total_recall, largest=False).indices
            candidates = all_code_embs[candidate_indices]
            batch_candidate_urls = [all_urls[idx] for idx in candidate_indices.tolist()]
            
            # Re-rank with original embeddings
            sim_start = time.time()
            hash_scores = F.cosine_similarity(candidates, query_emb.expand_as(candidates))
            hash_similarity_time += time.time() - sim_start
            
            hash_sort_start = time.time()
            hash_result = process_scores(hash_scores, batch_candidate_urls, query_url, args.total_recall)
            hash_sorting_time += time.time() - hash_sort_start
            hash_total_time += time.time() - hash_start_time
            hash_results.append(hash_result)
            
            # --- Brute-Force Baseline ---
            base_start_time = time.time()
            
            base_sim_start = time.time()
            base_scores = F.cosine_similarity(all_code_embs, query_emb.expand_as(all_code_embs))
            base_similarity_time += time.time() - base_sim_start
            
            base_sort_start = time.time()
            base_result = process_scores(base_scores, all_urls, query_url, args.total_recall)
            base_sorting_time += time.time() - base_sort_start
            base_total_time += time.time() - base_start_time
            base_results.append(base_result)
    
    # Save query embeddings
    base_query_embs = torch.cat([model(nl_inputs=batch[1]) for batch in dataloader if model(nl_inputs=batch[1]) is not None]).detach().cpu().numpy()
    hash_query_embs = torch.cat([model.get_binary_hash(model(nl_inputs=batch[1]), is_code=False) for batch in dataloader if model(nl_inputs=batch[1]) is not None]).detach().cpu().numpy()
    np.save(embedding_dir / f"base_query_embeddings_{args.lang}.npy", base_query_embs)
    np.save(embedding_dir / f"hash_query_embeddings_{args.lang}.npy", hash_query_embs)
    logger.info(f"Saved base query embeddings to {embedding_dir / f'base_query_embeddings_{args.lang}.npy'}")
    logger.info(f"Saved hashed query embeddings to {embedding_dir / f'hash_query_embeddings_{args.lang}.npy'}")
    
    # Log timing
    logger.info(f"Base Model Similarity Time ({args.lang}): {base_similarity_time:.4f} seconds")
    logger.info(f"Base Model Sorting Time ({args.lang}): {base_sorting_time:.4f} seconds")
    logger.info(f"Base Model Total Time ({args.lang}): {base_total_time:.4f} seconds")
    logger.info(f"Hashed Model Similarity Time ({args.lang}): {hash_similarity_time:.4f} seconds")
    logger.info(f"Hashed Model Sorting Time ({args.lang}): {hash_sorting_time:.4f} seconds")
    logger.info(f"Hashed Model Total Time ({args.lang}): {hash_total_time:.4f} seconds")
    logger.info(f"Similarity Time Difference (Base - Hash) ({args.lang}): {base_similarity_time - hash_similarity_time:.4f} seconds")
    logger.info(f"Sorting Time Difference (Base - Hash) ({args.lang}): {base_sorting_time - hash_sorting_time:.4f} seconds")
    logger.info(f"Total Time Difference (Base - Hash) ({args.lang}): {base_total_time - hash_total_time:.4f} seconds")
    
    # Compute and log metrics
    def compute_metrics(results, total_recall):
        mrr = np.mean([1.0 / r['rank'] if r['rank'] <= total_recall else 0 for r in results])
        success = {k: np.mean([r[f'success@{k}'] for r in results]) for k in [1, 5, 10]}
        return {"MRR": mrr, **success}

    # Compute and log metrics
    base_metrics = compute_metrics(base_results, args.total_recall)
    hash_metrics = compute_metrics(hash_results, args.total_recall)
    
    logger.info(f"Evaluation Results (Base Model - {args.lang}):")
    for metric, value in base_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"Evaluation Results (Hash Model - {args.lang}):")
    for metric, value in hash_metrics.items():
        logger.info(f"{metric}: {value:.4f}")



def allocate_recalls(probs: torch.Tensor, total_recall: int, num_clusters: int) -> torch.Tensor:
    base_recall = total_recall - num_clusters
    allocations = torch.floor(probs * base_recall).long()
    remainder = base_recall - allocations.sum().item()
    if remainder > 0:
        _, top_indices = torch.topk(probs - (allocations/base_recall), remainder)
        allocations[top_indices] += 1
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
    ranks = [res['rank'] for res in results]
    success_at_1 = [res['success@1'] for res in results]
    success_at_5 = [res['success@5'] for res in results]
    success_at_10 = [res['success@10'] for res in results]
    
    reciprocal_ranks = [1/r if r <= total_recall else 0 for r in ranks]
    mrr = np.mean(reciprocal_ranks)
    
    return {
        'MRR': round(mrr, 4),
        'Success@1': round(np.mean(success_at_1), 4),
        'Success@5': round(np.mean(success_at_5), 4),
        'Success@10': round(np.mean(success_at_10), 4),
        'RetrievalTime': None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="java", type=str, help="Programming language (e.g., java, python)")
    parser.add_argument("--train_file", type=str, help="Override default train file path")
    parser.add_argument("--valid_file", type=str, help="Override default valid file path")
    parser.add_argument("--unif_model_dir", default="unif_model", type=str)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1.34e-4, type=float)
    parser.add_argument("--nl_length", default=128, type=int, help="Max NL sequence length after tokenization")
    parser.add_argument("--code_length", default=256, type=int, help="Max code sequence length after tokenization")
    parser.add_argument("--hash_epochs", default=30, type=int)
    parser.add_argument("--class_epochs", default=5, type=int)
    parser.add_argument("--num_clusters", default=10, type=int)
    parser.add_argument("--hash_dim", default=128, type=int)
    parser.add_argument("--beta", default=0.6, type=float)
    parser.add_argument("--eta", default=0.4, type=float)
    parser.add_argument("--mu", default=1.5, type=float)
    parser.add_argument("--lambda1", default=0.1, type=float)
    parser.add_argument("--lambda2", default=0.1, type=float)
    parser.add_argument("--total_recall", default=100, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.device = device
    
    args.train_file = args.train_file or f"dataset/{args.lang}/train.jsonl"
    args.valid_file = args.valid_file or f"dataset/{args.lang}/valid.jsonl"
    args.eval_data_file = args.valid_file or f"dataset/{args.lang}/valid.jsonl"
    args.codebase_file = args.train_file or f"dataset/{args.lang}/train.jsonl"
    
    unif_model_dir = Path(args.unif_model_dir)
    if not (unif_model_dir / args.lang).exists() or not (unif_model_dir / args.lang / f"fasttext_model_{args.lang}.bin").exists():
        logger.info(f"Training new UNIFEmbedder for {args.lang}")
        embedder = train_unif_embedder(args.train_file, args.unif_model_dir, args.lang)
    else:
        logger.info(f"Loading existing UNIFEmbedder for {args.lang}")
        embedder = UNIFEmbedder.load(unif_model_dir / args.lang, args.lang)
    
    base_model = BaseModel(embedder, device=args.device)
    base_model.lang = args.lang

    model = CoSHCModel(base_model, args.hash_dim, args.num_clusters, device=args.device)
    model = model.to(args.device)
    # model.code_hash = model.code_hash.to(args.device)
    # model.nl_hash = model.nl_hash.to(args.device)
    # model.classifier = model.classifier.to(args.device)
    logger.info("CoSHC model loaded")
    
    if args.do_train:
        train_coshc(args, model, args.train_file)
    if args.do_eval:
        evaluate_coshc(args, model, args.valid_file)
        # evaluate_coshc(args, model, args.valid_file, embedder)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total Running Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()