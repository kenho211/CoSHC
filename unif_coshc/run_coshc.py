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
    code_batch = [item[0] for item in batch]
    query_batch = [item[1] for item in batch]
    return code_batch, query_batch

from dataclasses import dataclass

@dataclass
class InputFeatures:
    code: list
    query: list
    url: str = ""

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        with open(file_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.examples.append({
                    "code": entry["code_tokens"],
                    "query": entry["docstring_tokens"],
                    "url": entry.get("url", f"ex_{len(self.examples)}")
                })
        self.tokenizer = tokenizer
        self.args = args
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Get tokenized examples
        code_tokens = self.examples[idx]["code"]
        query_tokens = self.examples[idx]["query"]
        
        # Convert to IDs using tokenizer
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
        
        return (
            torch.tensor(code_ids, dtype=torch.long),
            torch.tensor(query_ids, dtype=torch.long)
        )

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

# class InputFeatures(object):
#     """A single training/test features for a example."""
#     def __init__(self,
#                  code_tokens,
#                  code_ids,
#                  nl_tokens,
#                  nl_ids,
#                  url,

#     ):
#         self.code_tokens = code_tokens
#         self.code_ids = code_ids
#         self.nl_tokens = nl_tokens
#         self.nl_ids = nl_ids
#         self.url=url

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


# Model Definitions
# class BaseModel(torch.nn.Module):
#     def __init__(self, unif_embedder):
#         super(BaseModel, self).__init__()
#         self.embedder = unif_embedder
    
#     def forward(self, code_inputs=None, nl_inputs=None):
#         device = 'mps' if torch.backends.mps.is_available() else 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
#         if code_inputs is not None:
#             if not code_inputs:
#                 logger.error("code_inputs is empty")
#                 return None
#             snippets = [{"code": code, "language": "java"} for code in code_inputs]
#             embeddings = self.embedder.embed_snippets(snippets)
#             if embeddings is None or embeddings.size == 0:
#                 logger.error("Failed to generate code embeddings")
#                 return None
#             return torch.tensor(embeddings, dtype=torch.float32).to(device)
#         elif nl_inputs is not None:
#             if not nl_inputs:
#                 logger.error("nl_inputs is empty")
#                 return None
#             embeddings = self.embedder.embed_queries(nl_inputs)
#             if embeddings is None or embeddings.size == 0:
#                 logger.error("Failed to generate query embeddings")
#                 return None
#             return torch.tensor(embeddings, dtype=torch.float32).to(device)
#         logger.error("No valid inputs provided")
#         return None

class BaseModel(torch.nn.Module):
    def __init__(self, embedder, tokenizer=None, device=None):
        super().__init__()
        self.embedder = embedder
        self.tokenizer = tokenizer if tokenizer is not None else DummyTokenizer()
        self.device = device if device is not None else torch.device("cpu")
    
    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            if isinstance(code_inputs, torch.Tensor):
                if code_inputs.dim() == 1:
                    code_inputs = code_inputs.unsqueeze(0)
                if code_inputs.nelement() == 0:
                    return None
                code_inputs = [self.tokenizer.decode(ids.tolist()).split() for ids in code_inputs]
            
            if not code_inputs:
                return None
                
            snippets = [{"code": code, "language": "java"} for code in code_inputs]
            embeddings = self.embedder.embed_snippets(snippets)
            if embeddings is None:
                return None
            return torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            
        elif nl_inputs is not None:
            if isinstance(nl_inputs, torch.Tensor):
                if nl_inputs.dim() == 1:
                    nl_inputs = nl_inputs.unsqueeze(0)
                if nl_inputs.nelement() == 0:
                    return None
                nl_inputs = [self.tokenizer.decode(ids.tolist()).split() for ids in nl_inputs]
            
            if not nl_inputs:
                return None
                
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
        self.code_hash = torch.nn.Linear(embed_dim, hash_dim)
        self.nl_hash = torch.nn.Linear(embed_dim, hash_dim)
        self.alpha = 1.0
        self.classifier = torch.nn.Linear(embed_dim, num_clusters)
    
    def get_binary_hash(self, embeddings, is_code=True):
        if embeddings is None:
            return None
            
        if isinstance(embeddings, tuple):
            # Take the first element if it's a tuple
            embeddings = embeddings[0]
            
        hash_layer = self.code_hash if is_code else self.nl_hash
        logits = hash_layer(embeddings)
        return torch.tanh(self.alpha * logits)
    
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

# CoSHC Training
def train_coshc(args, model, train_file):
    dataset = TextDataset(train_file)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=4, collate_fn=custom_collate_fn)
    
    base_query_embs, base_code_embs = [], []
    hash_query_embs, hash_code_embs = [], []
    
    embedding_dir = Path("embeddings/train") / args.lang
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if embeddings already exist
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
        # Generate all embeddings during training
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
    
        # Concatenate and save embeddings
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
    
    # Hashing Training (No Early Stopping)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.hash_epochs):
        total_train_loss = 0
        for batch in dataloader:
            code_embs = model(code_inputs=batch[0])
            nl_embs = model(nl_inputs=batch[1])
            if code_embs is None or nl_embs is None:
                continue
            S_target = compute_similarity_matrix(code_embs, nl_embs, args.beta, args.eta, args.train_batch_size)
            B_code = model.get_binary_hash(code_embs, is_code=True)
            B_nl = model.get_binary_hash(nl_embs, is_code=False)
            loss = hashing_loss(B_code, B_nl, S_target, args.mu, args.lambda1, args.lambda2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        logger.info(f"Hashing Epoch {epoch}, Train Loss: {total_train_loss / len(dataloader):.4f}")
        model.alpha += 1.0

# # Evaluation with Retrieval Time Logging
# def evaluate_coshc(args, model, valid_file):
#     dataset = TextDataset(valid_file)
#     dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=custom_collate_fn)
    
#     base_query_embs, base_code_embs = [], []
#     hash_query_embs, hash_code_embs = [], []
#     device = args.device
    
#     for batch in dataloader:
#         q_embs = model(nl_inputs=batch[1])
#         c_embs = model(code_inputs=batch[0])
#         if q_embs is not None and c_embs is not None:
#             base_query_embs.append(q_embs.cpu())
#             base_code_embs.append(c_embs.cpu())
#             hash_q_embs = model.get_binary_hash(q_embs, is_code=False).cpu()
#             hash_c_embs = model.get_binary_hash(c_embs, is_code=True).cpu()
#             hash_query_embs.append(hash_q_embs)
#             hash_code_embs.append(hash_c_embs)
    
#     if not base_query_embs or not base_code_embs:
#         logger.error("No valid embeddings for evaluation")
#         return
    
#     base_query_embs = torch.cat(base_query_embs).detach().numpy()
#     base_code_embs = torch.cat(base_code_embs).detach().numpy()
#     hash_query_embs = torch.cat(hash_query_embs).detach().numpy()
#     hash_code_embs = torch.cat(hash_code_embs).detach().numpy()
    
#     embedding_dir = Path("embeddings/valid") / args.lang
#     embedding_dir.mkdir(parents=True, exist_ok=True)
#     np.save(embedding_dir / f"base_query_embeddings_{args.lang}.npy", base_query_embs)
#     np.save(embedding_dir / f"base_code_embeddings_{args.lang}.npy", base_code_embs)
#     np.save(embedding_dir / f"hash_query_embeddings_{args.lang}.npy", hash_query_embs)
#     np.save(embedding_dir / f"hash_code_embeddings_{args.lang}.npy", hash_code_embs)
#     logger.info(f"Saved base query embeddings to {embedding_dir / f'base_query_embeddings_{args.lang}.npy'}")
#     logger.info(f"Saved base code embeddings to {embedding_dir / f'base_code_embeddings_{args.lang}.npy'}")
#     logger.info(f"Saved hashed query embeddings to {embedding_dir / f'hash_query_embeddings_{args.lang}.npy'}")
#     logger.info(f"Saved hashed code embeddings to {embedding_dir / f'hash_code_embeddings_{args.lang}.npy'}")
    
#     base_query_embs_norm = base_query_embs / np.linalg.norm(base_query_embs, axis=1, keepdims=True)
#     base_code_embs_norm = base_code_embs / np.linalg.norm(base_code_embs, axis=1, keepdims=True)
#     hash_query_embs_norm = hash_query_embs / np.linalg.norm(hash_query_embs, axis=1, keepdims=True)
#     hash_code_embs_norm = hash_code_embs / np.linalg.norm(hash_code_embs, axis=1, keepdims=True)
    
#     base_start_time = time.time()
#     base_similarity = base_query_embs_norm @ base_code_embs_norm.T
#     base_ranks = np.argsort(-base_similarity, axis=1)
#     base_retrieval_time = time.time() - base_start_time
    
#     hash_start_time = time.time()
#     hash_similarity = hash_query_embs_norm @ hash_code_embs_norm.T
#     hash_ranks = np.argsort(-hash_similarity, axis=1)
#     hash_retrieval_time = time.time() - hash_start_time
    
#     logger.info(f"Base Model Retrieval Time ({args.lang}): {base_retrieval_time:.4f} seconds")
#     logger.info(f"Hashed Model Retrieval Time ({args.lang}): {hash_retrieval_time:.4f} seconds")
#     logger.info(f"Time Difference (Base - Hash) ({args.lang}): {base_retrieval_time - hash_retrieval_time:.4f} seconds")
    
#     def evaluate_topk(similarity_matrix, top_k_values=[1, 5, 10]):
#         results = {f"Success@{k}": 0 for k in top_k_values}
#         reciprocal_ranks = []
#         for query_idx in range(similarity_matrix.shape[0]):
#             scores = similarity_matrix[query_idx]
#             sorted_indices = np.argsort(-scores)
#             rank = np.where(sorted_indices == query_idx)[0][0] + 1
#             for k in top_k_values:
#                 if rank <= k:
#                     results[f"Success@{k}"] += 1
#             reciprocal_ranks.append(1 / rank)
#         num_queries = similarity_matrix.shape[0]
#         for k in top_k_values:
#             results[f"Success@{k}"] /= num_queries
#         results["MRR"] = np.mean(reciprocal_ranks)
#         return results
    
#     metrics = evaluate_topk(base_similarity)
#     logger.info(f"Evaluation Results (Base Model - {args.lang}):")
#     for metric, value in metrics.items():
#         logger.info(f"{metric}: {value:.4f}")
    
#     metrics = evaluate_topk(hash_similarity)
#     logger.info(f"Evaluation Results (Hash Model - {args.lang}):")
#     for metric, value in metrics.items():
#         logger.info(f"{metric}: {value:.4f}")
def collate_fn(batch):
    code_batch, query_batch = zip(*batch)
    
    # Pad sequences to max length in batch
    code_padded = pad_sequence(
        code_batch, 
        batch_first=True, 
        padding_value=0  # Using 0 as pad_id
    )
    query_padded = pad_sequence(
        query_batch,
        batch_first=True,
        padding_value=0
    )
    
    return code_padded, query_padded

def evaluate_coshc(args, model, valid_file, tokenizer, embedder):
    """Two-stage evaluation: Hash recall + Re-rank, adapted for tokenized string data"""
    query_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    query_urls = [example['url'] for example in query_dataset.examples]
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn, shuffle=False)
    
    code_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    code_urls = [example['url']for example in code_dataset.examples]
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn, shuffle=False)

    
    all_code_embs = []
    all_code_hashes = []
    all_code_clusters = []

    model.eval()
    logger.info("Precomputing code representations")
    for idx, batch in enumerate(code_dataloader):
        logger.info(f"Precomputing code representations batch {idx} of {len(code_dataloader)}")
        logger.debug(f"Batch[0] type: {type(batch[0])}, Batch[0]: {batch[0][:2] if isinstance(batch[0], (list, torch.Tensor)) else batch[0]}")
        
        print(batch[0])
        code_inputs = batch[0].to(args.device, non_blocking=True)
        
        with torch.no_grad():
            code_embs = model(code_inputs=code_inputs)
        code_hashes = model.get_binary_hash(code_embs, is_code=True)
        code_clusters = model.classifier(code_embs)
        all_code_embs.append(code_embs)
        all_code_hashes.append(code_hashes)
        all_code_clusters.append(code_clusters)
    
    all_code_embs = torch.cat(all_code_embs)
    all_code_hashes = torch.cat(all_code_hashes)
    all_code_clusters = torch.cat(all_code_clusters)
    logger.info("Precomputing code representations completed")

    hash_total_time = 0
    hash_similarity_time = 0
    hash_sorting_time = 0
    brute_total_time = 0
    brute_similarity_time = 0
    brute_sorting_time = 0

    hash_results = []
    brute_results = []

    # results = []
    # similarity_time = 0
    # sorting_time = 0

    logger.info("Processing queries")
    for query_index, query_batch in enumerate(query_dataloader):
        logger.info(f"Processing query batch {query_index} of {len(query_dataloader)}")
        logger.debug(f"Batch[1] type: {type(query_batch[1])}, Batch[1]: {query_batch[1][:2] if isinstance(query_batch[1], (list, torch.Tensor)) else query_batch[1]}")
        
        if isinstance(query_batch[1], torch.Tensor):
            nl_inputs = query_batch[1].to(args.device, non_blocking=True)
        elif isinstance(query_batch[1], list):
            if query_batch[1] and isinstance(query_batch[1][0], str):
                nl_sequences = [embedder.encoder.encode(seq) for seq in query_batch[1]]
            elif query_batch[1] and isinstance(query_batch[1][0], list):
                nl_sequences = [embedder.encoder.encode(' '.join(seq)) for seq in query_batch[1]]
            else:
                raise ValueError(f"Unexpected batch[1] content: {query_batch[1][:2]}")
            nl_inputs = pad_sequence(
                [torch.tensor(seq, dtype=torch.long) for seq in nl_sequences],
                batch_first=True,
                padding_value=0
            ).to(args.device, non_blocking=True)
        else:
            raise ValueError(f"Unexpected type for batch[1]: {type(query_batch[1])}")
        
        with torch.no_grad():
            nl_embs = model(nl_inputs=nl_inputs)
        
        nl_hashes = model.get_binary_hash(nl_embs, is_code=False)
        probs = torch.softmax(model.classifier(nl_embs), dim=1)
        
        start_idx = query_index * args.eval_batch_size
        end_idx = min(start_idx + len(nl_inputs), len(query_urls))
        query_batch_urls = query_urls[start_idx:end_idx]
        
        for i in range(len(nl_embs)):
            query_url = query_batch_urls[i]
            dists = (all_code_hashes != nl_hashes[i]).sum(dim=1)
            recall_counts = allocate_recalls(probs[i], args.total_recall, args.num_clusters)
            
            candidates = []
            candidate_indices = []
            
            for cluster_id, count in enumerate(recall_counts):
                mask = (torch.argmax(all_code_clusters, dim=1) == cluster_id)
                cluster_dists = dists[mask]
                cluster_indices = cluster_dists.topk(min(count, len(cluster_dists)), largest=False).indices
                original_indices = torch.where(mask)[0][cluster_indices]
                candidate_indices.extend(original_indices.tolist())
                candidates.append(all_code_embs[mask][cluster_indices])
            
            candidates = torch.cat(candidates)
            batch_candidate_urls = [code_urls[idx] for idx in candidate_indices]
            logger.debug(f"Query URL: {query_url}, Candidates include correct: {query_url in batch_candidate_urls}, Num candidates: {len(batch_candidate_urls)}")
            
            hash_start_time = time.time()
            hash_sim_start = time.time()
            hash_scores = F.cosine_similarity(candidates, nl_embs[i].unsqueeze(0).expand_as(candidates))
            hash_similarity_time += time.time() - hash_sim_start
            
            hash_result = process_scores(hash_scores, batch_candidate_urls, query_url, args.total_recall)
            hash_sorting_time += hash_result['sorting_time']
            hash_results.append(hash_result)
            hash_total_time += time.time() - hash_start_time

            # --- Without Hashing (Brute Force) ---
            brute_start_time = time.time()
            brute_sim_start = time.time()
            brute_scores = F.cosine_similarity(all_code_embs, nl_embs[i].unsqueeze(0).expand_as(all_code_embs))
            brute_similarity_time += time.time() - brute_sim_start
            
            brute_candidate_urls = code_urls  # All codes are candidates
            brute_result = process_scores(brute_scores, brute_candidate_urls, query_url, args.total_recall)
            brute_sorting_time += brute_result['sorting_time']
            brute_results.append(brute_result)
            brute_total_time += time.time() - brute_start_time

    # Compute metrics for both approaches
    hash_metrics = compute_metrics(hash_results, args.total_recall)
    hash_metrics["TotalTime"] = hash_total_time
    hash_metrics["SimilarityTime"] = hash_similarity_time
    hash_metrics["SortingTime"] = hash_sorting_time
    
    brute_metrics = compute_metrics(brute_results, args.total_recall)
    brute_metrics["TotalTime"] = brute_total_time
    brute_metrics["SimilarityTime"] = brute_similarity_time
    brute_metrics["SortingTime"] = brute_sorting_time

    # Log results
    logger.info(f"Evaluation Results with Hashing for {args.lang}:")
    for metric, value in hash_metrics.items():
        logger.info(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
    
    logger.info(f"Evaluation Results without Hashing (Brute Force) for {args.lang}:")
    for metric, value in brute_metrics.items():
        logger.info(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
    
    logger.info(f"Time Efficiency Comparison for {args.lang}:")
    logger.info(f"Hashing Total Time: {hash_total_time:.4f}s, Brute Force Total Time: {brute_total_time:.4f}s")
    logger.info(f"Time Difference (Brute - Hash): {brute_total_time - hash_total_time:.4f}s")
    logger.info(f"Similarity Time Difference (Brute - Hash): {brute_similarity_time - hash_similarity_time:.4f}s")
    logger.info(f"Sorting Time Difference (Brute - Hash): {brute_sorting_time - hash_sorting_time:.4f}s")

    return hash_metrics, brute_metrics


def allocate_recalls(probs: torch.Tensor, total_recall: int, num_clusters: int) -> torch.Tensor:
    base_recall = total_recall - num_clusters
    allocations = torch.floor(probs * base_recall).long()
    remainder = base_recall - allocations.sum().item()
    if remainder > 0:
        _, top_indices = torch.topk(probs - (allocations/base_recall), remainder)
        allocations[top_indices] += 1
    return allocations + 1

def process_scores(scores: torch.Tensor, candidate_urls: list, query_url: str, total_recall: int) -> dict:
    start_time = time.time()
    sorted_indices = torch.argsort(scores, descending=True)
    sorting_time = time.time() - start_time
    sorted_urls = [candidate_urls[i] for i in sorted_indices.cpu().numpy()]
    try:
        rank = sorted_urls.index(query_url) + 1
    except ValueError:
        rank = total_recall + 1
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

# Main Execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="java", type=str, help="Programming language (e.g., java, python)")
    parser.add_argument("--train_file", type=str, help="Override default train file path")
    parser.add_argument("--valid_file", type=str, help="Override default valid file path")
    parser.add_argument("--unif_model_dir", default="unif_model", type=str)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--nl_length", default=128, type=int, help="Max NL sequence length after tokenization")
    parser.add_argument("--code_length", default=256, type=int, help="Max code sequence length after tokenization")
    parser.add_argument("--hash_epochs", default=10, type=int)
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
        embedder = train_unif_embedder(
            args.train_file, args.unif_model_dir, args.lang, 
            only_base=args.only_base, batch_size=args.train_batch_size, 
            ft_epochs=args.ft_epochs, sim_epochs=args.sim_epochs
        )
    else:
        logger.info(f"Loading existing UNIFEmbedder for {args.lang}")
        embedder = UNIFEmbedder.load(unif_model_dir / args.lang, args.lang)
    
    tokenizer = DummyTokenizer()

    base_model = BaseModel(embedder, tokenizer, device = args.device)
    base_model.lang = args.lang

    model = CoSHCModel(base_model, args.hash_dim, args.num_clusters, device = args.device)
    model = model.to(args.device)
    logger.info("CoSHC model loaded")
    
    if args.do_train:
        train_coshc(args, model, args.train_file)
    if args.do_eval:
        evaluate_coshc(args, model, args.valid_file, tokenizer, embedder)  # Pass embedder
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total Running Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()