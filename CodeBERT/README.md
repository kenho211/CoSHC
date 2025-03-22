# Accelerated Code Search with CodeBERT + CoSHC

[![ACL 2022](https://img.shields.io/badge/ACL-2022-blue)](https://aclanthology.org/2022.acl-long.181)

Enhanced implementation of CodeBERT with CoSHC (Code Search with Deep Hashing and Classification) from ACL 2022 paper.

## Key Features ✨
- **94% faster retrieval** via hierarchical search strategy
- **7x memory reduction** using compact binary hashes
- **99%+ accuracy preservation** with re-ranking mechanism
- Dual-phase architecture: Efficient recall + Precise re-rank

## Data Preprocessing 🛠️

Modified requirements for CoSHC compatibility:
```markdown
1. Maintain consistent code-description pairs for alignment
2. Ensure code snippets can be parsed for AST generation
3. Preserve functional clusters in code corpus
4. Filter criteria (original + CoSHC additions):
   - Remove non-clusterable code snippets
   - Exclude code with ambiguous functionality
   - Ensure minimum 3 API calls per method for better clustering

## Installation ⚙️
### New Dependencies for CoSHC:
```bash
pip install scikit-learn==1.0.2       # Clustering
pip install faiss-cpu==1.7.2          # Efficient similarity search
conda install -c pytorch pytorch=1.9  # GPU-accelerated hashing
```

## Modified CodeBERT Baseline 🔧
### Architectural Changes (Sec 4.3):
- Bi-Encoder Architecture
- Training Objective using cosine simlarity based cross entropy as loss function

## CoSHC Implementation ⚡
### Architecture Overview (Paper Fig 1):

