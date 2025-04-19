# CNCKG: A Credible Node Classification Method for Knowledge Graph

This repository provides a reference implementation of **CNCKG (Credible Node Classification for Knowledge Graphs)**, a novel method that combines interpretable feature representation with classification difficulty modeling. CNCKG is designed to enhance both accuracy and credibility in node classification tasks on Knowledge Graphs (KGs).

> **Paper**: *CNCKG: a Credible Node Classification Method for Knowledge Graph*  

---

## 🌟 Highlights

- ✅ Interpretable node feature representation strategy  
- 🔍 Incorporation of classification difficulty via credibility theory  
- 🔁 Dynamic update strategy for evolving KGs  
- 🔧 Progressive learning model (IBIDI) from easy to hard  
- 📊 Comprehensive performance across 5 public KGs  

---

## 📁 Repository Contents

We thank the reviewers for their suggestions. To ensure the **reproducibility** of our research, we publicly share the following in this repository:

1. **`src/`** — Full implementation of:
   - Feature representation via neighborhood dictionaries  
   - Node classification using the IBIDI progressive learning strategy  
   - Inference pipelines on benchmark datasets  

2. **`docs/`** — Documentation including:
   - Setup and environment instructions  
   - Hyperparameter configurations  
   - Evaluation metrics scripts (e.g., accuracy and Cr)

3. **`data/`** — Links and loaders for open-source datasets originally released by [Ristoski et al.](https://):
   - AIFB  
   - MUTAG  
   - BGS  
   - AM  
   - DBpedia Movies  


---

## 🚀 Getting Started

```bash
git clone https://github.com/gplinked/CNCKG.git
cd CNCKG
pip install -r requirements.txt
python run_experiments.py --dataset AIFB

---

## 🔧 Note: We are currently organizing and refining the codebase.
The complete reproducible code will be made available by April 20, 2025.

---
