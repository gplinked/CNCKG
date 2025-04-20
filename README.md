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

1. **`CNCKG/INK`** — Full implementation of:
   - Feature representation via neighborhood dictionaries  
   - Dynamic Update strategy  

2. **`CNCKG/IBIDI`** — Full implementation of:
   - IBIDI algorithm that runnning the node classification  


3. **`CNCKG/INK/ink_bennchmark/data_node_class`** — Links and loaders for open-source datasets originally released by [Ristoski et al.]:
   - AIFB  
   - MUTAG  
   - BGS  
   - AM  
   DBpedia Movies is lies in  `CNCKG/INK/ink_bennchmark/DBPedia/movies`


---

## 🚀 Getting Started

```bash
git clone https://github.com/gplinked/CNCKG.git
cd CNCKG
pip install -r requirements.txt

---

## 🔧 How to start trainning

1. Download data from links in `CNCKG/INK/ink_bennchmark/data_node_class` and `CNCKG/INK/ink_bennchmark/DBPedia/movies` and put them in corresponding folder.
2. Calculate the centrality of dataset and put it into 'CNCKG/IBIDI/dataset'
3. Run 'CNCKG/INK/ink_bennchmark/main.py'
4. Run 'CNCKG/BIDI/ComputeCr' to get results of Cr value.
5. If you want to run dynamic update algorighm, please refers to `CNCKG/INK/ink_bennchmark/INK_updated` and '`CNCKG/INK/ink_bennchmark/data_node_class/INK_updated_dbpedia`'
---
