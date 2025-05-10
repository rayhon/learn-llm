# Topic Modeling
Topic modeling is an unsupervised method to identify abstract topics within a dataset. The most common way to do topic modeling is to use clustering algorithms to group nodes with the closest proximity.

A number of excellent topic modeling libraries exist in Python today. BERTopic and Top2Vec are two of the most popular. Both use:
* sentence-transformers to encode data into vectors
* UMAP for dimensionality reduction and 
* HDBSCAN to cluster nodes.

Given that an embeddings index has already encoded and indexed data, we'll take a different approach. txtai builds a graph running a query for each node against the index. In addition to topic modeling, this also opens up much more functionality which will be covered later.

Topic modeling in txtai is done using **community detection algorithms**. Similar nodes are group together. There are settings to control how much granularity is used to group nodes. In other words, topics can be very specific or broad, depending on these settings. Topics are labeled by building a BM25 index over each topic and finding the most common terms associated with the topic.


# References
* https://neuml.hashnode.dev/introducing-the-semantic-graph
* https://neuml.github.io/txtai/pipeline/text/llm/

## txtai + Elasticsearch
https://github.com/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb

## Installation

1. Create and activate the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Upgrade pip, setuptools, and wheel:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```
3. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install `autoawq` without dependencies (to skip `triton`):
   ```bash
   pip install autoawq --no-deps
   ```