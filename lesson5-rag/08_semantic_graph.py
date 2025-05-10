from datasets import load_dataset

from txtai.embeddings import Embeddings

# Create embeddings instance with a semantic graph
embeddings = Embeddings({
  "path": "sentence-transformers/all-MiniLM-L6-v2",
  "content": True,
  "functions": [
    {"name": "graph", "function": "graph.attribute"},
  ],
  "expressions": [
      {"name": "category", "expression": "graph(indexid, 'category')"},
      {"name": "topic", "expression": "graph(indexid, 'topic')"},
      {"name": "topicrank", "expression": "graph(indexid, 'topicrank')"}
  ],
  "graph": {
      "limit": 15,
      "minscore": 0.1,
      "topics": {
          "categories": ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports",
                         "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
      }
  }
})

# Load dataset
dataset = load_dataset("ag_news", split="train")
rows = dataset["text"]

# Index dataset
embeddings.index((x, text, None) for x, text in enumerate(rows))

# Store reference to graph
graph = embeddings.graph
len(embeddings.graph.topics)
list(graph.topics.keys())[:5]


print(embeddings.search("select text from txtai where topic = 'sox_red_boston_series' and topicrank = 0", 1)[0]["text"])

for x, topic in enumerate(list(graph.topics.keys())[:5]):
  print(graph.categories[x], topic)
  
print(embeddings.search("select text from txtai where similar('book')", 1)[0]["text"])
print(embeddings.search("select text from txtai where category='Sports' and similar('book')", 1)[0]["text"])


centrality = graph.centrality()

topics = list(graph.topics.keys())

for uid in list(centrality.keys())[:5]:
  topic = graph.attribute(uid, "topic")
  print(f"{topic} ({topics.index(topic)})")
