from txtai import Embeddings

# Works with a list, dataset or generator
data = [
  "US tops 5 million confirmed virus cases",
  "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
  "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
  "The National Park Service warns against sacrificing slower friends in a bear attack",
  "Maine man wins $1M from $25 lottery ticket",
  "Make huge profits without work, earn up to $100,000 a day"
]

# Create embeddings with a graph index

embeddings = Embeddings(
  path="sentence-transformers/nli-mpnet-base-v2",
  content=True,
  functions=[
    {"name": "graph", "function": "graph.attribute"},
  ],
  expressions=[
    {"name": "category", "expression": "graph(indexid, 'category')"},
    {"name": "topic", "expression": "graph(indexid, 'topic')"},
  ],
  graph={
    "topics": {
      "categories": ["health", "climate", "finance", "world politics"]
    }
  }
)

embeddings.index(data)
embeddings.search("select topic, category, text from txtai")

