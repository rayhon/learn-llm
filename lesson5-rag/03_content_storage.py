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

# Create embeddings with content enabled.
# The default behavior is to only store indexed vectors.
embeddings = Embeddings(
  path="sentence-transformers/nli-mpnet-base-v2",
  content=True,
  objects=True
)

# Create an index for the list of text
embeddings.index(data)
print(embeddings.search("feel good story", 1)[0]["text"])


# Create an index for the list of text
embeddings.index([{"text": text, "length": len(text)} for text in data])

# Filter by score
print(embeddings.search("select text, score from txtai where similar('hiking danger') and score >= 0.15"))

# Filter by metadata field 'length'
print(embeddings.search("select text, length, score from txtai where similar('feel good story') and score >= 0.05 and length >= 40"))

# Run aggregate queries
print(embeddings.search("select count(*), min(length), max(length), sum(length) from txtai"))