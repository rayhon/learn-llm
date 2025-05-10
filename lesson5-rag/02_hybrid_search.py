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

# Create an embeddings
embeddings = Embeddings(
  hybrid=True,
  path="sentence-transformers/nli-mpnet-base-v2"
)

# Create an index for the list of text
embeddings.index(data)
print("%-20s %s" % ("Query", "Best Match"))
print("-" * 50)
# Run an embeddings search for each query
for query in ("feel good story", "climate change", 
    "public health story", "war", "wildlife", "asia",
    "lucky", "dishonest junk"):
  # Extract uid of first result
  # search result format: (uid, score)
  uid = embeddings.search(query, 1)[0][0]
  # Print text
  print("%-20s %s" % (query, data[uid]))