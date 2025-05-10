import urllib
from IPython.display import Image
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

# Get an image
request = urllib.request.urlopen("https://raw.githubusercontent.com/neuml/txtai/master/demo.gif")
# Upsert new record having both text and an object
embeddings.upsert([("txtai", {"text": "txtai executes machine-learning workflows to transform data and build AI-powered semantic search applications.", "object": request.read()}, None)])
# Query txtai for the most similar result to "machine learning" and get associated object
result = embeddings.search("select object from txtai where similar('machine learning') limit 1")[0]["object"]
# Display image
Image(result.getvalue(), width=600)