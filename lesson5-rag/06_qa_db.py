from datasets import load_dataset
from txtai.embeddings import Embeddings

ds = load_dataset("web_questions", split="train")

for row in ds.select(range(5)):
  print(row["question"], row["answers"])
  
  

# Create embeddings index with content enabled. The default behavior is to only store indexed vectors.
embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})

# Map question to text and store content
embeddings.index([(uid, {"url": row["url"], "text": row["question"], "answer": ", ".join(row["answers"])}, None) for uid, row in enumerate(ds)])


def question(text):
  return embeddings.search(f"select text, answer, score from txtai where similar('{text}') limit 1")

question("What is the timezone of NYC?")
question("Things to do in New York")
question("Microsoft founder")
question("Apple founder university")
question("What country uses the Yen?")
question("Show me a list of Pixar movies")
question("Tell me an animal found offshore in Florida")