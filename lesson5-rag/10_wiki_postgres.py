from txtai import Embeddings

# URL set in code for demo purposes. Use environment variables in production.
url = "postgresql+psycopg2://postgres:mysecretpassword@localhost/postgres"

# Create embeddings
embeddings = Embeddings(
    content=url,
    backend="pgvector",
    pgvector={
        "url": url
    }
)

# Load dataset
wikipedia = Embeddings()
wikipedia.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")

query = """
SELECT id, text FROM txtai
order by percentile desc
LIMIT 100000
"""

# Index dataset
embeddings.index(wikipedia.search(query))

embeddings.search("Tell me about a mythical horse", 1)

embeddings.search("What is the main ingredient in Ketchup?", 1)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Commit results to the database
embeddings.save("test")
embeddings.close()

