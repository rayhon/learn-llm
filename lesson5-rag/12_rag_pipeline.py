from txtai import RAG
from txtai import Embeddings, LLM

# Load Wikipedia Embeddings database
embeddings = Embeddings()
embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")

# Create LLM
llm = LLM("TheBloke/Mistral-7B-OpenOrca-AWQ")

# llm([
#     {"role": "system", "content": "You are a friendly assistant. You answer questions from users."},
#     {"role": "user", "content": f"""
#         Answer the following question using only the context below. Only include information specifically discussed.

#         question: {question}
#         context: {text} 
#     """}
# ])

# Create RAG pipeline using existing components. LLM parameter can also be a model path.
rag = RAG(embeddings, llm, template=prompt)