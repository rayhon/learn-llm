# ref: https://github.com/neuml/txtai/blob/master/examples/61_Integrate_txtai_with_Postgres.ipynb
# ref: https://neuml.hashnode.dev/integrate-txtai-with-postgres

from txtai import Embeddings

# URL set in code for demo purposes. Use environment variables in production.
url = "postgresql+psycopg2://postgres:mysecretpassword@localhost/postgres"


# Create embeddings
embeddings = Embeddings(
    content=url,
    backend="pgvector",
    pgvector={
        "url": url
    },
    graph={
        "backend": "rdbms",
        "url": url,
        "approximate": False,
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

# Check if embeddings are recorded
print("\nChecking embeddings in database:")
print("Total embeddings:", embeddings.count())

# Try a simple search to verify
results = embeddings.search("Roman Empire", 1)
print("\nSample search result:", results)

# Check graph nodes
print("\nChecking graph nodes:")
print("Total graph nodes:", embeddings.graph.count())


# Try progressive queries to understand the graph structure
print("\nTesting progressive graph queries:")

# First query works
g1 = embeddings.graph.search("MATCH (n) RETURN n LIMIT 5", graph=True)
print("\n1. Sample nodes:", list(g1.scan()))

# The original query
g = embeddings.graph.search("""
MATCH P=({id: "Roman Empire"})-[*1..3]->({id: "Saxons"})-[*1..3]->({id: "Vikings"})-[*1..3]->({id: "Battle of Hastings"})
RETURN P
LIMIT 20
""", graph=True)

print("\n1. Original complex path query:", list(g.scan()))

# Try a variation with only one hop between nodes
g_alt1 = embeddings.graph.search("""
MATCH P=({id: "Roman Empire"})-[*1..5]->({id: "Battle of Hastings"})
RETURN P
LIMIT 20
""", graph=True)

print("\n2. Direct path query:", list(g_alt1.scan()))

# Try with less specific matching (text data might contain the names rather than exact ID matches)
g_alt2 = embeddings.graph.search("""
MATCH P=(n1)-[*1..3]->(n2)-[*1..3]->(n3)-[*1..3]->(n4)
WHERE n1.data CONTAINS "Roman Empire" 
  AND n2.data CONTAINS "Saxon" 
  AND n3.data CONTAINS "Viking"
  AND n4.data CONTAINS "Battle of Hastings"
RETURN P
LIMIT 20
""", graph=True)

print("\n3. Content-based path query:", list(g_alt2.scan()))

# Plot the successful direct path query
if g_alt1.count() > 0:
    print("\nPlotting direct path from Roman Empire to Battle of Hastings...")
    plot(g_alt1)

# Try an alternative approach with less strict ordering
g_alt3 = embeddings.graph.search("""
MATCH P=(start)-[*1..10]->(end)
WHERE start.id = "Roman Empire" AND end.id = "Battle of Hastings"
RETURN P
LIMIT 5
""", graph=True)

print("\n4. Less restrictive path query:", list(g_alt3.scan()))

# Try looking for individual components
g_roman_to_saxon = embeddings.graph.search("""
MATCH P=({id: "Roman Empire"})-[*1..3]->({id: "Saxons"})
RETURN P
LIMIT 1
""", graph=True)

g_saxon_to_viking = embeddings.graph.search("""
MATCH P=({id: "Saxons"})-[*1..3]->({id: "Vikings"})
RETURN P
LIMIT 1
""", graph=True)

g_viking_to_hastings = embeddings.graph.search("""
MATCH P=({id: "Vikings"})-[*1..3]->({id: "Battle of Hastings"})
RETURN P
LIMIT 1
""", graph=True)

print("\n5. Individual path segments:")
print("  Roman Empire → Saxons:", list(g_roman_to_saxon.scan()))
print("  Saxons → Vikings:", list(g_saxon_to_viking.scan()))
print("  Vikings → Battle of Hastings:", list(g_viking_to_hastings.scan()))

import matplotlib.pyplot as plt
import networkx as nx

def plot(graph):
    labels = {x: f"{graph.attribute(x, 'id')}" for x in graph.scan()}
    colors = ["#D32F2F", "#0277bd", "#7e57c2", "#757575"]

    results = embeddings.batchsimilarity(labels.values(), ["Roman Empire", "Germanic Barbarians", "Viking conquest and siege", "Norman Conquest of England"])
    colors = [colors[x[0][0]] for x in results]

    options = {
        "node_size": 2000,
        "node_color": colors,
        "edge_color": "#454545",
        "font_color": "#efefef",
        "font_size": 11,
        "alpha": 1.0,
    }

    fig, ax = plt.subplots(figsize=(20, 9))
    pos = nx.spring_layout(graph.backend, seed=512, k=0.9, iterations=50)
    nx.draw_networkx(graph.backend, pos=pos, labels=labels, **options)
    ax.set_facecolor("#303030")
    ax.axis("off")
    fig.set_facecolor("#303030")

    plt.show()
    
# Only plot if we have nodes
if g.count() > 0:
    plot(g)
else:
    print("No nodes found in the graph. Try a different search query.")

embeddings.save("test")

# Query with a search string
query = str(list(embeddings.transform("Roman Empire")))
query = f"""
SELECT id, (embedding <#> '{query}') * -1 AS score, text FROM sections s \
JOIN vectors v ON s.indexid = v.indexid \
ORDER by score desc LIMIT 5
"""

#!PGPASSWORD=pass psql -h localhost -U postgres -c "{query}"

# Find top n results closest to an existing row
query = """
SELECT id, text FROM sections s \
JOIN vectors v ON s.indexid = v.indexid \
WHERE v.indexid != 738 ORDER by v.embedding <#> (SELECT embedding FROM vectors WHERE indexid=738) LIMIT 5
"""

# !PGPASSWORD=pass psql -h localhost -U postgres -c "{query}"



