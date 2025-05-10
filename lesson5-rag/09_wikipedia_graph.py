from txtai import Embeddings

# Load dataset
wikipedia = Embeddings()
wikipedia.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")

query = """
SELECT id, text FROM txtai WHERE similar('Viking raids in France') and percentile >= 0.5
"""

results = wikipedia.search(query, 5)

for x in results:
    print(x)
    

import json
from txtai import LLM
llm = LLM("TheBloke/Mistral-7B-OpenOrca-GGUF")

data = []
for result in results:
    prompt = f"""<|im_start|>system
    You are a friendly assistant. You answer questions from users.<|im_end|>
    <|im_start|>user
    Extract an entity relationship graph from the following text. Output as JSON

    Nodes must have label and type attributes. Edges must have source, target and relationship attributes.

    text: {result['text']} <|im_end|>
    <|im_start|>assistant
    """

    try:
        data.append(json.loads(llm(prompt, maxlength=4096)))
    except:
        pass

print(data)

    

def stream():
    nodes = {}

    for row in data.copy():
        # Create nodes
        for node in row["nodes"]:
            if node["label"] not in nodes:
                node["id"] = len(nodes)
                node["data"] = [node["label"]]
                nodes[node["label"]] = node

        for edge in row["edges"]:
            source = nodes.get(edge["source"])
            target = nodes.get(edge["target"])

            if source and target:
                if "relationships" not in source:
                    source["relationships"] = []

                source["relationships"].append({"id": target["id"], "relationship": edge["relationship"]})

    return nodes.values()

# Create embeddings instance with a semantic graph
embeddings = Embeddings(
    autoid = "uuid5",
    path = "intfloat/e5-base",
    instructions = {
        "query": "query: ",
        "data": "passage: "
    },
    columns = {
        "text": "label"
    },
    content = True,
    graph = {
        "approximate": False,
        "topics": {}
    }
)

embeddings.index(stream())


import matplotlib.pyplot as plt
import networkx as nx

def plot(graph):
    labels = {x: f"{graph.attribute(x, 'text')} ({x})" for x in graph.scan()}
    lookup = {
      "Person": "#d32f2f",
      "Location": "#0277bd",
      "Event": "#e64980",
      "Role": "#757575"
    }

    colors = []
    for x in graph.scan():
      value = embeddings.search("select type from txtai where id = :x", parameters={"x": x})[0]["type"]
      colors.append(lookup.get(value, "#7e57c2"))

    options = {
        "node_size": 2000,
        "node_color": colors,
        "edge_color": "#454545",
        "font_color": "#efefef",
        "font_size": 11,
        "alpha": 1.0
    }

    fig, ax = plt.subplots(figsize=(20, 9))
    pos = nx.spring_layout(graph.backend, seed=0, k=3, iterations=250)
    nx.draw_networkx(graph.backend, pos=pos, labels=labels, **options)
    ax.set_facecolor("#303030")
    ax.axis("off")
    fig.set_facecolor("#303030")

plot(embeddings.graph)