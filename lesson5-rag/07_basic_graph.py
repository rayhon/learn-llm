import networkx as nx

from txtai.graph import GraphFactory

# Create graph
graph = GraphFactory.create({"backend": "networkx"})
graph.initialize()

# Add nodes
nodes = [(0, "dog"), (1, "fox"), (2, "wolf"), (3, "zebra"), (4, "horse")]
labels = {uid:text for uid, text in nodes}
for uid, text in nodes:
  graph.addnode(uid, text=text)

# Add relationships
edges = [(0, 1, 1), (0, 2, 1), (1, 2, 1), (2, 3, 0.25), (3, 4, 1)]
for source, target, weight in edges:
  graph.addedge(source, target, weight=weight)

# Print centrality and path between 0 and 4
print("Centrality:", {labels[k]:v for k, v in graph.centrality().items()})
print("Path (dog->horse):", " -> ".join([labels[uid] for uid in graph.showpath(0, 4)]))

# Visualize graph
nx.draw(graph.backend, nx.shell_layout(graph.backend), labels=labels, with_labels=True,
        node_size=2000, node_color="#03a9f4", edge_color="#cfcfcf", font_color="#fff")