import json
import sys
import torch
import numpy as np
import pygmtools as pygm
import os
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
import functools


synthetic_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"situational_graphs_datasets/src", "graph_datasets")
sys.path.append(synthetic_dataset_dir)

from SyntheticDatasetGenerator import SyntheticDatasetGenerator
import graph_visualizer as gv
with open(os.path.join(os.path.dirname(synthetic_dataset_dir),"graph_datasets/config", "graph_matching.json")) as f:
    synteticdataset_settings = json.load(f)

# Set PyGmTool backend
pygm.set_backend('pytorch')

# One-hot encoding for node types
node_type_mapping = {"room": [1, 0, 0], "wall": [0, 1, 0], "ws": [0, 0, 1]}

# One-hot encoding for edge types
edge_type_mapping = {
    "ws_belongs_room": [1, 0, 0, 0],
    "ws_belongs_wall": [0, 1, 0, 0],
    "ws_same_room": [0, 0, 1, 0],
    "ws_same_wall": [0, 0, 0, 1]
}

# Extract node and edge features
def extract_features(graph):
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    node_features = []
    for node in nodes:
        attrs = graph.nodes[node]
        
        # One-hot encode node type
        node_type = node_type_mapping.get(attrs.get("type", "room"), [0, 0, 1])  # Default to "ws"

        # Extract 'center' (always present, default [0,0,0])
        center = np.array(attrs.get("center", [0, 0, 0]), dtype=np.float32)

        # Extract 'normal' (only for 'ws', otherwise [0,0,0])
        normal = np.array(attrs.get("normal", [0, 0, 0] if attrs.get("type") != "ws" else [0, 0, 0]), dtype=np.float32)

        # Concatenate center and normal to form Geometric_info (1x6)
        geometric_info = np.concatenate((center, normal))

        # Concatenate one-hot encoding with geometric info
        node_feature_vector = np.concatenate((node_type, geometric_info))
        node_features.append(node_feature_vector)

    edge_features = []

    (6, 0, {'type': 'ws_belongs_room', 'x': [], 'viz_feat': 'red', 'linewidth': 1.0, 'alpha': 0.5})

    # Extract Edge Features
    edge_features = []
    for edge in edges:
        start_node, end_node = edge  # Unpack edge tuple
        attrs = graph.edges[edge]

        # One-hot encode edge type
        edge_type = edge_type_mapping.get(attrs.get("type", "ws_belongs_room"), [1, 0, 0, 0])  # Default "ws_belongs_room"

        # Combine [start node, end node] with one-hot encoded type
        edge_feature_vector = np.concatenate(([start_node, end_node], edge_type))
        edge_features.append(edge_feature_vector)

    return np.array(node_features, dtype=np.float32), np.array(edge_features, dtype=np.float32)


def normalize_features(node_features, edge_features):
    """
    Normalizes node and edge features:
    - One-hot features (categorical) remain unchanged.
    - Spatial coordinates (center) are standardized (Z-score normalization).
    - Directional vectors (normal) are L2 normalized.
    - Edge feature indices (start_node, end_node) remain unchanged.

    Parameters:
        node_features (np.array): Nx9 node feature matrix [one-hot (3) + center (3) + normal (3)].
        edge_features (np.array): Mx6 edge feature matrix [start_node, end_node + one-hot (4)].

    Returns:
        normalized_node_features (np.array): Normalized node features.
        normalized_edge_features (np.array): Normalized edge features.
    """
    # Extract different feature types
    one_hot_nodes = node_features[:, :3]  # First 3 columns (One-hot encoding)
    spatial_coords = node_features[:, 3:5]  # Spatial coordinates (center x, y)
    direction_vectors = node_features[:, 5:]  # Normal vectors

    # Normalize spatial coordinates (Z-score standardization)
    scaler = StandardScaler()
    spatial_coords_norm = scaler.fit_transform(spatial_coords)

    # Normalize direction vectors (L2 normalization)
    #direction_vectors_norm = normalize(direction_vectors, norm='l2')  # Normalize each row
    #not needed since the normal vector is already normalized
    direction_vectors_norm = direction_vectors
    
    # Reconstruct normalized node features
    normalized_node_features = np.hstack([one_hot_nodes, spatial_coords_norm, direction_vectors_norm])

    # Edge Features Normalization
    start_end_nodes = edge_features[:, :2]  # Keep indices as-is
    one_hot_edges = edge_features[:, 2:]  # Edge type one-hot encoding (4 values)

    # Reconstruct normalized edge features
    normalized_edge_features = np.hstack([start_end_nodes, one_hot_edges])

    return normalized_node_features, normalized_edge_features

def compute_affinity_matrix(g1_node_feat, g2_node_feat, g1_edge_feat, g2_edge_feat):

    # Suppose these are shape [n1, d] in NumPy. Add batch dim for PyTorch: [1, n1, d]
    g1_node_feat_torch = torch.tensor(g1_node_feat, dtype=torch.float32).unsqueeze(0)
    g2_node_feat_torch = torch.tensor(g2_node_feat, dtype=torch.float32).unsqueeze(0)

    # Edge features: also add batch dim if needed
    g1_edge_feat_torch = torch.tensor(g1_edge_feat, dtype=torch.float32).unsqueeze(0)
    g2_edge_feat_torch = torch.tensor(g2_edge_feat, dtype=torch.float32).unsqueeze(0)

    # Convert to sparse format (batch mode)
    # Check doc: some versions of pygm.utils.dense_to_sparse support batch; 
    # others require manual looping. If not, do it unbatched (depends on PyGMTools version).
    conn1, edge1 = pygm.utils.dense_to_sparse(g1_edge_feat_torch)  # shape [1, 2, E], [1, E]
    conn2, edge2 = pygm.utils.dense_to_sparse(g2_edge_feat_torch)

    # Now pass 1D n1, n2 => batch=1
    n1 = torch.tensor([g1_node_feat.shape[0]])  # e.g. [16]
    n2 = torch.tensor([g2_node_feat.shape[0]])  # e.g. [12]

    def node_aff_fn(X, Y):
        """
        X, Y: shape [B, n, d], [B, m, d]
        We want [B, n, m].
        """
        # Use batched matmul: shape [B, n, m]
        return torch.bmm(X, Y.transpose(1, 2))

    # Edge affinity
    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.0)

    # Build the (batched) affinity matrix
    K = pygm.utils.build_aff_mat(
        g1_node_feat_torch, edge1, conn1,
        g2_node_feat_torch, edge2, conn2,
        n1, n2,  # shape [1], matches batch=1
        edge_aff_fn=gaussian_aff,
        node_aff_fn=node_aff_fn
    )

    print("K shape:", K.shape)  # [1, (n1*n2), (n1*n2)] for batch=1
    return K

# Load pickle files
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"AS_Datasets","test")
print(dataset_dir)
dataset = SyntheticDatasetGenerator(synteticdataset_settings)
dataset.deserialize_dataset(digraphs = True, path = dataset_dir, number = 10)

# Load graph data
graph1 = dataset.graphs["original"][0]
graph2 = dataset.graphs["noise"][0]

# Extract features
g1_node_feat, g1_edge_feat = extract_features(graph1)
g2_node_feat, g2_edge_feat = extract_features(graph2)

print("Node Features:")
print(g1_node_feat)
print("Edge Features:")
print(g1_edge_feat)

# Normalize features
g1_node_feat, g1_edge_feat = normalize_features(g1_node_feat, g1_edge_feat)
g2_node_feat, g2_edge_feat = normalize_features(g2_node_feat, g2_edge_feat)


print("Node Features:")
print(g1_node_feat)
print("Edge Features:")
print(g1_edge_feat)

# K = compute_affinity_matrix(g1_node_feat, g2_node_feat, g1_edge_feat, g2_edge_feat)

# Create adjacency matrices, it ignores edge attributes
adj1 = np.zeros((len(graph1.nodes), len(graph1.nodes)))
for src, dst in graph1.edges():
    adj1[src, dst] = 1

adj2 = np.zeros((len(graph2.nodes), len(graph2.nodes)))
for src, dst in graph2.edges():
    adj2[src, dst] = 1

# Convert to PyTorch tensors
g1_node_feat = torch.tensor(g1_node_feat)
g2_node_feat = torch.tensor(g2_node_feat)
adj1 = torch.tensor(adj1)
adj2 = torch.tensor(adj2)

desired_dim = 1024

# Padding with 0s
g1_node_feat_padded = F.pad(g1_node_feat, (0, desired_dim - g1_node_feat.shape[1]))
g2_node_feat_padded = F.pad(g2_node_feat, (0, desired_dim - g2_node_feat.shape[1]))

# Apply PCA-GM model for graph matching
match_result = pygm.ipca_gm(
    A1=adj1.to(torch.float32), A2=adj2.to(torch.float32),
    feat1=g1_node_feat_padded, feat2=g2_node_feat_padded,
    pretrain='voc'
)

# X, net = pygm.ngm(K, return_network=True)

# match_result = X

# Convert to a DataFrame for better readability
df_match = pd.DataFrame(match_result.detach().numpy())

matched = pygm.hungarian(match_result)
df_matched = pd.DataFrame(matched)


# # Display the DataFrame in the terminal
# print("Graph Matching Matrix:")
# print(df_match.to_string(index=True, header=True))
# print("Hungarian:")
# print(df_matched.to_string(index=True, header=True))

gv.draw_graphs(graph1,graph2)
gv.draw_graph_matching(graph1, graph2, matched, np.eye(len(graph1.nodes)))
gv.draw_aligned_graph(graph1, graph2, matched, np.eye(len(graph1.nodes)))

# Display the graphs before alignment
gv.visualize_nxgraph_pair(graph1, graph2, "Original Graphs", visualize_alone=True, g1digraph=True, g2digraph=True)
# Display the graphs after alignment
gv.plot_matching_with_visualization(graph1, graph2, matched, "Matched Graphs", visualize_alone=True, g1digraph=True, g2digraph=True)