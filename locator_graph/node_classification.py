# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import argparse
import pandas as pd
import ast
import torch
from tqdm import tqdm



import ast

model_code = """
model = Sequential()
model.add(GRU(256 , return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256))
model.add(Dropout(0.4))
model.add(Dense(64 ,  activation = 'relu'))
model.add(Dense(1))

"""

# Define default values for variables
default_values = {
    'vocab_size': 10000,
    'embedding_dim': 100,
    'max_length': 80,
    'gru_dim': 64,
    'dense_dim': 32,
    'max_features': 10000
}

# Define default attributes for different layer types
default_attributes = {
    'Embedding': {
        'input_dim': None,
        'output_dim': None,
        'input_length': None
    },
    'GRU': {
        'units': None,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'use_bias': True,
        'kernel_initializer': 'glorot_uniform',
        'recurrent_initializer': 'orthogonal',
        'bias_initializer': 'zeros',
        'unit_forget_bias': True,
        'kernel_regularizer': None,
        'recurrent_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'recurrent_constraint': None,
        'bias_constraint': None,
        'dropout': 0.0,
        'recurrent_dropout': 0.0,
        'implementation': 1,
        'return_sequences': False,
        'return_state': False,
        'go_backwards': False,
        'stateful': False,
        'unroll': False,
        'time_major': False,
        'reset_after': True
    },
    'Dense': {
        'units': None,
        'activation': None,
        'use_bias': True,
        'kernel_initializer': 'glorot_uniform',
        'bias_initializer': 'zeros',
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None,
        'lora_rank': None
    },
    'Dropout': {
        'rate': None,
        'noise_shape': None,
        'seed': None
    },
    'LSTM': {
        'units': None,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'use_bias': True,
        'kernel_initializer': 'glorot_uniform',
        'recurrent_initializer': 'orthogonal',
        'bias_initializer': 'zeros',
        'unit_forget_bias': True,
        'kernel_regularizer': None,
        'recurrent_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'recurrent_constraint': None,
        'bias_constraint': None,
        'dropout': 0.0,
        'recurrent_dropout': 0.0,
        'implementation': 2,
        'return_sequences': False,
        'return_state': False,
        'go_backwards': False,
        'stateful': False,
        'unroll': False,
        'time_major': False
    },
    'Conv2D': {
        'filters': None,
        'kernel_size': None,
        'strides': (1, 1),
        'padding': 'valid',
        'data_format': None,
        'dilation_rate': (1, 1),
        'activation': None,
        'use_bias': True,
        'kernel_initializer': 'glorot_uniform',
        'bias_initializer': 'zeros',
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None
    },
    'Conv1D': {
        'filters': None,
        'kernel_size': None,
        'strides': 1,
        'padding': 'valid',
        'data_format': None,
        'dilation_rate': 1,
        'activation': None,
        'use_bias': True,
        'kernel_initializer': 'glorot_uniform',
        'bias_initializer': 'zeros',
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None
    },
    'MaxPooling2D': {
        'pool_size': (2, 2),
        'strides': None,
        'padding': 'valid',
        'data_format': None
    },
    'MaxPooling1D': {
        'pool_size': 2,
        'strides': None,
        'padding': 'valid'
    },
    'Flatten': {}
}

class ModelGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.graph = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add':
            layer = node.args[0]
            if isinstance(layer.func, ast.Name) and layer.func.id == 'Bidirectional':
                wrapped_layer = layer.args[0]
                self.add_layer(wrapped_layer, bidirectional=True)
            else:
                self.add_layer(layer)
        self.generic_visit(node)

    def get_value(self, node):
        if isinstance(node, ast.Name):
            return default_values.get(node.id, node.id)
        return ast.literal_eval(node)

    def get_layer_type(self, layer_func):
        if isinstance(layer_func, ast.Attribute):
            return layer_func.attr
        elif isinstance(layer_func, ast.Name):
            return layer_func.id
        return None

    def add_layer(self, layer, bidirectional=False):
        layer_type = self.get_layer_type(layer.func)
        if layer_type is None:
            return

        layer_attributes = default_attributes.get(layer_type, {}).copy()

        # Handle positional arguments
        for i, arg in enumerate(layer.args):
            keys = list(layer_attributes.keys())
            if i < len(keys):
                layer_attributes[keys[i]] = self.get_value(arg)

        # Handle keyword arguments
        for kw in layer.keywords:
            layer_attributes[kw.arg] = self.get_value(kw.value)

        if bidirectional:
            self.graph.append((f'Bidirectional_{layer_type}_forward', layer_attributes.copy()))
            self.graph.append((f'Bidirectional_{layer_type}_backward', layer_attributes.copy()))
        else:
            self.graph.append((layer_type, layer_attributes))

# Parse the code
tree = ast.parse(model_code)

# Build the graph
builder = ModelGraphBuilder()
builder.visit(tree)

# Print the graph with default attributes
for node in builder.graph:
    print(node)




def balance_dataset(graphs, labels):
    """
    Balances the dataset by oversampling the minority class or undersampling the majority class.

    Args:
        graphs (list): List of graphs.
        labels (list): List of labels corresponding to the graphs.

    Returns:
        balanced_graphs (list): Balanced list of graphs.
        balanced_labels (list): Balanced list of labels.
    """
    # Separate by class
    class_indices = {}
    for index, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(index)

    # Determine the maximum class size
    max_count = max(len(indices) for indices in class_indices.values())

    balanced_graphs = []
    balanced_labels = []

    for label, indices in class_indices.items():
        class_graphs = [graphs[i] for i in indices]
        class_labels = [labels[i] for i in indices]

        # Oversample minority class
        if len(class_graphs) < max_count:
            class_graphs_oversampled, class_labels_oversampled = resample(
                class_graphs,
                class_labels,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
            balanced_graphs.extend(class_graphs_oversampled)
            balanced_labels.extend(class_labels_oversampled)
        else:
            balanced_graphs.extend(class_graphs)
            balanced_labels.extend(class_labels)

    return balanced_graphs, balanced_labels

# Example usage



def preprocess_graphs(graphs, labels, possible_activations):
    """
    Preprocesses multiple graphs into node features and edge index for GNN input.

    Args:
        graphs (list of list): List of graphs, where each graph is a list of tuples (layer, attributes).
        labels (list): List of labels corresponding to each graph.
        possible_activations (list): List of possible activation functions for one-hot encoding.

    Returns:
        Batch: PyTorch Geometric Batch object containing all processed graphs.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    try:
        encoder.fit(np.array(possible_activations).reshape(-1, 1))
    except Exception as e:
        print(f"Error fitting encoder: {e}")
        raise

    # Attributes to keep
    attributes_to_keep = [
        'units', 'activation', 'recurrent_activation', 'kernel_regularizer', 'recurrent_regularizer',
        'bias_regularizer', 'activity_regularizer', 'recurrent_dropout', 'dropout'
    ]

    data_list = []
    max_node_feature_length = 0

    for graph_index, (graph, label) in enumerate(zip(graphs, labels)):
        node_features = []
        edge_index = []

        try:
            for layer, attributes in graph:
                encoded_attributes = {}

                if 'activation' in attributes:
                    activation = attributes.get('activation', 'None')
                    if activation not in possible_activations:
                        print(f"Warning: Activation function '{activation}' not in possible_activations")
                        activation = 'None'
                    encoded_activation = encoder.transform([[activation]]).flatten().tolist()
                    encoded_attributes['activation'] = encoded_activation

                if 'recurrent_activation' in attributes:
                    recurrent_activation = attributes.get('recurrent_activation', 'None')
                    if recurrent_activation not in possible_activations:
                        print(f"Warning: Recurrent activation function '{recurrent_activation}' not in possible_activations")
                        recurrent_activation = 'None'
                    encoded_recurrent_activation = encoder.transform([[recurrent_activation]]).flatten().tolist()
                    encoded_attributes['recurrent_activation'] = encoded_recurrent_activation

                for key in attributes_to_keep:
                    if key in attributes and key not in ['activation', 'recurrent_activation']:
                        value = attributes.get(key, 0)
                        if value is None:
                            value = 0
                        encoded_attributes[key] = value

                combined_features = []
                if 'activation' in encoded_attributes:
                    combined_features.extend(encoded_attributes['activation'])
                if 'recurrent_activation' in encoded_attributes:
                    combined_features.extend(encoded_attributes['recurrent_activation'])

                for key in attributes_to_keep:
                    if key in encoded_attributes and key not in ['activation', 'recurrent_activation']:
                        combined_features.append(encoded_attributes[key])

                node_features.append(combined_features)
                max_node_feature_length = max(max_node_feature_length, len(combined_features))

            if not node_features:
                print(f"Warning: Skipping empty node_features for graph {graph_index}")
                continue

            # Ensure all node features have the same length
            node_features = [f + [0] * (max_node_feature_length - len(f)) for f in node_features]
            node_features = np.array(node_features, dtype=float)
            x = torch.tensor(node_features, dtype=torch.float)

            # Create edge index
            num_nodes = len(node_features)
            if num_nodes > 1:
                edge_index = torch.tensor(
                    [[i, i + 1] for i in range(num_nodes - 1)] +
                    [[i + 1, i] for i in range(num_nodes - 1)], dtype=torch.long
                ).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge_index for single-node graphs

            # Convert label to tensor
            y = torch.tensor([label], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        except Exception as e:
            print(f"Error processing graph {graph_index}: {e}")
            continue

    try:
        if not data_list:
            print("Warning: No valid graphs to create a batch.")
            return None

        # Create batch from list of graphs
        batch = Batch.from_data_list(data_list)

        # Debugging: Print shapes of tensors in data_list
        for i, data in enumerate(data_list):
            print(f"Graph {i} - x shape: {data.x.shape}, edge_index shape: {data.edge_index.shape}")

    except Exception as e:
        print(f"Error creating batch: {e}")
        raise

    return batch

# Define the possible activations
possible_activations = [ None,
    'relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign',
    'selu', 'elu', 'exponential', 'linear', 'None',
 'gelu', 'selu', 'softmax', 'softplus', 'softsign',               'leaky_relu', 'silu', 'hard_silu', 'mish',
                         'hard_sigmoid', 'relu6'
]




def calculate_accuracy(pred, labels):
    pred_labels = pred.argmax(dim=1)  # Get the predicted labels
    correct = (pred_labels == labels).sum().item()  # Count correct predictions
    accuracy = correct / labels.size(0)  # Calculate accuracy
    return accuracy

def train(batch, epochs=100):
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the parameter gradients
        out = model(batch)  # Forward pass
        loss = loss_fn(out, batch.y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters

        # Calculate accuracy
        accuracy = calculate_accuracy(out, batch.y)

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')

class GNNNodeClassification(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNNodeClassification, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, num_classes)  # `num_classes` corresponds to the number of classes for node classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Output layer for node classification (no global pooling)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)  # log_softmax over the class dimension


def main(input_csv, base_dir):
    df = pd.read_csv(input_csv)

    df['full_path'] = base_dir + df['Filename']

    all_buggy_files = df[df['buggy'] == True]['full_path'].tolist()
    all_right_models = df[df['buggy'] == False]['full_path'].tolist()
    
    all_right_graphs = []
    all_right_labels = []
    for path in tqdm(all_right_models):
        with open(path, 'r') as file:
            content = file.read()
        try:
            # Parse the code
            tree = ast.parse(content)

            # Build the graph
            builder = ModelGraphBuilder()
            builder.visit(tree)
            all_right_graphs.append(builder.graph)
            all_right_labels.append(0)
        except Exception as e:
            print(f"Error parsing file {path}: {e}")

    all_fault_graphs = []
    all_fault_labels = []
    for path in tqdm(all_buggy_files):
        with open(path, 'r') as file:
            content = file.read()
            label = path.split('/')[-1].split('.')[0].split('_')[1]
            label = 'b1' if label in ['b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19'] else label
            label = label.replace('b', '')
            label = int(label)
        try:
            # Parse the code
            tree = ast.parse(content)

            # Build the graph
            builder = ModelGraphBuilder()
            builder.visit(tree)
            all_fault_graphs.append(builder.graph)
            all_fault_labels.append(label)
        except Exception as e:
            print(f"Error parsing file {path}: {e}")

    all_right_labels = [0] * len(all_right_graphs)
    all_fault_labels = [1] * len(all_fault_graphs)

    all_class_labels = all_right_labels + all_fault_labels

    all_graphs = all_right_graphs + all_fault_graphs
    all_labels = all_right_labels + all_fault_labels

    print(len(all_graphs), len(all_labels), len(all_class_labels))

    balanced_graphs, balanced_labels = balance_dataset(all_graphs, all_class_labels)

    # Now you can proceed with further processing or batching
    batch = preprocess_graphs(balanced_graphs, balanced_labels, possible_activations)

    batch.y = node_labels

    model = GNNNodeClassification(num_node_features=batch.num_features, num_classes=2)  # Binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    train(batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process fault localization dataset and train a GNN model.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for model files.")
    
    args = parser.parse_args()
    main(args.input_csv, args.base_dir)
