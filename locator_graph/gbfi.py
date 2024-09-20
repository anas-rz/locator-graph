import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm
import ast
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data, Batch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

import torch.nn.functional as F
from torch_geometric.nn import TAGConv, global_mean_pool

import argparse



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

print(f"Sample code \n {model_code}")
# Parse the code
tree = ast.parse(model_code)

# Build the graph
builder = ModelGraphBuilder()
builder.visit(tree)
print("Created Graph:")
# Print the graph with default attributes
for node in builder.graph:
    print(node)


def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    """
    Trains the GNN model on the given graph data.

    Args:
        model (torch.nn.Module): The GNN model.
        data (Batch): The batched graph data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc, f1 = test_model(model, data)
            # if acc >= 0.63:
            #     break
            print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}')


def test_model(model, data):
    """
    Evaluates the GNN model on the given graph data.

    Args:
        model (torch.nn.Module): The GNN model.
        data (Batch): The batched graph data.

    Returns:
        tuple: Accuracy and mean F1 score of the model on the data.
    """
    model.eval()
    _, pred = model(data).max(dim=1)  # Get predictions
    correct = float(pred.eq(data.y).sum().item())  # Calculate correct predictions
    accuracy = correct / len(data.y)  # Calculate accuracy
    
    # Calculate F1 score
    f1 = f1_score(data.y.cpu(), pred.cpu(), average='macro')  # Use 'macro' for mean F1 score
    
    return accuracy, f1

class GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN, self).__init__()
        self.conv1 = TAGConv(num_node_features, 16)
        self.conv2 = TAGConv(16, 32)
        self.conv3 = TAGConv(32, 16)
        self.fc = torch.nn.Linear(16, 13)

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Final dense layer for classification
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

def main(base_dir, file_path, epochs=1000):
    df = pd.read_csv(file_path)
    df['full_path'] = base_dir + df['Filename']
    all_buggy_files = df[df['buggy'] == True]['full_path'].tolist()
    all_right_models = df[df['buggy'] == False]['full_path'].tolist()
    print(f"Number of correct models: {len(all_right_models)}")
    print(f"Number of buggy models: {len(all_buggy_files)}")
    print("Converting all Codes to Graphs using AST:")


    all_fault_graphs = []
    all_fault_labels = []
    for path in tqdm(all_buggy_files):
        try:
            with open(path, 'r') as file:
                content = file.read()
            label = path.split('/')[-1].split('.')[0].split('_')[1]
            label = 'b1' if label in ['b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19'] else label
            label = label.replace('b', '')
            label = int(label)
        
            # Parse the code
            tree = ast.parse(content)

            # Build the graph
            builder = ModelGraphBuilder()
            builder.visit(tree)
            all_fault_graphs.append(builder.graph)
            all_fault_labels.append(label)

        except Exception as e:
            pass

    all_graphs =  all_fault_graphs  
    all_labels =  all_fault_labels 

    balanced_graphs, balanced_labels = balance_dataset(all_graphs, all_labels)



    possible_activations = [ None,
        'relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign',
        'selu', 'elu', 'exponential', 'linear', 'None',
    'gelu', 'selu', 'softmax', 'softplus', 'softsign',               'leaky_relu', 'silu', 'hard_silu', 'mish',
                            'hard_sigmoid', 'relu6'
    ]
    possible_regularizers = [None, 'l1_0.2', 'l1_0.3', 'l1_0.4', 'l1_0.5', 'l1_0.6', 'l1_0.7', 'l1_0.8', 'l1_0.9',
                            'l1_l2_0.2', 'l1_l2_0.3', 'l1_l2_0.4', 'l1_l2_0.5', 'l1_l2_0.6', 'l1_l2_0.7', 'l1_l2_0.8',
                            'l2_0.2', 'l2_0.3', 'l2_0.4', 'l2_0.5', 'l2_0.6', 'l2_0.7', 'l2_0.8', 'l2_0.9']
    batch = preprocess_graphs(balanced_graphs, balanced_labels, possible_activations, possible_regularizers)
    num_node_features = batch.num_node_features
    train_data, test_data = train_test_split(batch, test_size=0.25)
    val_data, test_data = train_test_split(test_data, test_size=0.1)

    train_batch = Batch.from_data_list(train_data)
    val_batch = Batch.from_data_list(val_data)
    test_batch = Batch.from_data_list(test_data)


    model = GNN(num_node_features)

    train_model(model, train_batch, 1000)

    accuracy, f1 = test_model(model, val_batch)
    print(f'Val Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    accuracy, f1 = test_model(model, test_batch)
    print(f'Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for loading fault localization dataset")

    # Add arguments
    parser.add_argument('--base_dir', type=str, default="/content/drive/MyDrive/fault_loc_dataset/",
                        help="Base directory for the dataset")
    parser.add_argument('--csv_file', type=str, default="/content/drive/MyDrive/fault_loc_dataset/fault_loc_dataset.csv",
                        help="Path to the CSV file")
    parser.add_argument('--epochs', type=int, default=1000,
                        help="Number of epochs for training")

    # Parse arguments
    args = parser.parse_args()

    # Call main function
    main(args.base_dir, args.csv_file, epochs=args.epochs)
    
