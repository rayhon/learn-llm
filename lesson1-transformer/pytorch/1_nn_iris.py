import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Create a Model Class that inherits nn.Module
class Model(nn.Module):
  # Input layer (4 features of the flower) -->
  # Hidden Layer1 (number of neurons) -->
  # H2 (n) -->
  # output (3 classes of iris flowers)
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x


def load_and_prepare_data(url, test_size=0.2, random_state=41):
    """
    Load and prepare the iris dataset for training
    
    Args:
        url (str): URL to the iris dataset
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) as torch tensors
    """
    # Load data
    df = pd.read_csv(url)
    
    # Convert labels to integers
    label_mapping = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
    df['variety'] = df['variety'].map(label_mapping)
    
    # Split features and target
    X = df.drop('variety', axis=1).values
    y = df['variety'].values
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train, epochs=100, learning_rate=0.01):
    """
    Train the neural network model
    
    Args:
        model (nn.Module): The neural network model
        X_train (torch.Tensor): Training features
        y_train (torch.Tensor): Training labels
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        
    Returns:
        list: Training losses per epoch
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    for i in range(epochs):
        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.detach().numpy())
        
        # Print progress
        if i % 10 == 0:
            print(f'Epoch: {i} and loss: {loss}')
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return losses


def plot_training_loss(losses, save_path=None):
    """
    Plot the training loss over epochs
    
    Args:
        losses (list): List of losses per epoch
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses)
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.title('Training Loss Over Time')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def get_flower_name(label):
    """
    Convert numeric label to flower name
    
    Args:
        label (int): Numeric label (0, 1, or 2)
    
    Returns:
        str: Flower name
    """
    flower_names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }
    return flower_names.get(label, "Unknown")


def evaluate_model(model, X_test, y_test, show_detail=False, label_mapping=None):
    """
    Evaluate the model on test data
    
    Args:
        model (nn.Module): Trained model
        X_test (torch.Tensor): Test features
        y_test (torch.Tensor): Test labels
        show_detail (bool): Whether to show detailed prediction analysis
        label_mapping (dict, optional): Dictionary mapping numeric labels to their string representations
        
    Returns:
        float: Accuracy score
    """
    model.eval()
    correct = 0
    
    if show_detail:
        print("\nDetailed Prediction Analysis:")
        print("="*80)
        header = f"{'No.':<4} {'Predicted Probabilities':<45} {'True':<8} {'Predicted':<8} {'Result'}"
        if label_mapping:
            header = f"{'No.':<4} {'Predicted Probabilities':<45} {'True Label':<15} {'Predicted Label':<15} {'Result'}"
        print(header)
        print("-"*80)
    
    with torch.no_grad():
        for i, data in enumerate(X_test):
            # Get model prediction
            y_val = model(data)
            pred_label = y_val.argmax().item()
            true_label = y_test[i].item()
            
            # Check if prediction is correct
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
            
            # Print detailed output if requested
            if show_detail:
                if label_mapping:
                    true_name = label_mapping.get(true_label, str(true_label))
                    pred_name = label_mapping.get(pred_label, str(pred_label))
                    result = "✓" if is_correct else "✗"
                    print(f"{i+1:<4} {str(y_val.numpy()):<45} {true_name:<15} {pred_name:<15} {result}")
                else:
                    result = "✓" if is_correct else "✗"
                    print(f"{i+1:<4} {str(y_val.numpy()):<45} {true_label:<8} {pred_label:<8} {result}")
    
    accuracy = correct / len(y_test)
    
    if show_detail:
        print("="*80)
        print(f"\nSummary:")
        print(f"Total samples: {len(y_test)}")
        print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy


def save_model(model, filepath):
    """
    Save the model weights to a file
    
    Args:
        model (nn.Module): The model to save
        filepath (str): Path where to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a saved model
    
    Args:
        filepath (str): Path to the saved model file
        
    Returns:
        Model: The loaded model
    """
    model = Model()  # Create a new model instance
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")
    return model


def main():
    # Set random seed for reproducibility
    torch.manual_seed(41)
    
    # Initialize model
    model = Model()
    
    # Load and prepare data
    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    X_train, X_test, y_train, y_test = load_and_prepare_data(url)
    
    # Train model
    losses = train_model(model, X_train, y_train)
    
    # Plot training progress
    plot_training_loss(losses)
    
    # Example: Define a label mapping for iris dataset (optional)
    iris_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    
    # Evaluate model with details and label mapping
    print("\nEvaluation before saving:")
    accuracy = evaluate_model(model, X_test, y_test, show_detail=True, label_mapping=iris_mapping)
    
    # Save the model
    save_model(model, 'iris_model.pt')
    
    # Load the model back
    loaded_model = load_model('iris_model.pt')
    
    # Evaluate loaded model
    print("\nEvaluation after loading:")
    loaded_accuracy = evaluate_model(
        loaded_model, 
        X_test, 
        y_test, 
        show_detail=True,
        label_mapping=iris_mapping
    )
    
    # Verify the results match
    print(f"\nOriginal model accuracy: {accuracy:.4f}")
    print(f"Loaded model accuracy: {loaded_accuracy:.4f}")


if __name__ == "__main__":
    main()

