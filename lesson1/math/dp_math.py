import numpy as np

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    Args: 
        x: Input array of shape (batch_size, num_classes) or (num_classes,)
    Returns:
        Softmax probabilities of same shape as input
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted_x)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    probabilities = exp_x / sum_exp_x
    
    return probabilities.squeeze() if x.ndim == 1 else probabilities

def matrix_multiplication(a, b):
    """
    Perform matrix multiplication.
    
    Args:
        a: First input array of shape (m, n)
        b: Second input array of shape (n, p)
    
    Returns:
        Result array of shape (m, p)
    """
    return np.matmul(a, b)

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First input vector
        b: Second input vector
    
    Returns:
        Similarity score between -1 and 1
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def batch_cosine_similarity(a, b):
    """
    Compute cosine similarity for batches of vectors.
    
    Args:
        a: First batch of vectors of shape (batch_size, n_features)
        b: Second batch of vectors of shape (batch_size, n_features)
    
    Returns:
        Array of similarity scores for each pair
    """
    dot_products = np.sum(a * b, axis=1)
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    return dot_products / (norms_a * norms_b)

def sigmoid(x):
    """
    Compute sigmoid activation function.
    
    Args:
        x: Input array
    
    Returns:
        Array of same shape with sigmoid activation applied
    """
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    """
    Compute ReLU (Rectified Linear Unit) activation function.
    
    Args:
        x: Input array
    
    Returns:
        Array of same shape with ReLU activation applied
    """
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """
    Compute Leaky ReLU activation function.
    
    Args:
        x: Input array
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Array of same shape with Leaky ReLU activation applied
    """
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    """
    Compute hyperbolic tangent activation function.
    
    Args:
        x: Input array
    
    Returns:
        Array of same shape with tanh activation applied
    """
    return np.tanh(x)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Compute cross entropy loss between true and predicted values.
    
    Args:
        y_true: One-hot encoded labels
        y_pred: Predicted probabilities
        epsilon: Small constant for numerical stability (default: 1e-15)
    
    Returns:
        Scalar loss value
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def l2_regularization(weights):
    """
    Compute L2 regularization term.
    
    Args:
        weights: Input array of weights
    
    Returns:
        Scalar regularization value
    """
    return np.sum(weights ** 2)

def weighted_sum(inputs, weights, bias):
    """
    Compute the weighted sum for a layer.
    
    Args:
        inputs: Input array of shape (batch_size, input_dim)
        weights: Weight array of shape (input_dim, output_dim)
        bias: Bias array of shape (output_dim,)
    
    Returns:
        Output array of shape (batch_size, output_dim)
    """
    return np.matmul(inputs, weights) + bias

def layer_forward(inputs, weights, biases):
    """
    Compute forward pass for a neural network layer.
    
    Args:
        inputs: Input array of shape (batch_size, input_dim)
        weights: Weight array of shape (input_dim, output_dim)
        biases: Bias array of shape (output_dim,)
    
    Returns:
        Output array of shape (batch_size, output_dim)
    """
    return weighted_sum(inputs, weights, biases)

def test_functions():
    """Test the implementations with example inputs."""
    # Test softmax with specific example
    print("Testing Softmax Function:")
    print("-" * 50)
    logits = np.array([
        [2.0, 1.0, 0.5],  # First example
        [3.0, 2.0, 1.0]   # Second example
    ])
    probabilities = softmax(logits)
    print("Logits:\n", logits)
    print("\nSoftmax probabilities:\n", probabilities)
    print("\nSum of probabilities (should be 1 for each example):", np.sum(probabilities, axis=1))
    
    print("\nTesting Other Functions:")
    print("-" * 50)
    
    
    # Test data for other functions
    x = np.array([[1, 2, 3], [4, 5, 6]])
    weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    bias = np.array([0.1, 0.2])
    
    print("\nLayer forward output:")
    print(layer_forward(x, weights, bias))
    
    print("\nActivation functions output:")
    print("Sigmoid:", sigmoid(x))
    print("ReLU:", relu(x))
    print("Leaky ReLU:", leaky_relu(x))
    print("Tanh:", tanh(x))
    
    # Test cosine similarity
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print("\nCosine similarity:", cosine_similarity(a, b))

if __name__ == "__main__":
    test_functions()
