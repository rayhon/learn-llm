import math
from typing import List, Tuple
from math import exp
import numpy as np
import time
import random


def softmax(input_vectors):
    """
    Compute softmax values for each vector in the input.
    Works for both single vector and multiple vectors (batch).
    
    Args:
        input_vectors: Single vector [x1, x2, ...] or batch of vectors [[x1, x2, ...], [y1, y2, ...]]
        
    Returns:
        Probabilities of same shape as input, where each vector sums to 1
    """
    # Handle single vector case
    if not isinstance(input_vectors[0], list):
        input_vectors = [input_vectors]
    
    result = []
    for vector in input_vectors:
        # Find max for numerical stability
        max_val = max(vector)
        
        # Calculate exp(x - max) for each element
        exponents = [exp(x - max_val) for x in vector]
        
        # Calculate sum for normalization
        sum_of_exponents = sum(exponents)
        
        # Calculate probabilities and round to 3 decimal places
        probabilities = [round(exp_x / sum_of_exponents, 3) for exp_x in exponents]
        
        result.append(probabilities)
    
    # Return single vector if input was single vector
    return result[0] if not isinstance(input_vectors[0], list) else result


def softmax_raw(x: List[List[float]]) -> List[List[float]]:
    """
    Compute softmax values for each row of scores in x.
    Using raw mathematical operations without NumPy.
    """
    def softmax_single_row(row: List[float]) -> List[float]:
        # Find max for numerical stability
        max_val = max(row)
        # Compute exp of each element
        exp_vals = [math.exp(x - max_val) for x in row]
        # Sum of all exp values
        sum_exp = sum(exp_vals)
        # Normalize by sum
        return [exp_x / sum_exp for exp_x in exp_vals]
    
    return [softmax_single_row(row) for row in x]

def matrix_multiplication_raw(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """
    Perform matrix multiplication without using NumPy.
    """
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("Invalid matrix dimensions for multiplication")
    
    # Initialize result matrix with zeros
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    
    # Perform multiplication
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    
    return result

def dot_product_raw(a: List[float], b: List[float]) -> float:
    """
    Compute dot product between two vectors.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return sum(x * y for x, y in zip(a, b))

def vector_magnitude_raw(v: List[float]) -> float:
    """
    Compute magnitude (L2 norm) of a vector.
    """
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity_raw(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors without NumPy.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    
    magnitude_product = vector_magnitude_raw(a) * vector_magnitude_raw(b)
    if magnitude_product == 0:
        return 0.0
    
    return dot_product_raw(a, b) / magnitude_product

def sigmoid_raw(x: float) -> float:
    """
    Compute sigmoid activation function.
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def relu_raw(x: float) -> float:
    """
    Compute ReLU (Rectified Linear Unit) activation function.
    """
    return max(0.0, x)

def leaky_relu_raw(x: float, alpha: float = 0.01) -> float:
    """
    Compute Leaky ReLU activation function.
    """
    return x if x > 0 else alpha * x

def tanh_raw(x: float) -> float:
    """
    Compute hyperbolic tangent activation function.
    """
    try:
        exp_pos = math.exp(x)
        exp_neg = math.exp(-x)
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)
    except OverflowError:
        return 1.0 if x > 0 else -1.0

def cross_entropy_loss_raw(y_true: List[float], y_pred: List[float], epsilon: float = 1e-15) -> float:
    """
    Compute cross entropy loss between true and predicted values.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input vectors must have same length")
    
    # Clip predictions to avoid log(0)
    y_pred_clipped = [max(min(p, 1.0 - epsilon), epsilon) for p in y_pred]
    
    # Compute cross entropy
    return -sum(t * math.log(p) for t, p in zip(y_true, y_pred_clipped)) / len(y_true)

def l2_regularization_raw(weights: List[float]) -> float:
    """
    Compute L2 regularization term for given weights.
    """
    return np.sum(weights ** 2)

def weighted_sum_raw(inputs: List[float], weights: List[float], bias: float = 0.0) -> float:
    """
    Compute the weighted sum (linear combination) of inputs and weights with bias.
    This is the basic operation performed by a neuron in neural networks.
    Formula: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
    
    Parameters:
        inputs: List of input values [x₁, x₂, ..., xₙ]
        weights: List of weight values [w₁, w₂, ..., wₙ]
        bias: Bias term b (default: 0.0)
    
    Returns:
        The weighted sum of inputs and weights plus bias
    """
    if len(inputs) != len(weights):
        raise ValueError("Number of inputs must match number of weights")
    
    return np.matmul(inputs, weights) + bias  # or: inputs @ weights + bias

def layer_weighted_sum_raw(inputs: List[float], weights: List[List[float]], biases: List[float]) -> List[float]:
    """
    Compute weighted sums for an entire layer of neurons.
    This represents the operation performed by a full layer in neural networks.
    
    Parameters:
        inputs: List of input values [x₁, x₂, ..., xₙ]
        weights: List of weight lists for each neuron [[w₁₁, w₁₂, ...], [w₂₁, w₂₂, ...], ...]
        biases: List of bias terms for each neuron [b₁, b₂, ...]
    
    Returns:
        List of weighted sums for each neuron in the layer
    """
    if not weights or len(biases) != len(weights):
        raise ValueError("Number of bias terms must match number of neurons (weight lists)")
    
    return [weighted_sum_raw(inputs, neuron_weights, bias) 
            for neuron_weights, bias in zip(weights, biases)]

# Example usage and testing
def test_functions():
    # Add these tests before the existing tests
    # Test weighted sum (single neuron)
    inputs = [0.5, 0.8, 0.2]
    weights = [0.4, 0.3, 0.6]
    bias = 0.1
    print("Single neuron weighted sum:", weighted_sum_raw(inputs, weights, bias))
    
    # Test layer weighted sums (multiple neurons)
    layer_weights = [[0.4, 0.3, 0.6],
                    [0.2, 0.5, 0.7]]
    layer_biases = [0.1, 0.2]
    print("Layer weighted sums:", layer_weighted_sum_raw(inputs, layer_weights, layer_biases))
    
    # Test softmax
    scores = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    print("Softmax output:", softmax_raw(scores))
    
    # Test matrix multiplication
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    print("Matrix multiplication:", matrix_multiplication_raw(a, b))
    
    # Test cosine similarity
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [4.0, 5.0, 6.0]
    print("Cosine similarity:", cosine_similarity_raw(vec1, vec2))
    
    # Test activation functions
    x = 2.0
    print("Sigmoid:", sigmoid_raw(x))
    print("ReLU:", relu_raw(x))
    print("Leaky ReLU:", leaky_relu_raw(x))
    print("Tanh:", tanh_raw(x))
    
    # Test loss function
    y_true = [1.0, 0.0, 0.0]
    y_pred = [0.9, 0.05, 0.05]
    print("Cross entropy loss:", cross_entropy_loss_raw(y_true, y_pred))
    
    # Test L2 regularization
    weights = [0.1, 0.2, 0.3]
    print("L2 regularization:", l2_regularization_raw(weights))

def test_softmax_comparison():
    """Compare the basic softmax with the numerically stable version"""
    print("Comparing Softmax Implementations:")
    print("-" * 50)
    
    # Test case 1: Small numbers
    input1 = [1.0, 2.0, 3.0]
    print("\nTest 1 - Small numbers:")
    print("Input:", input1)
    print("Basic softmax:", softmax(input1))
    print("Stable softmax:", softmax_raw([input1])[0])
    
    # Test case 2: Large numbers (where numerical stability matters)
    input2 = [100.0, 101.0, 102.0]
    print("\nTest 2 - Large numbers:")
    print("Input:", input2)
    try:
        print("Basic softmax:", softmax(input2))
    except OverflowError:
        print("Basic softmax: OVERFLOW ERROR!")
    print("Stable softmax:", softmax_raw([input2])[0])
    
    # Test case 3: Very negative numbers
    input3 = [-100.0, -101.0, -102.0]
    print("\nTest 3 - Very negative numbers:")
    print("Input:", input3)
    print("Basic softmax:", softmax(input3))
    print("Stable softmax:", softmax_raw([input3])[0])

def test_improved_softmax():
    """Test the improved softmax implementation with various inputs"""
    print("\nTesting Improved Softmax:")
    print("-" * 50)
    
    # Test 1: Single vector
    single_vector = [2.0, 1.0, 0.5]
    print("\nTest 1 - Single vector:")
    print("Input:", single_vector)
    print("Output:", softmax(single_vector))
    print("Sum of probabilities:", sum(softmax(single_vector)))
    
    # Test 2: Multiple vectors (batch)
    batch_vectors = [
        [2.0, 1.0, 0.5],  # First vector
        [3.0, 2.0, 1.0]   # Second vector
    ]
    print("\nTest 2 - Batch of vectors:")
    print("Input:", batch_vectors)
    result = softmax(batch_vectors)
    print("Output:", result)
    print("Sum of probabilities for each vector:", [sum(vec) for vec in result])
    
    # Test 3: Large numbers (numerical stability test)
    large_numbers = [100.0, 101.0, 102.0]
    print("\nTest 3 - Large numbers (stability test):")
    print("Input:", large_numbers)
    print("Output:", softmax(large_numbers))
    print("Sum of probabilities:", sum(softmax(large_numbers)))
    
    # Test 4: Mixed numbers
    mixed_batch = [
        [1.0, 2.0, 3.0],      # Normal numbers
        [100.0, 101.0, 102.0], # Large numbers
        [-1.0, -2.0, -3.0]     # Negative numbers
    ]
    print("\nTest 4 - Mixed batch:")
    print("Input:", mixed_batch)
    result = softmax(mixed_batch)
    print("Output:", result)
    print("Sum of probabilities for each vector:", [sum(vec) for vec in result])

def numpy_softmax(x):
    """NumPy vectorized implementation of softmax"""
    # Convert input to numpy array if it isn't already
    x = np.array(x)
    # Handle single vector case
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Compute softmax
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def test_performance():
    """Compare performance of loop-based vs vectorized implementation"""
    print("\nPerformance Comparison:")
    print("-" * 50)
    
    # Test cases with different sizes
    sizes = [
        (100, 5),     # 100 vectors of size 5
        (1000, 10),   # 1000 vectors of size 10
        (10000, 20)   # 10000 vectors of size 20
    ]
    
    for n_vectors, vector_size in sizes:
        # Generate random test data
        test_data = [[random.random() for _ in range(vector_size)] 
                    for _ in range(n_vectors)]
        test_data_np = np.array(test_data)
        
        print(f"\nTesting with {n_vectors} vectors of size {vector_size}:")
        
        # Time loop-based version
        start = time.time()
        _ = softmax(test_data)
        loop_time = time.time() - start
        print(f"Loop-based time: {loop_time:.4f} seconds")
        
        # Time NumPy version
        start = time.time()
        _ = numpy_softmax(test_data_np)
        numpy_time = time.time() - start
        print(f"NumPy time: {numpy_time:.4f} seconds")
        
        print(f"NumPy is {loop_time/numpy_time:.1f}x faster")

def test_deep_learning_scenarios():
    """
    Demonstrate how softmax is typically used in deep learning scenarios
    """
    print("\nDeep Learning Softmax Examples:")
    print("-" * 50)
    
    # Scenario 1: Single sample prediction (vector input)
    print("\nScenario 1: Single Sample Prediction")
    print("----------------------------------------")
    # Simulating final layer output for one image (3 classes: cat, dog, bird)
    logits = [2.0, 1.0, 0.5]
    probabilities = softmax(logits)
    print("Model output (logits):", logits)
    print("Class probabilities:", probabilities)
    print("Sum of probabilities:", sum(probabilities))  # Should be 1.0
    
    # Scenario 2: Batch prediction (matrix input)
    print("\nScenario 2: Batch Prediction")
    print("----------------------------------------")
    # Simulating final layer output for 3 images
    batch_logits = [
        [2.0, 1.0, 0.5],  # First image
        [3.0, 2.0, 1.0],  # Second image
        [0.1, 2.0, 3.0]   # Third image
    ]
    batch_probabilities = softmax(batch_logits)
    
    print("Batch predictions:")
    class_names = ['cat', 'dog', 'bird']
    for i, (logits, probs) in enumerate(zip(batch_logits, batch_probabilities)):
        print(f"\nSample {i + 1}:")
        print("Logits:", logits)
        print("Probabilities:", probs)
        predicted_class = class_names[probs.index(max(probs))]
        print(f"Predicted class: {predicted_class} with confidence {max(probs):.3f}")
    
    # Scenario 3: Making a decision
    print("\nScenario 3: Decision Making")
    print("----------------------------------------")
    # Usually we use a threshold for confidence
    confidence_threshold = 0.5
    
    logits = [5.0, 1.0, 0.5]  # Strong prediction for first class
    probs = softmax(logits)
    max_prob = max(probs)
    predicted_class = class_names[probs.index(max_prob)]
    
    print("Logits:", logits)
    print("Probabilities:", probs)
    print(f"Prediction: {predicted_class}" + 
          f" (Confidence: {max_prob:.3f}" +
          f" {'> ' if max_prob > confidence_threshold else '< '}threshold {confidence_threshold})")

if __name__ == "__main__":
    test_functions()
    test_softmax_comparison()
    test_improved_softmax()
    test_performance()
    test_deep_learning_scenarios() 