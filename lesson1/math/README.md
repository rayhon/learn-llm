# Deep Learning Mathematical Functions

## Summary
| File | Purpose |
|------|---------|
| `test_numpy.py` | Performance comparison between traditional for-loop implementations and NumPy vectorized operations. Demonstrates why NumPy is preferred for numerical computations through benchmarking different implementations. |
| `numpy_math.py` | Implementation of common mathematical functions used in deep learning using NumPy for efficient vectorized computation. Includes optimized versions of softmax, matrix operations, activation functions, and loss functions. |
| `raw_math.py` | Pure Python implementations of the same mathematical functions without NumPy, using basic Python loops and math module. Useful for understanding the underlying computations and comparing with vectorized implementations. |

This module provides a collection of essential mathematical functions commonly used in deep learning and neural networks. Each function is implemented using NumPy for efficient numerical computations.

## Installation Requirements

```bash
pip install numpy
```

## Function Descriptions

### 1. Softmax Function
```python
def softmax(x)
```
- **Purpose**: Converts raw model outputs (logits) into probability distributions
- **Use Cases**:
  - Output layer of classification networks
  - Attention mechanisms in transformers
- **Mathematical Properties**:
  - Outputs sum to 1
  - Preserves relative order of inputs
  - Numerically stable implementation
- **Input Shape**: (n_samples, n_features)
- **Output Range**: [0, 1]

### 2. Matrix Multiplication
```python
def matrix_multiplication(a, b)
```
- **Purpose**: Performs matrix multiplication between two matrices
- **Use Cases**:
  - Layer computations in neural networks
  - Feature transformations
  - Weight applications
- **Requirements**:
  - Matrix A shape: (m, n)
  - Matrix B shape: (n, p)
  - Result shape: (m, p)

### 3. Cosine Similarity
```python
def cosine_similarity(a, b)
```
- **Purpose**: Measures similarity between two vectors
- **Use Cases**:
  - Similarity search
  - Document comparison
  - Feature matching
- **Properties**:
  - Scale-invariant
  - Output range: [-1, 1]
  - 1: Perfect similarity
  - -1: Perfect dissimilarity
  - 0: Orthogonal vectors

### 4. Activation Functions

#### 4.1 Sigmoid
```python
def sigmoid(x)
```
- **Purpose**: Squashes input to range [0, 1]
- **Use Cases**:
  - Binary classification
  - Gates in LSTM
- **Properties**:
  - Smooth, differentiable
  - Can cause vanishing gradients
  - Output range: [0, 1]

#### 4.2 ReLU (Rectified Linear Unit)
```python
def relu(x)
```
- **Purpose**: Introduces non-linearity without vanishing gradient
- **Use Cases**:
  - Hidden layers in deep networks
- **Properties**:
  - Fast computation
  - Sparse activation
  - No upper bound
  - Output range: [0, ∞)

#### 4.3 Leaky ReLU
```python
def leaky_relu(x, alpha=0.01)
```
- **Purpose**: Prevents "dying ReLU" problem
- **Use Cases**:
  - Alternative to standard ReLU
- **Properties**:
  - Small positive slope for negative values
  - No dead neurons
  - Output range: (-∞, ∞)

#### 4.4 Tanh
```python
def tanh(x)
```
- **Purpose**: Squashes input to range [-1, 1]
- **Use Cases**:
  - Hidden layers
  - Normalized data processing
- **Properties**:
  - Zero-centered
  - Stronger gradients than sigmoid
  - Output range: [-1, 1]

### 5. Loss Functions

#### 5.1 Cross Entropy Loss
```python
def cross_entropy_loss(y_true, y_pred, epsilon=1e-15)
```
- **Purpose**: Measures prediction error for classification
- **Use Cases**:
  - Classification tasks
  - Training objective
- **Properties**:
  - Non-negative
  - Penalizes confident wrong predictions
  - Numerically stable implementation

### 6. Regularization

#### 6.1 L2 Regularization
```python
def l2_regularization(weights)
```
- **Purpose**: Prevents overfitting
- **Use Cases**:
  - Weight decay in training
  - Model regularization
- **Properties**:
  - Penalizes large weights
  - Encourages smaller, distributed weights
  - Also known as weight decay

## Usage Example

```python
import numpy as np
from test_math import *

# Softmax example
scores = np.array([[1, 2, 3], [4, 5, 6]])
probabilities = softmax(scores)

# Matrix multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = matrix_multiplication(a, b)

# Activation functions
x = np.array([-2, -1, 0, 1, 2])
activated = relu(x)
```

## Notes
- All functions are optimized for use with NumPy arrays
- Implementations include numerical stability considerations
- Functions are vectorized for efficient computation
- Documentation includes shape information for proper use

## Contributing
Feel free to contribute additional mathematical functions or optimizations by submitting a pull request.
