import numpy as np
# Simple embeddings for demonstration

# Each word is represented by a 4-dimensional vector
embeddings = {
 'The': np.array([1, 0, 0, 0]),
 'quick': np.array([0, 1, 0, 0]),
 'brown': np.array([0, 0, 1, 0]),
 'fox': np.array([0, 0, 0, 1]),
 'jumped': np.array([1, 1, 0, 0])
}

# Weight matrices (simplified)
W_Q = W_K = W_V = np.array([[1, 0],
 [0, 1],
 [0, 0],
 [0, 0]])
 

def compute_attention(self, input_words):
    # Convert words to embeddings
    E = np.array([embeddings[word] for word in input_words])

    # Compute K, V for all tokens (matrix multiplication using @ as the operator)
    K = E @ W_K  # Shape: (seq_len, 2)
    V = E @ W_V  # Shape: (seq_len, 2)

    # Compute Q for the last token
    Q = E[-1] @ W_Q  # Shape: (1, 2)

    # Compute scaled attention scores
    scale = np.sqrt(2)  # sqrt of key/query dimension (2 in this case)
    scores = (Q @ K.T) / scale  # Shape: (1, seq_len)

    # Apply softmax to get attention weights
    attention_weights = self.softmax(scores)  # Shape: (1, seq_len)

    # Apply attention weights to values
    output = attention_weights @ V  # Shape: (1, 2)

    return output

def compute_attention_with_cache(self, input_words):
    """Compute attention using KV cache"""
    # Get the new token (last word in sequence)
    new_word = input_words[-1]
    e_new = embeddings[new_word]

    # Compute K and V for new token
    K_new = e_new @ W_K  # Shape: (2,)
    V_new = e_new @ W_V  # Shape: (2,)

    # Update cached K and V
    if self.cached_K is None:
        self.cached_K = K_new.reshape(1, -1)  # Shape: (1, 2)
        self.cached_V = V_new.reshape(1, -1)  # Shape: (1, 2)
    else:
        self.cached_K = np.vstack([self.cached_K, K_new])  # Shape: (seq_len, 2)
        self.cached_V = np.vstack([self.cached_V, V_new])  # Shape: (seq_len, 2)

    # Compute Q for the last token
    Q = e_new @ W_Q  # Shape: (2,)

    # Compute scaled attention scores using cached K
    scale = np.sqrt(2)  # sqrt of key/query dimension (2 in this case)
    scores = (Q @ self.cached_K.T) / scale  # Shape: (1, seq_len)

    # Apply softmax to get attention weights
    attention_weights = self.softmax(scores)  # Shape: (1, seq_len)

    # Compute attention output using cached V
    output = attention_weights @ self.cached_V  # Shape: (1, 2)

    return output

# The Problem: Redundant Computations
# Looking at the code below, we can see that for each new token:
# 1. We recompute K and V matrices for ALL previous tokens
# 2. The size of the matrices grows with each token
# 3. Many computations are repeated unnecessarily

# Step 1: Generate "brown"
input_words_step1 = ['The', 'quick']
output_step1 = compute_attention(input_words_step1)
# Step 2: Generate "fox"
input_words_step2 = ['The', 'quick', 'brown']
output_step2 = compute_attention(input_words_step2)
# Step 3: Generate "jumped"
input_words_step3 = ['The', 'quick', 'brown', 'fox']
output_step3 = compute_attention(input_words_step3)

# The Solution: KV Cache
# Step 1: Generate "brown"
input_words_step1 = ['The', 'quick']
output_step1 = compute_attention_with_cache(input_words_step1)
# Step 2: Generate "fox"
input_words_step2 = ['The', 'quick', 'brown']
output_step2 = compute_attention_with_cache(input_words_step2)
# Step 3: Generate "jumped"
input_words_step3 = ['The', 'quick', 'brown', 'fox']
output_step3 = compute_attention_with_cache(input_words_step3)