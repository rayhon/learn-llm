import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Any

def scale_vector_regular(vector: np.ndarray, scale: float) -> np.ndarray:
    result = np.zeros_like(vector)
    for i in range(len(vector)):
        result[i] = scale * vector[i]
    return result

def scale_vector_numpy(vector: np.ndarray, scale: float) -> np.ndarray:
    return scale * vector

def dot_product_regular(v1: np.ndarray, v2: np.ndarray) -> float:
    product = 0
    for i in range(len(v1)):
        product += v1[i] * v2[i]
    return product

def dot_product_numpy(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1.T, v2)

def vector_multiply_regular(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    result = np.zeros_like(v1)
    for i in range(len(v1)):
        result[i] = v1[i] * v2[i]
    return result

def vector_multiply_numpy(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return v1 * v2

def matrix_multiply_regular(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    result = np.zeros_like(m1)
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            result[i, j] = m1[i, j] * m2[i, j]
    return result

def matrix_multiply_numpy(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return np.multiply(m1, m2)

def time_function(func: Callable, *args) -> float:
    """Time a function execution and return milliseconds."""
    start = time.process_time()
    _ = func(*args)
    end = time.process_time()
    return 1000 * (end - start)

def analyze_time_complexity(operations: List[Tuple[str, Callable, tuple]], sizes: List[int] = None):
    """
    Analyze time complexity for multiple operations.
    
    Args:
        operations: List of tuples containing (operation_name, function, sample_args)
            - operation_name: str, name of the operation for the legend
            - function: the function to test
            - sample_args: tuple of sample arguments that shows the expected input types
        sizes: List of sizes to test. If None, defaults to standard sizes
    """
    if sizes is None:
        sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    
    complexity = pd.DataFrame()
    complexity['sizes'] = sizes
    
    for op_name, func, sample_args in operations:
        times = []
        for size in sizes:
            # Create new args with the same structure as sample_args but with new size
            args = []
            for arg in sample_args:
                if isinstance(arg, np.ndarray):
                    if len(arg.shape) == 2:  # Matrix
                        if arg.shape[0] == arg.shape[1]:  # Square matrix
                            args.append(np.random.rand(size, size))
                        else:  # Non-square matrix/vector
                            args.append(np.random.rand(size, arg.shape[1]))
                    else:  # 1D array
                        args.append(np.random.rand(size))
                else:  # For scalars, just use the sample value
                    args.append(arg)
            
            times.append(time_function(func, *args))
        
        complexity[op_name] = times
    
    return complexity

def plot_time_complexity(complexity_df: pd.DataFrame, title: str = "Time Complexity Analysis"):
    """
    Plot time complexity analysis results.
    
    Args:
        complexity_df: DataFrame with 'sizes' column and operation columns
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))  # Match original figure size
    
    # Plot each operation
    for column in complexity_df.columns:
        if column != 'sizes':
            plt.plot(complexity_df['sizes'], complexity_df[column], label=column)
    
    plt.xscale('log')
    # Removed yscale('log') to match original linear y-axis
    plt.xlabel("Vector Size")
    plt.ylabel("Time taken in ms")
    plt.title(title)
    plt.legend()  # Simplified legend placement
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

def main():
    # Generate test data
    v1 = np.random.rand(1000000, 1)
    v2 = np.random.rand(1000000, 1)
    m1 = np.random.rand(1000, 1000)
    m2 = np.random.rand(1000, 1000)

    # Define all operations to test
    operations = [
        ("Vector Scaling (Regular)", scale_vector_regular, (v1, 2)),
        ("Vector Scaling (NumPy)", scale_vector_numpy, (v1, 2)),
        ("Dot Product (Regular)", dot_product_regular, (v1, v2)),
        ("Dot Product (NumPy)", dot_product_numpy, (v1, v2)),
        ("Vector Multiplication (Regular)", vector_multiply_regular, (v1, v2)),
        ("Vector Multiplication (NumPy)", vector_multiply_numpy, (v1, v2)),
        ("Matrix Multiplication (Regular)", matrix_multiply_regular, (m1, m2)),
        ("Matrix Multiplication (NumPy)", matrix_multiply_numpy, (m1, m2))
    ]

    # Run and time each operation
    print("\nOperation Times:")
    print("-" * 50)
    for name, func, sample_args in operations:
        time_ms = time_function(func, *sample_args)
        print(f"{name:<30} = {time_ms:.2f} ms")

    # Example of using the new analysis functions
    print("\nAnalyzing time complexity across different sizes...")
    # Define operations for analysis
    analysis_operations = [
        ("Dot Product (regular)", dot_product_regular, (v1, v2)),
        ("Dot Product (numpy)", dot_product_numpy, (v1, v2))
    ]
    
    complexity_df = analyze_time_complexity(analysis_operations)
    plot_time_complexity(complexity_df, "Time Complexity: Dot Product Operation")
    
    print("\nTime complexity analysis results (in milliseconds):")
    print(complexity_df)

if __name__ == "__main__":
    main()  