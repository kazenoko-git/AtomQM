import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def generate_grid(x_range, y_range, z_range):
    x = np.linspace(x_range[0], x_range[1], num=100)
    y = np.linspace(y_range[0], y_range[1], num=100)
    z = np.linspace(z_range[0], z_range[1], num=100)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z

def compute_some_operations(X, Y, Z, progress_callback=None):
    size = X.size
    results = np.empty(size)
    
    for i in range(size):
        results[i] = some_complex_function(X[i], Y[i], Z[i])
        if progress_callback and i % (size // 100) == 0:  # Update every 1%
            progress_callback(i / size)
    
    return results

def some_complex_function(x, y, z):
    # Placeholder for the actual computation
    return np.sqrt(x**2 + y**2 + z**2)

# Other functions can be defined here...