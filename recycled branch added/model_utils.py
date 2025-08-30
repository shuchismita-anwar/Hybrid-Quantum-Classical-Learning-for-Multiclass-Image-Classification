import os
import numpy as np


def save_best_model(params, filename="fmnist[1289]+8+[3,5] .npz"):
    """Save the best model's parameters to a file."""
    np.savez(filename, params=params)
    print(f"Model saved as {filename}")


def load_best_model(filename="best_model.npz"):
    """Load the model's parameters from a file."""
    if os.path.exists(filename):
        data = np.load(filename)
        return data['params']
    else:
        print(f"No model file found at {filename}")
        return None