import numpy as np

def normalize(arr, axis=-1):
    return arr / (np.linalg.norm(arr, axis=axis, keepdims=True) + np.finfo(np.float32).eps)
