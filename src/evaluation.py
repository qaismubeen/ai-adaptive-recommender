import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_at_k(recommended_items: list, relevant_items: set, k: int = 10) -> float:
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & relevant_items)
    return hits / k