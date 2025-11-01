from typing import Dict, Union
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Union[float, None]]:
    metrics = {"silhouette": None, "davies_bouldin": None}
    # Silhouette requires at least 2 clusters and less than n_samples clusters
    if len(set(labels)) >= 2 and len(set(labels)) < len(labels):
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            metrics["silhouette"] = None
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            metrics["davies_bouldin"] = None
    return metrics
