from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


class Segmenter:
    def __init__(self, method: str = "kmeans", n_clusters: int = 4, random_state: int = 42, **kwargs: Any):
        self.method = method.lower()
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self.n_clusters = n_clusters

    def _init_model(self):
        if self.method == "kmeans":
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10, **self.kwargs)
        elif self.method == "dbscan":
            self.model = DBSCAN(**self.kwargs)
        elif self.method == "gmm":
            self.model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state, **self.kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X: np.ndarray):
        self._init_model()
        if self.method == "gmm":
            self.model.fit(X)
        else:
            self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        if self.method == "gmm":
            return self.model.predict(X)
        else:
            return self.model.labels_ if hasattr(self.model, "labels_") else self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    def profile(self, df: pd.DataFrame, labels: np.ndarray, id_col: Optional[str] = None) -> pd.DataFrame:
        prof_df = df.copy()
        prof_df["segment"] = labels
        if id_col and id_col in prof_df.columns:
            group_cols = ["segment"]
        else:
            group_cols = ["segment"]
        summary = prof_df.groupby(group_cols).agg(['mean', 'median', 'count'])
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        return summary.reset_index()
