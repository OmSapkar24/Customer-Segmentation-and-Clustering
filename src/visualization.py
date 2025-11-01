from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def reduce_pca(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=random_state)
    Xr = pca.fit_transform(X)
    return Xr, pca


def reduce_tsne(X: np.ndarray, n_components: int = 2, random_state: int = 42, **kwargs) -> np.ndarray:
    tsne = TSNE(n_components=n_components, random_state=random_state, **kwargs)
    return tsne.fit_transform(X)


def reduce_umap(X: np.ndarray, n_components: int = 2, random_state: int = 42, **kwargs) -> np.ndarray:
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
    return reducer.fit_transform(X)


def scatter_2d(X2: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "2D Projection", palette: str = "tab10") -> plt.Axes:
    sns.set(style="whitegrid")
    ax = sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=labels, palette=palette, s=40, edgecolor="none")
    ax.set_title(title)
    if labels is None:
        ax.legend_.remove() if ax.legend_ else None
    return ax


def plot_segment_profiles(df: pd.DataFrame, labels: np.ndarray, numeric_cols: Optional[list] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    prof = df.copy()
    prof['segment'] = labels
    agg = prof.groupby('segment')[numeric_cols].mean()
    fig, ax = plt.subplots(figsize=figsize)
    agg.T.plot(kind='bar', ax=ax)
    ax.set_title('Segment Profiles (mean by numeric feature)')
    ax.set_ylabel('Mean value')
    plt.tight_layout()
    return fig
