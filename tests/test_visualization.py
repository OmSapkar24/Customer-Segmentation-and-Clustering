"""Unit tests for visualization module."""
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from src.visualization import (
    reduce_pca,
    reduce_tsne,
    reduce_umap,
    scatter_2d,
    plot_segment_profiles
)


class TestDimensionalityReduction(unittest.TestCase):
    """Test cases for dimensionality reduction functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample high-dimensional data
        self.X, self.labels = make_blobs(
            n_samples=100,
            n_features=10,
            centers=3,
            random_state=42
        )

    def test_reduce_pca(self):
        """Test PCA dimensionality reduction."""
        X_reduced, pca = reduce_pca(self.X, n_components=2, random_state=42)
        self.assertEqual(X_reduced.shape, (100, 2))
        self.assertIsNotNone(pca)
        # Check explained variance
        self.assertEqual(len(pca.explained_variance_ratio_), 2)

    def test_reduce_pca_different_components(self):
        """Test PCA with different number of components."""
        X_reduced, pca = reduce_pca(self.X, n_components=3, random_state=42)
        self.assertEqual(X_reduced.shape, (100, 3))

    def test_reduce_tsne(self):
        """Test t-SNE dimensionality reduction."""
        X_reduced = reduce_tsne(self.X, n_components=2, random_state=42)
        self.assertEqual(X_reduced.shape, (100, 2))
        self.assertIsInstance(X_reduced, np.ndarray)

    def test_reduce_tsne_with_kwargs(self):
        """Test t-SNE with additional parameters."""
        X_reduced = reduce_tsne(
            self.X,
            n_components=2,
            random_state=42,
            perplexity=30
        )
        self.assertEqual(X_reduced.shape, (100, 2))

    def test_reduce_umap(self):
        """Test UMAP dimensionality reduction."""
        X_reduced = reduce_umap(self.X, n_components=2, random_state=42)
        self.assertEqual(X_reduced.shape, (100, 2))
        self.assertIsInstance(X_reduced, np.ndarray)

    def test_reduce_umap_with_kwargs(self):
        """Test UMAP with additional parameters."""
        X_reduced = reduce_umap(
            self.X,
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1
        )
        self.assertEqual(X_reduced.shape, (100, 2))


class TestVisualization(unittest.TestCase):
    """Test cases for visualization functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, self.labels = make_blobs(
            n_samples=100,
            n_features=2,
            centers=3,
            random_state=42
        )
        self.df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_scatter_2d_with_labels(self):
        """Test 2D scatter plot with labels."""
        ax = scatter_2d(self.X, labels=self.labels, title="Test Plot")
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_title(), "Test Plot")

    def test_scatter_2d_without_labels(self):
        """Test 2D scatter plot without labels."""
        ax = scatter_2d(self.X, labels=None, title="Test Plot")
        self.assertIsNotNone(ax)

    def test_scatter_2d_custom_palette(self):
        """Test 2D scatter plot with custom color palette."""
        ax = scatter_2d(
            self.X,
            labels=self.labels,
            title="Test Plot",
            palette="viridis"
        )
        self.assertIsNotNone(ax)

    def test_plot_segment_profiles(self):
        """Test segment profile plotting."""
        fig = plot_segment_profiles(
            self.df,
            self.labels[:len(self.df)],
            numeric_cols=['feature1', 'feature2', 'feature3']
        )
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_segment_profiles_auto_detect(self):
        """Test segment profile plotting with auto-detection of numeric columns."""
        fig = plot_segment_profiles(
            self.df,
            self.labels[:len(self.df)]
        )
        self.assertIsNotNone(fig)

    def test_plot_segment_profiles_custom_figsize(self):
        """Test segment profile plotting with custom figure size."""
        fig = plot_segment_profiles(
            self.df,
            self.labels[:len(self.df)],
            figsize=(15, 10)
        )
        self.assertIsNotNone(fig)
        # Check figure size
        self.assertEqual(fig.get_size_inches()[0], 15)
        self.assertEqual(fig.get_size_inches()[1], 10)

    def test_visualization_consistency(self):
        """Test that visualizations are consistent for same input."""
        ax1 = scatter_2d(self.X, labels=self.labels, title="Test")
        ax2 = scatter_2d(self.X, labels=self.labels, title="Test")
        self.assertEqual(ax1.get_title(), ax2.get_title())


if __name__ == '__main__':
    unittest.main()
