"""Unit tests for segmenter module."""
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from src.segmenter import (
    KMeansSegmenter,
    DBSCANSegmenter,
    AgglomerativeSegmenter,
    find_optimal_k
)


class TestKMeansSegmenter(unittest.TestCase):
    """Test cases for KMeansSegmenter."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample clusterable data
        self.X, self.y_true = make_blobs(
            n_samples=100,
            n_features=4,
            centers=3,
            random_state=42
        )

    def test_kmeans_initialization(self):
        """Test KMeans segmenter initialization."""
        segmenter = KMeansSegmenter(n_clusters=3)
        self.assertEqual(segmenter.n_clusters, 3)
        self.assertIsNone(segmenter.model)

    def test_kmeans_fit(self):
        """Test KMeans fitting."""
        segmenter = KMeansSegmenter(n_clusters=3, random_state=42)
        segmenter.fit(self.X)
        self.assertIsNotNone(segmenter.model)
        self.assertEqual(len(segmenter.model.cluster_centers_), 3)

    def test_kmeans_predict(self):
        """Test KMeans prediction."""
        segmenter = KMeansSegmenter(n_clusters=3, random_state=42)
        segmenter.fit(self.X)
        labels = segmenter.predict(self.X)
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(len(np.unique(labels)), 3)

    def test_kmeans_fit_predict(self):
        """Test KMeans fit_predict."""
        segmenter = KMeansSegmenter(n_clusters=3, random_state=42)
        labels = segmenter.fit_predict(self.X)
        self.assertEqual(len(labels), len(self.X))
        self.assertIsNotNone(segmenter.model)

    def test_kmeans_get_centers(self):
        """Test getting cluster centers."""
        segmenter = KMeansSegmenter(n_clusters=3, random_state=42)
        segmenter.fit(self.X)
        centers = segmenter.get_centers()
        self.assertEqual(centers.shape, (3, self.X.shape[1]))


class TestDBSCANSegmenter(unittest.TestCase):
    """Test cases for DBSCANSegmenter."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, _ = make_blobs(
            n_samples=100,
            n_features=4,
            centers=3,
            random_state=42
        )

    def test_dbscan_initialization(self):
        """Test DBSCAN segmenter initialization."""
        segmenter = DBSCANSegmenter(eps=0.5, min_samples=5)
        self.assertEqual(segmenter.eps, 0.5)
        self.assertEqual(segmenter.min_samples, 5)

    def test_dbscan_fit_predict(self):
        """Test DBSCAN fit_predict."""
        segmenter = DBSCANSegmenter(eps=0.5, min_samples=5)
        labels = segmenter.fit_predict(self.X)
        self.assertEqual(len(labels), len(self.X))
        # DBSCAN can have noise points (label -1)
        self.assertTrue(len(np.unique(labels)) >= 1)


class TestAgglomerativeSegmenter(unittest.TestCase):
    """Test cases for AgglomerativeSegmenter."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, _ = make_blobs(
            n_samples=100,
            n_features=4,
            centers=3,
            random_state=42
        )

    def test_agglomerative_initialization(self):
        """Test Agglomerative segmenter initialization."""
        segmenter = AgglomerativeSegmenter(n_clusters=3)
        self.assertEqual(segmenter.n_clusters, 3)

    def test_agglomerative_fit_predict(self):
        """Test Agglomerative fit_predict."""
        segmenter = AgglomerativeSegmenter(n_clusters=3)
        labels = segmenter.fit_predict(self.X)
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(len(np.unique(labels)), 3)


class TestOptimalK(unittest.TestCase):
    """Test cases for optimal k finding."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, _ = make_blobs(
            n_samples=100,
            n_features=4,
            centers=3,
            random_state=42
        )

    def test_find_optimal_k(self):
        """Test finding optimal k using elbow method."""
        optimal_k = find_optimal_k(self.X, k_range=range(2, 6), method='elbow')
        self.assertIsInstance(optimal_k, int)
        self.assertTrue(2 <= optimal_k <= 6)

    def test_find_optimal_k_silhouette(self):
        """Test finding optimal k using silhouette method."""
        optimal_k = find_optimal_k(self.X, k_range=range(2, 6), method='silhouette')
        self.assertIsInstance(optimal_k, int)
        self.assertTrue(2 <= optimal_k <= 6)


if __name__ == '__main__':
    unittest.main()
