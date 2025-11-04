"""Unit tests for metrics module."""
import unittest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from src.metrics import (
    calculate_silhouette_score,
    calculate_davies_bouldin_score,
    calculate_calinski_harabasz_score,
    calculate_inertia,
    evaluate_clustering,
    stability_score
)


class TestMetrics(unittest.TestCase):
    """Test cases for clustering metrics."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample clusterable data
        self.X, self.y_true = make_blobs(
            n_samples=100,
            n_features=4,
            centers=3,
            cluster_std=0.5,
            random_state=42
        )
        # Fit a simple model for testing
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.labels = kmeans.fit_predict(self.X)

    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        score = calculate_silhouette_score(self.X, self.labels)
        self.assertIsInstance(score, float)
        # Silhouette score should be between -1 and 1
        self.assertTrue(-1 <= score <= 1)
        # For well-separated clusters, should be positive
        self.assertGreater(score, 0)

    def test_silhouette_score_invalid_labels(self):
        """Test silhouette score with invalid labels."""
        # Single cluster should raise an error or return None
        single_label = np.zeros(len(self.X))
        result = calculate_silhouette_score(self.X, single_label)
        # Should handle gracefully (return None or raise exception)
        self.assertTrue(result is None or isinstance(result, float))

    def test_davies_bouldin_score(self):
        """Test Davies-Bouldin score calculation."""
        score = calculate_davies_bouldin_score(self.X, self.labels)
        self.assertIsInstance(score, float)
        # DB score should be non-negative, lower is better
        self.assertGreaterEqual(score, 0)

    def test_calinski_harabasz_score(self):
        """Test Calinski-Harabasz score calculation."""
        score = calculate_calinski_harabasz_score(self.X, self.labels)
        self.assertIsInstance(score, float)
        # CH score should be positive, higher is better
        self.assertGreater(score, 0)

    def test_inertia(self):
        """Test inertia calculation."""
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(self.X)
        inertia = calculate_inertia(kmeans)
        self.assertIsInstance(inertia, float)
        # Inertia should be non-negative
        self.assertGreaterEqual(inertia, 0)

    def test_evaluate_clustering_all_metrics(self):
        """Test comprehensive clustering evaluation."""
        results = evaluate_clustering(
            self.X,
            self.labels,
            metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz']
        )
        self.assertIsInstance(results, dict)
        self.assertIn('silhouette', results)
        self.assertIn('davies_bouldin', results)
        self.assertIn('calinski_harabasz', results)
        # Check all values are numeric
        for value in results.values():
            self.assertIsInstance(value, (int, float))

    def test_evaluate_clustering_single_metric(self):
        """Test evaluation with single metric."""
        results = evaluate_clustering(
            self.X,
            self.labels,
            metrics=['silhouette']
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 1)
        self.assertIn('silhouette', results)

    def test_stability_score(self):
        """Test clustering stability score."""
        # Test stability by running clustering multiple times
        score = stability_score(
            self.X,
            n_clusters=3,
            n_iterations=5,
            random_state=42
        )
        self.assertIsInstance(score, float)
        # Stability score should be between 0 and 1
        self.assertTrue(0 <= score <= 1)

    def test_metrics_consistency(self):
        """Test that metrics are consistent for same input."""
        score1 = calculate_silhouette_score(self.X, self.labels)
        score2 = calculate_silhouette_score(self.X, self.labels)
        self.assertEqual(score1, score2)

    def test_metrics_with_different_clusters(self):
        """Test metrics with different numbers of clusters."""
        # Test with 2 clusters
        kmeans2 = KMeans(n_clusters=2, random_state=42)
        labels2 = kmeans2.fit_predict(self.X)
        score2 = calculate_silhouette_score(self.X, labels2)
        
        # Test with 4 clusters
        kmeans4 = KMeans(n_clusters=4, random_state=42)
        labels4 = kmeans4.fit_predict(self.X)
        score4 = calculate_silhouette_score(self.X, labels4)
        
        # Both should be valid
        self.assertIsInstance(score2, float)
        self.assertIsInstance(score4, float)


if __name__ == '__main__':
    unittest.main()
