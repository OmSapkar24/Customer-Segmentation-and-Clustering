"""Unit tests for preprocessing module."""
import unittest
import numpy as np
import pandas as pd
from src.preprocessing import (
    load_data,
    clean_data,
    handle_missing,
    encode_categorical,
    scale_features,
    preprocess_pipeline
)


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature2': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })

    def test_handle_missing_mean(self):
        """Test missing value imputation with mean strategy."""
        df = self.sample_df.copy()
        result = handle_missing(df, strategy='mean')
        self.assertFalse(result.isnull().any().any())
        # Check if NaN was replaced with mean
        expected_mean = self.sample_df['feature1'].mean()
        self.assertEqual(result.loc[2, 'feature1'], expected_mean)

    def test_handle_missing_median(self):
        """Test missing value imputation with median strategy."""
        df = self.sample_df.copy()
        result = handle_missing(df, strategy='median')
        self.assertFalse(result.isnull().any().any())

    def test_handle_missing_drop(self):
        """Test missing value removal."""
        df = self.sample_df.copy()
        result = handle_missing(df, strategy='drop')
        self.assertEqual(len(result), 4)  # One row should be dropped
        self.assertFalse(result.isnull().any().any())

    def test_encode_categorical(self):
        """Test categorical encoding."""
        df = self.sample_df.copy()
        result = encode_categorical(df, columns=['category'])
        # Check if category column is encoded
        self.assertIn('category_A', result.columns)
        self.assertIn('category_B', result.columns)
        self.assertIn('category_C', result.columns)

    def test_scale_features_standard(self):
        """Test standard scaling."""
        df = self.sample_df[['feature1', 'feature2']].copy().dropna()
        scaled, scaler = scale_features(df, method='standard')
        # Check if scaled data has mean ~0 and std ~1
        self.assertAlmostEqual(scaled.mean().mean(), 0, places=5)
        self.assertAlmostEqual(scaled.std().mean(), 1, places=5)

    def test_scale_features_minmax(self):
        """Test min-max scaling."""
        df = self.sample_df[['feature1', 'feature2']].copy().dropna()
        scaled, scaler = scale_features(df, method='minmax')
        # Check if scaled data is between 0 and 1
        self.assertTrue((scaled >= 0).all().all())
        self.assertTrue((scaled <= 1).all().all())

    def test_scale_features_robust(self):
        """Test robust scaling."""
        df = self.sample_df[['feature1', 'feature2']].copy().dropna()
        scaled, scaler = scale_features(df, method='robust')
        self.assertIsNotNone(scaled)
        self.assertEqual(scaled.shape, df.shape)

    def test_clean_data(self):
        """Test data cleaning function."""
        df_with_duplicates = pd.concat([self.sample_df, self.sample_df.iloc[[0]]])
        cleaned = clean_data(df_with_duplicates)
        # Check duplicates are removed
        self.assertLess(len(cleaned), len(df_with_duplicates))

    def test_preprocess_pipeline(self):
        """Test full preprocessing pipeline."""
        df = self.sample_df.copy()
        result = preprocess_pipeline(
            df,
            numeric_cols=['feature1', 'feature2'],
            categorical_cols=['category'],
            missing_strategy='mean',
            scaling_method='standard'
        )
        # Check output is properly processed
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.isnull().any().any())


if __name__ == '__main__':
    unittest.main()
