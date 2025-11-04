#!/usr/bin/env python
"""Main CLI script for Customer Segmentation and Clustering pipeline.

This script provides a command-line interface to run the full pipeline:
- Data cleaning and preprocessing
- Clustering with various algorithms
- Evaluation metrics
- Visualization and reporting

Usage:
    python main.py --data data/sample.csv --output results/ --algorithm kmeans --clusters 3
    python main.py --help
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_pipeline
from src.segmenter import KMeansSegmenter, DBSCANSegmenter, AgglomerativeSegmenter
from src.metrics import evaluate_clustering
from src.visualization import reduce_pca, scatter_2d, plot_segment_profiles

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_preprocess(
    data_path: str,
    numeric_cols: list = None,
    categorical_cols: list = None,
    missing_strategy: str = 'mean',
    scaling_method: str = 'standard'
) -> pd.DataFrame:
    """Load and preprocess data.
    
    Args:
        data_path: Path to the data file (CSV format)
        numeric_cols: List of numeric columns
        categorical_cols: List of categorical columns
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop')
        scaling_method: Scaling method ('standard', 'minmax', 'robust')
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    logger.info(f"Preprocessing with numeric_cols={numeric_cols}, categorical_cols={categorical_cols}")
    processed_df = preprocess_pipeline(
        df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        missing_strategy=missing_strategy,
        scaling_method=scaling_method
    )
    logger.info(f"Preprocessed data shape: {processed_df.shape}")
    
    return processed_df


def run_clustering(
    X: np.ndarray,
    algorithm: str = 'kmeans',
    n_clusters: int = 3,
    **kwargs
) -> tuple:
    """Run clustering algorithm.
    
    Args:
        X: Feature matrix
        algorithm: Clustering algorithm ('kmeans', 'dbscan', 'agglomerative')
        n_clusters: Number of clusters
        **kwargs: Additional arguments for the algorithm
    
    Returns:
        Tuple of (labels, model)
    """
    logger.info(f"Running {algorithm} clustering with n_clusters={n_clusters}")
    
    if algorithm == 'kmeans':
        segmenter = KMeansSegmenter(n_clusters=n_clusters, **kwargs)
    elif algorithm == 'dbscan':
        segmenter = DBSCANSegmenter(**kwargs)
    elif algorithm == 'agglomerative':
        segmenter = AgglomerativeSegmenter(n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    labels = segmenter.fit_predict(X)
    logger.info(f"Clustering complete. Found {len(np.unique(labels))} unique clusters")
    
    return labels, segmenter


def evaluate_model(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluate clustering quality.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating clustering quality")
    metrics = evaluate_clustering(
        X,
        labels,
        metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz']
    )
    logger.info(f"Metrics: {metrics}")
    
    return metrics


def save_results(
    output_dir: str,
    X: np.ndarray,
    labels: np.ndarray,
    metrics: Dict[str, float],
    df: pd.DataFrame = None
) -> None:
    """Save clustering results and visualizations.
    
    Args:
        output_dir: Output directory path
        X: Feature matrix
        labels: Cluster labels
        metrics: Evaluation metrics
        df: Original/processed DataFrame
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")
    
    # Save metrics
    metrics_path = output_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save labels
    labels_path = output_path / 'labels.npy'
    np.save(labels_path, labels)
    logger.info(f"Saved labels to {labels_path}")
    
    # Create visualizations
    X_pca, _ = reduce_pca(X, n_components=2)
    
    # PCA plot
    plt.figure(figsize=(10, 8))
    scatter_2d(X_pca, labels=labels, title="Clustering Results (PCA)")
    pca_plot_path = output_path / 'pca_clustering.png'
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PCA plot to {pca_plot_path}")
    
    # Segment profiles plot
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            fig = plot_segment_profiles(df, labels[:len(df)], numeric_cols=numeric_cols)
            profile_plot_path = output_path / 'segment_profiles.png'
            fig.savefig(profile_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved segment profiles plot to {profile_plot_path}")
    
    # Save CSV with labels
    if df is not None:
        result_df = df.copy()
        result_df['cluster'] = labels[:len(df)]
        csv_path = output_path / 'clustered_data.csv'
        result_df.to_csv(csv_path, index=False)
        logger.info(f"Saved clustered data to {csv_path}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Customer Segmentation and Clustering Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data data/sample.csv --output results/ --algorithm kmeans --clusters 3
  python main.py --data data/sample.csv --algorithm dbscan --output results/ --eps 0.5
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to input data file (CSV format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/',
        help='Output directory for results (default: results/)'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['kmeans', 'dbscan', 'agglomerative'],
        default='kmeans',
        help='Clustering algorithm (default: kmeans)'
    )
    parser.add_argument(
        '--clusters',
        type=int,
        default=3,
        help='Number of clusters (default: 3)'
    )
    parser.add_argument(
        '--missing-strategy',
        type=str,
        choices=['mean', 'median', 'drop'],
        default='mean',
        help='Strategy for handling missing values (default: mean)'
    )
    parser.add_argument(
        '--scaling-method',
        type=str,
        choices=['standard', 'minmax', 'robust'],
        default='standard',
        help='Scaling method (default: standard)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    try:
        # Load and preprocess data
        df = load_and_preprocess(
            args.data,
            missing_strategy=args.missing_strategy,
            scaling_method=args.scaling_method
        )
        
        X = df.values
        
        # Run clustering
        labels, segmenter = run_clustering(
            X,
            algorithm=args.algorithm,
            n_clusters=args.clusters,
            random_state=args.random_seed
        )
        
        # Evaluate
        metrics = evaluate_model(X, labels)
        
        # Save results
        save_results(args.output, X, labels, metrics, df)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
