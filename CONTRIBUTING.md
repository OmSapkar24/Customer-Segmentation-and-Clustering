# Contributing to Customer Segmentation and Clustering

We appreciate your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before submitting a bug report, please:

1. Check if the bug has already been reported in [Issues](https://github.com/OmSapkar24/Customer-Segmentation-and-Clustering/issues)
2. Use a clear and descriptive title
3. Include specific examples to demonstrate the steps
4. Describe the behavior you observed and what you expected
5. Include your environment details (OS, Python version, package versions)

See [.github/ISSUE_TEMPLATE/bug_report.md](.github/ISSUE_TEMPLATE/bug_report.md) for a template.

### Suggesting Enhancements

Enhancement suggestions are welcome! When suggesting enhancements:

1. Use a clear and descriptive title
2. Explain the use case and benefits
3. Describe the current behavior and desired behavior
4. List any similar features in other projects

### Pull Requests

To submit a pull request:

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature/amazing-feature`
3. Make your changes and commit: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request with a clear description

## Development Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/OmSapkar24/Customer-Segmentation-and-Clustering.git
cd Customer-Segmentation-and-Clustering

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools
```

## Development Guidelines

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting

Run formatting before committing:

```bash
black src/ tests/
flake8 src/ tests/
isort src/ tests/
```

### Testing

All contributions should include tests:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/

# Run specific test file
python -m pytest tests/test_preprocessing.py
```

### Documentation

- Add docstrings to all functions and classes (Google style)
- Update README.md if adding major features
- Include type hints in function signatures
- Add examples in docstrings for complex functions

### Example of well-documented function:

```python
def process_data(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """Process data using specified method.
    
    This function applies data preprocessing techniques to normalize
    and clean the input dataframe.
    
    Args:
        df: Input DataFrame to process
        method: Processing method ('standard', 'minmax', 'robust')
    
    Returns:
        Processed DataFrame
    
    Raises:
        ValueError: If method is not recognized
    
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3]})
        >>> result = process_data(df, method='standard')
        >>> result.shape
        (3, 1)
    """
    # Implementation
    pass
```

## Commit Guidelines

Write clear, descriptive commit messages:

```
Add feature: Brief description

More detailed explanation of changes if needed.
References issue #123.
```

Commit message format:
- Use imperative mood ("add" not "added" or "adds")
- Start with a capital letter
- Don't end with a period
- Limit subject line to 50 characters
- Reference issues and pull requests liberally

## Project Structure

```
Customer-Segmentation-and-Clustering/
├── data/                    # Data files
├── src/                     # Source code
│   ├── preprocessing.py    # Data preprocessing
│   ├── segmenter.py        # Clustering algorithms
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Visualization functions
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── reports/                # Generated reports
├── main.py                 # CLI entry point
├── requirements.txt        # Package dependencies
└── README.md               # Project README
```

## Questions?

Feel free to open an issue with your question or create a discussion. The maintainers and community will help!

## Acknowledgments

Thank you for contributing to this project! Your efforts help improve the project for everyone.
