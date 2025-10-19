# 👥 Customer Segmentation & Clustering for Growth

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

Actionable customer segments using K-Means, DBSCAN, and Gaussian Mixture Models to drive personalization, retention, and LTV growth.

## 🎯 Objectives
- Identify high-value segments for targeted campaigns
- Personalize offers based on behavior and demographics
- Improve retention and upsell by segment-specific strategies

## 🧠 Approach
1. Data cleaning and feature standardization (scaling, winsorization)
2. EDA with PCA/t-SNE/UMAP for structure discovery
3. Clustering with K-Means, DBSCAN, and GMM
4. Optimal k via silhouette, Davies–Bouldin, elbow
5. Segment profiling and business recommendations

## 📊 Results (example)
- 4 stable segments discovered with silhouette = 0.61
- +18% CTR on personalized campaigns for Segment A
- +12% ARPU uplift via cross-sell in Segment C

## 🛠 Tech Stack
- Python, Pandas, NumPy, Scikit-learn
- Imbalanced-learn, Yellowbrick, UMAP-learn
- Matplotlib, Seaborn, Plotly

## 📦 Installation
```bash
git clone https://github.com/OmSapkar24/Customer-Segmentation-and-Clustering.git
cd Customer-Segmentation-and-Clustering
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Quickstart
```python
import pandas as pd
from src.segmenter import Segmenter

df = pd.read_csv('data/customers.csv')
seg = Segmenter(method='kmeans', n_clusters=4)
labels = seg.fit_predict(df)
profile = seg.profile(df, labels)
print(profile)
```

## 📁 Project Structure
```
Customer-Segmentation-and-Clustering/
├── README.md
├── requirements.txt
├── data/
│   └── customers.csv (example)
├── notebooks/
│   └── segmentation_experiments.ipynb
├── src/
│   ├── preprocessing.py
│   ├── segmenter.py
│   ├── metrics.py
│   └── visualization.py
└── reports/
    └── segment_profiles.png
```

## 🔮 Roadmap
- [ ] Automated k selection and stability analysis
- [ ] RFM and behavior-based hybrid segmentation
- [ ] Real-time segment assignment API
- [ ] SHAP-based segment explainability

## 📜 License
MIT License — see LICENSE.

## 👤 Author
Om Sapkar — Data Scientist & ML Engineer  
LinkedIn: https://www.linkedin.com/in/omsapkar1224/  
Email: omsapkar17@gmail.com
