# 🚓 PatrolIQ - Smart Safety Analytics Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E.svg)

## 📖 Project Overview
**PatrolIQ** is a comprehensive, data-driven urban safety intelligence platform designed for law enforcement agencies. Built using a sample of 500,000 recent records from the Chicago Crime Dataset, this application leverages **Unsupervised Machine Learning** to discover hidden crime patterns, identify geographic hotspots, and optimize police resource allocation.

By transforming complex, high-dimensional crime data into actionable insights, PatrolIQ answers critical daily policing questions such as: *"Where should we patrol tonight?"* and *"Which neighborhoods require specialized resources?"*

## ✨ Key Features & Dashboards

* **🌍 Geographic Analysis (Hotspot Detection):** Utilizes **HDBSCAN** and **K-Means** clustering to pinpoint high-density crime zones. Filters out isolated incidents to map true geographic danger zones and correlates them with arrest rates to guide patrol vs. investigative deployments.
* **🕒 Temporal Pattern Analysis:** Applies cyclical feature engineering (Sine/Cosine transformations) and **K-Means** clustering to group crimes by seasonal shifts and time-of-day spikes, enabling optimized police shift scheduling.
* **📉 Dimensionality Reduction:** Compresses 23 engineered features down to 9 principal components capturing >70% of data variance using **PCA**. Includes high-dimensional **t-SNE** 2D visualizations proving that distinct crime types and districts possess mathematically unique "fingerprints."
* **📊 MLflow Model Registry:** A fully integrated tracking server that logs hyperparameters, clustering evaluation metrics (like Silhouette Score), and manages production-ready model versions for strict accountability.

## 🛠️ Technology Stack
* **Data Processing & Engineering:** Pandas, NumPy, Scikit-Learn
* **Machine Learning (Unsupervised):** K-Means, DBSCAN, Hierarchical Agglomerative, PCA, t-SNE
* **Experiment Tracking:** MLflow (SQLite backend)
* **Data Visualization:** Plotly Express
* **Web Application & Deployment:** Streamlit, Streamlit Community Cloud

## 📂 Project Architecture

```text
PatrolIQ/
│
├── app.py                                  # Main Streamlit application landing page
├── README.md                               # Project documentation
├── requirements.txt                        # Cloud deployment dependencies
├── mlflow.db                               # MLflow SQLite tracking database
│
├── DATA/
│   └── chicago_crime_final_ready.csv       # Preprocessed & Engineered ML-ready dataset
│
├── images/                                 # t-SNE scatterplot visualizations
│   ├── tsne_crime_types.png
│   └── tsne_geography.png
│
└── pages/                                  # Multi-page dashboard modules
    ├── 1_🌍_Geographic_Analysis.py
    ├── 2_🕒_Temporal_Patterns.py
    ├── 3_📉_Dimensionality_Reduction.py
    └── 4_📊_Model_Monitoring.py