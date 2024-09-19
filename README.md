# Time Series Transformer

# Project Structure:
```
Time-Series-Transformer/
├── data/
│   ├── raw/
│   │   └── PJME_hourly.csv
│   └── processed/
│ 
├── src/
│   ├── configs/
│   │   └── config.yaml
│ 
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│ 
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer_model.py
│ 
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py
│ 
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│ 
│   ├── utils/
│   │   ├── __init__.py
│   │   └── utils.py
│ 
│   └── logs/
│       └── training.log
│ 
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│ 
├── scripts/
│   ├── run_training.sh
│   └── run_evaluation.sh
│ 
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py
│ 
├── .github/
│   └── workflows/
│       └── ci.yml
│ 
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```
