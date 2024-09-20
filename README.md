# Time Series Transformer

Still under development & testing...

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

# Citation
Implementation of the deep learning architecture is based on this [paper](https://arxiv.org/abs/2310.03589):
```
@misc{garza2023timegpt1,
      title={TimeGPT-1},
      author={Azul Garza and Max Mergenthaler-Canseco},
      year={2023},
      eprint={2310.03589},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
