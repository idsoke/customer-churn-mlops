# Customer Churn MLOps

End-to-end project for predicting customer churn, from data exploration to a served model API.

## Structure

```
customer-churn-mlops/
│
├── data/
│   └── raw/          # Raw, unprocessed datasets
│
├── notebooks/         # Exploratory data analysis and experiments
│
├── src/                # Reusable source code (data processing, training, evaluation)
│
├── models/             # Trained model artifacts
│
├── api/                # Model serving API
│
├── tests/              # Unit and integration tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

- Put raw data files into `data/raw/`.
- Explore data in `notebooks/`.
- Implement reusable pipeline code in `src/`.
- Save trained models to `models/`.
- Serve predictions via the API in `api/`.
- Run tests with `pytest tests/`.
