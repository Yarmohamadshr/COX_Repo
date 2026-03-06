# Customer Churn Prediction (Telecom)

## Business Problem
Telecommunication companies lose revenue when customers cancel their subscriptions. 
Predicting churn helps companies identify at-risk customers and take action to retain them.

## Dataset
Sample telecom dataset with features including:

- tenure
- MonthlyCharges
- Contract type

Target variable:
**Churn** (1 = customer leaves, 0 = stays)

## Project Structure

customer-churn-ml-project
│
├── data
│   └── telecom_churn_sample.csv
│
├── notebooks
│
├── src
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── outputs
│
├── main.py
├── requirements.txt
└── README.md

## Methodology

1. Load dataset
2. Clean and preprocess data
3. Train Random Forest model
4. Evaluate using Accuracy and ROC-AUC

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the pipeline:

python main.py

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- SHAP explainability
- Model deployment with FastAPI
