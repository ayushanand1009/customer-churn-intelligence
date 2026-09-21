# Customer Churn Intelligence

An end-to-end machine learning platform for customer churn prediction, risk prioritization, batch inference, and explainable customer-level predictions.

## Overview

Customer churn is a major business problem where organizations need to identify customers who are likely to discontinue their services.

This project develops a complete machine learning workflow that predicts customer churn probability and converts model predictions into actionable risk categories.

The platform combines:

- Machine learning based churn prediction
- Probability-based risk assessment
- Optimized decision threshold
- Customer-level prediction
- Batch CSV prediction
- SHAP-based explainability
- Model performance analytics
- Global feature importance
- Streamlit interactive dashboard

The final application uses **CatBoost** as the deployed prediction model.

---

## Key Features

### 1. Customer Churn Prediction

Predicts whether an individual customer is likely to churn using the trained CatBoost model.

The system provides:

- Churn probability
- Churn / No Churn prediction
- Risk level
- Customer-level prediction details

### 2. Single Customer Prediction

The application allows users to enter customer information and obtain an individual churn prediction.

The prediction interface provides:

- Churn probability
- Risk classification
- Final prediction
- Feature-level explanation

### 3. Batch Prediction

Users can upload a CSV containing customer information.

The system automatically generates:

- Churn predictions
- Churn probabilities
- Risk levels
- Batch-level statistics
- Downloadable prediction results

### 4. Explainable AI

The platform uses SHAP-based explanations to identify which customer attributes contributed to an individual prediction.

For each customer, the application can show:

- Feature
- Feature value
- SHAP value
- Impact
- Direction of influence

This helps users understand why a customer was classified as a churn risk.

### 5. Model Analytics

The platform provides model evaluation metrics including:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

It also provides global feature importance and model selection information.

### 6. Risk Prioritization

Predicted churn probabilities are converted into risk categories:

- Low Risk
- Medium Risk
- High Risk

This allows customer records to be prioritized for retention analysis.

---

## Machine Learning Pipeline

The project follows an end-to-end machine learning workflow:

```text
Customer Dataset
       |
       v
Data Preprocessing
       |
       v
Feature Preparation
       |
       v
Model Training
       |
       +------------------+
       |                  |
       v                  v
   XGBoost             CatBoost
       |                  |
       +--------+---------+
                |
                v
        Model Comparison
                |
                v
       OOF Validation
                |
                v
          PR-AUC Analysis
                |
                v
       Threshold Optimization
                |
                v
       Final CatBoost Model
                |
                v
        Customer Prediction
                |
        +-------+-------+
        |               |
        v               v
   Risk Scoring     SHAP Explanation
        |
        v
 Streamlit Dashboard
```

---

## Model Selection

The final deployed model is **CatBoost**.

The project uses **PR-AUC** as the primary model-selection metric and evaluates models using out-of-fold predictions.

### Model Selection Strategy

| Component | Method |
|---|---|
| Final Model | CatBoost |
| Primary Metric | PR-AUC |
| Model Selection | OOF PR-AUC |
| Threshold Optimization | OOF F1 Optimization |
| Decision Threshold | 0.540 |

Using out-of-fold predictions for model comparison and threshold selection helps avoid selecting the decision threshold directly from the final evaluation predictions.

---

## Final Model Performance

The deployed CatBoost model produced the following evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 0.7729 |
| Precision | 0.5534 |
| Recall | 0.7487 |
| F1 Score | 0.6364 |
| ROC-AUC | 0.8454 |
| PR-AUC | 0.6621 |

### Interpretation

The model is evaluated using both threshold-dependent and threshold-independent metrics.

ROC-AUC measures ranking performance across classification thresholds, while PR-AUC is particularly useful for evaluating the precision-recall trade-off for churn identification.

The selected decision threshold is:

```text
0.540
```

---

## Explainability

The project integrates SHAP-based feature explanations into the application.

For an individual customer prediction, the system identifies the most influential features and determines whether they:

- Increase churn risk
- Decrease churn risk

Example explanation output:

| Feature | SHAP Value | Feature Value | Direction |
|---|---:|---|---|
| Contract | -1.0374 | Two year | Decreases Churn Risk |
| tenure | -0.9762 | 72 | Decreases Churn Risk |
| InternetService | 0.5527 | Fiber optic | Increases Churn Risk |
| OnlineSecurity | -0.2464 | Yes | Decreases Churn Risk |
| AverageMonthlySpend | 0.2417 | 117.61 | Increases Churn Risk |

This makes the prediction more interpretable than providing only a binary churn label.

---

## Application Modules

The Streamlit application contains the following modules:

### Executive Dashboard

Provides a high-level overview of customer churn predictions and risk distribution.

### Single Customer

Generates a prediction for an individual customer.

### Batch Prediction

Processes multiple customers from an uploaded CSV file.

### Explainability

Provides customer-level SHAP explanations.

### Model Analytics

Displays model performance and global feature importance.

### Model Information

Displays technical information about:

- Model
- Primary metric
- Decision threshold
- Model selection strategy
- Validation methodology

---

## Technology Stack

### Programming

- Python

### Machine Learning

- CatBoost
- XGBoost
- Scikit-learn

### Explainable AI

- SHAP

### Data Processing

- Pandas
- NumPy

### Visualization

- Plotly
- Matplotlib

### Application

- Streamlit

### Model Serialization

- Joblib
- CatBoost model format

---

## Project Structure

```text
customer-churn-intelligence/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── sample_customers.csv
│
├── models/
│   ├── catboost_churn_model.cbm
│   ├── catboost_threshold.joblib
│   ├── churn_threshold.joblib
│   ├── feature_metadata.json
│   ├── feature_schema.json
│   ├── model_metadata.json
│   ├── model_performance.json
│   └── xgboost_churn_pipeline.joblib
│
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb
│
├── reports/
│   ├── catboost_threshold_analysis.csv
│   ├── final_customer_risk_predictions.csv
│   ├── final_model_comparison.csv
│   ├── model_comparison.csv
│   └── sample_predictions.csv
│
└── src/
    ├── __init__.py
    └── predictor.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ayushanand1009/customer-churn-intelligence.git
```

Navigate to the project directory:

```bash
cd customer-churn-intelligence
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## Batch Prediction Format

The batch prediction module accepts a CSV containing the customer features required by the model.

A sample input file is provided in:

```text
data/sample_customers.csv
```

After prediction, the application provides a downloadable CSV containing the prediction results.

---

## Prediction Workflow

For each customer, the system performs the following steps:

```text
Customer Features
       |
       v
Feature Validation
       |
       v
Model Inference
       |
       v
Churn Probability
       |
       v
Threshold = 0.540
       |
       +------------------+
       |                  |
       v                  v
   No Churn             Churn
       |
       v
Risk Classification
       |
       v
SHAP Explanation
```

---

## Risk Classification

The application converts predicted churn probabilities into customer risk levels.

```text
Low Risk
Medium Risk
High Risk
```

This enables users to focus attention on customers with higher predicted churn risk.

---

## Model Explainability

The application provides both:

### Local Explainability

Explains why the model produced a prediction for a specific customer.

### Global Explainability

Shows which features are generally important across the dataset.

This provides both customer-level and dataset-level model interpretation.

---

## Results

The final application demonstrates an end-to-end customer churn intelligence workflow covering:

- Model training
- Model comparison
- Out-of-fold evaluation
- PR-AUC based model selection
- Threshold optimization
- Customer-level prediction
- Batch inference
- Risk classification
- SHAP explainability
- Global feature importance
- Interactive visualization

---

## Future Improvements

Potential improvements include:

- Automated model retraining
- Model monitoring
- Data drift detection
- Prediction drift monitoring
- Customer retention recommendation engine
- Cost-sensitive churn optimization
- REST API deployment
- Database integration
- Authentication and role-based access
- Automated experiment tracking
- Cloud deployment
- Production model versioning

This project is intended for educational, portfolio, and research purposes.
