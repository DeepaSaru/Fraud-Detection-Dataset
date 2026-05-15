# Fraud Detection using Machine Learning

# Project Overview

This project focuses on detecting fraudulent financial transactions using Machine Learning, Exploratory Data Analysis (EDA) and behavioural feature engineering techniques.

The goal of the project was not just to build a classification model, but to deeply understand transaction behaviour patterns that may indicate fraud.

Fraud detection is one of the most business-critical applications of Data Science because fraudulent transactions can lead to:

* Financial losses
* Customer trust issues
* Regulatory risks
* Operational disruptions

This project explores how transaction behaviour, login activity, spending patterns and transaction timing can help identify suspicious activities.



# Objectives

* Perform data cleaning and preprocessing
* Conduct Exploratory Data Analysis (EDA)
* Analyse fraud behaviour patterns
* Engineer behavioural fraud indicators
* Handle imbalanced fraud data
* Train and evaluate machine learning models
* Understand why fraud model evaluation goes beyond accuracy



# Key Learning Areas

This project helped strengthen practical understanding of:

* Fraud Analytics
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Imbalanced Classification
* Classification Metrics
* Behavioural Pattern Detection
* Model Evaluation
* Business-focused Machine Learning


# Tech Stack

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| Python           | Core Programming Language            |
| Pandas           | Data Manipulation & Analysis         |
| NumPy            | Numerical Operations                 |
| Matplotlib       | Data Visualization                   |
| Seaborn          | Statistical Visualization            |
| Scikit-learn     | Machine Learning Models              |
| Jupyter Notebook | Development Environment              |
| OpenAI Codex     | AI-assisted coding & experimentation |


# Dataset Features

The dataset contains transaction-level behavioural information such as:

| Feature                | Description                               |
| ---------------------- | ----------------------------------------- |
| transaction_amount     | Amount involved in the transaction        |
| transaction_type       | Type of transaction                       |
| location               | Transaction location                      |
| is_foreign_transaction | Indicates international transactions      |
| device_type            | Device used for transaction               |
| is_new_device          | Indicates new device usage                |
| is_new_location        | Indicates unfamiliar transaction location |
| account_age_days       | Age of customer account                   |
| avg_transaction_amount | Historical average transaction amount     |
| transactions_last_24h  | Number of recent transactions             |
| failed_logins_24h      | Failed login attempts                     |
| time_since_last_txn    | Time gap between transactions             |
| txn_hour               | Hour of transaction                       |
| is_weekend             | Weekend transaction indicator             |
| is_fraud               | Target variable                           |


# Exploratory Data Analysis (EDA)

The project involved detailed Exploratory Data Analysis to understand:

* Fraud vs non-fraud transaction behaviour
* Transaction amount distributions
* Transaction timing patterns
* Device and location-based fraud behaviour
* Correlation between numerical variables
* Behavioural risk indicators

### EDA Techniques Used

- Univariate Analysis
- Bivariate Analysis
- Correlation Heatmaps
- Boxplots
- Countplots
- Fraud Rate Analysis
- Distribution Analysis


# Key Fraud Insights

Some important behavioural observations discovered during analysis:

* Fraudulent transactions showed higher transaction frequency
* Rapid consecutive transactions were strongly associated with fraud
* Fraudulent activity often occurred during unusual transaction hours
* New devices and unfamiliar locations carried higher fraud risk
* Fraud datasets are highly imbalanced, making evaluation more complex


# Feature Engineering

Behavioural feature engineering was one of the most important parts of this project.

Instead of relying only on raw transaction values, additional behavioural indicators were created.

## Engineered Features

### 1. Transaction Deviation

Measures how unusual the current transaction is compared to the user’s historical spending behaviour.

```python
fraud_df['transaction_deviation'] = (
    fraud_df['transaction_amount'] /
    fraud_df['avg_transaction_amount']
)
```

### 2. Rapid Transaction Flag

Identifies transactions occurring within short time intervals.

```python
fraud_df['rapid_txn_flag'] = (
    fraud_df['time_since_last_txn'] < 60
).astype(int)
```

### 3. High Failed Login Activity

Captures suspicious authentication behaviour.

```python
fraud_df['high_failed_login'] = (
    fraud_df['failed_logins_24h'] >= 3
).astype(int)
```

### 4. Night Transaction Indicator

Detects transactions during unusual hours.

```python
fraud_df['night_transaction'] = (
    fraud_df['txn_hour'].between(0, 5)
).astype(int)
```

### Why Feature Engineering Matters

Feature engineering helped the model better understand:

* Unusual customer behaviour
* Suspicious transaction timing
* Transaction bursts
* Behavioural deviations from historical spending

This improved the model’s ability to identify fraud patterns.


# Handling Imbalanced Data

Fraud detection datasets are naturally imbalanced because fraudulent transactions represent only a very small percentage of total transactions.

This creates challenges during model training.

## Stratified Train-Test Split

Used `stratify=y` to preserve class distribution across training and testing datasets.

```python
train_test_split(
    X,
    y,
    stratify=y,
    random_state=42
)
```

### Why Stratification Matters

Without stratification:

* fraud samples may become underrepresented
* evaluation becomes unreliable
* the model may barely learn fraud patterns


## Class Weight Balancing

Used:

```python
class_weight='balanced'
```

This increases the penalty for misclassifying fraudulent transactions.

### Important Learning

`class_weight` is NOT the same as:

* SMOTE
* Oversampling
* Undersampling

### Difference

| Technique     | What It Does                         |
| ------------- | ------------------------------------ |
| class_weight  | Changes model learning penalty       |
| SMOTE         | Generates synthetic minority samples |
| Oversampling  | Duplicates minority samples          |
| Undersampling | Removes majority samples             |

This was an important practical learning during the project.


# Machine Learning Models

Models explored:

* Logistic Regression
* Random Forest Classifier

The project focused on understanding:

* classification performance
* fraud recall
* false positives
* fraud detection sensitivity


# Model Evaluation

One of the biggest learnings from this project was understanding why:

> High accuracy does NOT necessarily mean a good fraud detection model.

Example:

A model predicting every transaction as non-fraud may still achieve:

```text
98% Accuracy
```

But it completely fails to detect fraud.

# Important Evaluation Metrics

Instead of relying only on accuracy, focus was placed on:

| Metric    | Importance                             |
| --------- | -------------------------------------- |
| Recall    | Detect as many fraud cases as possible |
| Precision | Reduce false fraud alerts              |
| F1-Score  | Balance between precision & recall     |
| ROC-AUC   | Overall classification performance     |

### Key Business Understanding

In fraud detection:

* Missing fraud is expensive
* False negatives are critical
* Recall often becomes more important than accuracy


# Project Highlights

- End-to-end fraud analytics workflow
- Behavioural feature engineering
- Advanced EDA & visualization
- Imbalanced classification handling
- Business-focused model evaluation
- Fraud behaviour interpretation


# Use of OpenAI Codex

This project also explored AI-assisted development workflows using OpenAI Codex.

Codex helped accelerate:

* exploratory coding
* debugging
* visualization generation
* experimentation
* workflow optimization

The focus remained strongly on:

* understanding the logic
* validating outputs
* interpreting fraud behaviour patterns
* applying business reasoning

AI-assisted development significantly improved productivity while allowing deeper focus on analytical thinking.


# Future Improvements

Potential future enhancements:

* XGBoost / LightGBM implementation
* Hyperparameter tuning
* SHAP explainability
* Real-time fraud detection pipeline
* Deployment using Flask or FastAPI
* Streamlit dashboard visualization
* Anomaly detection models


# Sample Visualizations

The project includes:

* Correlation heatmaps
* Fraud distribution plots
* Boxplots for fraud behaviour
* Transaction timing analysis
* Feature importance charts
* Behavioural risk visualizations


# Connect With Me

I’m actively exploring:

* Fraud Analytics
* Machine Learning
* Data Science
* Risk Analytics
* Behavioural Analytics

Feel free to connect or discuss interesting fraud detection datasets and ML applications.



