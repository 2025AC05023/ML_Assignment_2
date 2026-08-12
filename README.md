# ML Assignment 2: Bank Marketing Classification

## Problem Statement

The objective of this assignment is to predict whether a bank customer will subscribe to a term deposit based on demographic and campaign-related attributes. This is framed as a binary classification problem, where five supervised machine learning algorithms are trained, evaluated, and compared using standard performance metrics.

## Dataset Description

| Property | Details |
|---|---|
| Dataset | UCI Bank Marketing Dataset |
| Total records | 45,211 |
| Input features | 16 |
| Target variable | `y` (binary: yes / no) |
| Classification type | Binary Classification |
| Train/test split | 80:20 stratified split |

The dataset contains information collected from direct marketing campaigns conducted by a Portuguese banking institution. Each record corresponds to a single client contact, and the binary target variable `y` indicates whether the client subscribed to a term deposit.

Prior to model training, the dataset was subjected to a standard preprocessing pipeline. Categorical variables were encoded using appropriate encoding schemes, and numerical features were scaled where required. The processed dataset was partitioned into training and testing subsets using an 80:20 stratified split to preserve the original class distribution across both partitions.

The file `test_data.csv` is included as the test dataset used for the Streamlit demonstration and can be uploaded in the application.

## Machine Learning Models

The following five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

## Technologies Used

| Library | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Model training, preprocessing, and evaluation |
| Matplotlib | Static visualisation |
| Seaborn | Statistical data visualisation |
| Joblib | Model and artefact serialisation |
| Streamlit | Interactive web application |

## Model Comparison

The following results are taken from the generated evaluation reports.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| KNN | 0.8986 | 0.8500 | 0.6257 | 0.3318 | 0.4336 | 0.4070 |
| Gaussian Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.9073 | 0.9291 | 0.6698 | 0.4102 | 0.5088 | 0.4778 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved strong overall accuracy (0.9012) and ROC-AUC (0.9056), with good precision but lower recall on the positive class. |
| Decision Tree | Produced moderate F1 performance (0.4701) but recorded the lowest ROC-AUC (0.7015), indicating weaker class discrimination. |
| KNN | Maintained competitive accuracy (0.8986), but had the lowest recall (0.3318) and F1 score (0.4336) among the five models. |
| Gaussian Naive Bayes | Achieved the highest recall (0.5198), but lower precision (0.4059) and accuracy (0.8548) reduced its overall performance. |
| Random Forest | Delivered the highest accuracy (0.9073), ROC-AUC (0.9291), precision (0.6698), F1 score (0.5088), and MCC (0.4778). |

### Overall Winner

**Random Forest**

Random Forest is the overall winner because it achieved the best values for five of the six reported metrics: Accuracy, ROC-AUC, Precision, F1 Score, and MCC. Gaussian Naive Bayes achieved the highest Recall, but Random Forest provided the strongest overall balance across the required evaluation measures.

## Best Performing Model

**Random Forest** was selected as the best-performing model based on the following results:

| Metric | Value |
|---|---:|
| Accuracy | 0.9073 |
| ROC-AUC | 0.9291 |
| Precision | 0.6698 |
| Recall | 0.4102 |
| F1 Score | 0.5088 |
| MCC | 0.4778 |

## Conclusion

Five machine learning classification models were implemented and evaluated on the UCI Bank Marketing Dataset. Models were assessed using Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Matthews Correlation Coefficient (MCC). Based on the experimental evaluation, Random Forest achieved the best overall performance among the five classifiers and was selected as the final prediction model for this study. A Streamlit application was developed to support interactive prediction and model evaluation.

## Future Scope

Future work may explore hyperparameter tuning to further optimise individual model performance, along with systematic feature selection to reduce dimensionality and improve generalisation. Given the class imbalance present in the dataset, techniques such as SMOTE or class-weight adjustment could be considered in future experiments to improve recall on the minority class.

## GitHub Repository

[ML_Assignment_2 GitHub Repository](https://github.com/2025AC05023/ML_Assignment_2)

## Streamlit Application

[Live Streamlit Application](https://mlassignment2-9mp5rjvgkazsyoov9t8bsg.streamlit.app)

---

## Project Structure

```text
ML_Assignment_2/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   ├── feature_columns.joblib
│   ├── preprocessor.joblib
│   └── target_encoder.joblib
├── data/
│   └── bank.csv
├── models/
│   ├── decision_tree.joblib
│   ├── gaussian_naive_bayes.joblib
│   ├── knn.joblib
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
├── reports/
│   ├── categorical_summary.csv
│   ├── detailed_model_results.json
│   ├── eda_overview.json
│   ├── missing_values_report.csv
│   ├── model_comparison.csv
│   ├── model_metrics.csv
│   └── numeric_summary.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── ML_Assignment2.ipynb
├── README.md
├── requirements.txt
├── run_training.py
└── test_data.csv
```

## Setup and Installation

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the required dependencies.

```powershell
pip install -r requirements.txt
```

3. Ensure `data/bank.csv` is available for training. The included `test_data.csv` file is used as the test dataset for Streamlit upload and demonstration.

## Training

```powershell
python -m src.train --data-path data/bank.csv
```

Or run:

```powershell
python run_training.py
```

Generated outputs are saved in `models/`, `artifacts/`, and `reports/`.

## Streamlit Application (Local)

Run the Streamlit application locally with:

```powershell
streamlit run app/streamlit_app.py
```

Use `test_data.csv` when uploading test data in the Streamlit application.

## Requirements

All dependencies are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

## Notes

- The training dataset is expected to be semicolon-separated (`;`) and include the target column `y`.
- Train the models before using the local Streamlit app if model or artifact files are missing or need to be regenerated.
