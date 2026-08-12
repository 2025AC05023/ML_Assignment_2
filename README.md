# ML Assignment 2: Bank Marketing Classification

## Problem Statement

The objective of this assignment is to predict whether a bank customer will subscribe to a term deposit based on demographic and campaign-related attributes. This is framed as a binary classification problem, where five supervised machine learning algorithms are trained, evaluated, and compared using standard performance metrics.

## Dataset Description

| Property | Details |
|---|---|
| Dataset | UCI Bank Marketing Dataset |
| Total records | 45,211 |
| Input features | 16 |
| Target variable | `y` (yes / no) |
| Classification type | Binary Classification |

The dataset contains information collected from direct marketing campaigns conducted by a Portuguese banking institution. Each record corresponds to a single client contact, and the target variable indicates whether the client subscribed to a term deposit.

Prior to model training, the dataset was subjected to a standard preprocessing pipeline. Categorical variables were encoded using appropriate encoding schemes, and numerical features were scaled where required. The processed dataset was partitioned into training and testing subsets using an 80:20 stratified split to preserve the original class distribution across both partitions.

## Machine Learning Models

The following classification algorithms were implemented and evaluated:

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

| Model | Accuracy | ROC-AUC | F1 Score |
|---|---:|---:|---:|
| Logistic Regression | 90.1% | 0.906 | 0.452 |
| Decision Tree | 87.5% | 0.701 | 0.470 |
| KNN | 89.9% | 0.850 | 0.434 |
| Gaussian Naive Bayes | 85.5% | 0.810 | 0.456 |
| Random Forest | 90.7% | 0.929 | 0.509 |

## Observations

- **Logistic Regression** achieved 90.1% accuracy and a ROC-AUC of 0.906, performing well while remaining interpretable and computationally efficient.
- **Decision Tree** is straightforward to interpret but recorded the lowest ROC-AUC (0.701), indicating weaker discrimination between classes despite a reasonable accuracy of 87.5%.
- **KNN** produced competitive accuracy (89.9%) but yielded the lowest F1-score (0.434) among the five models, suggesting reduced precision or recall on the minority class.
- **Gaussian Naive Bayes** trained quickly and is computationally inexpensive; however, it achieved comparatively lower accuracy (85.5%) and ROC-AUC (0.810).
- **Random Forest** recorded the highest accuracy (90.7%) and ROC-AUC (0.929), as well as the highest F1-score (0.509), making it the best-performing model in this experiment.

## Best Performing Model

**Random Forest** was selected as the best-performing model based on the following results:

| Metric | Value |
|---|---|
| Accuracy | 90.7% |
| ROC-AUC | 0.929 |
| F1 Score | 0.509 |

## Conclusion

Five machine learning classification models were implemented and evaluated on the UCI Bank Marketing Dataset. Models were assessed using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Matthews Correlation Coefficient (MCC). Based on the experimental evaluation, Random Forest achieved the best overall performance among the five classifiers and was selected as the final prediction model for this study. A Streamlit application was developed to support interactive prediction and model evaluation.

## Future Scope

Future work may explore hyperparameter tuning to further optimise individual model performance, along with systematic feature selection to reduce dimensionality and improve generalisation. Ensemble optimisation strategies, such as stacking or boosting configurations, present additional avenues for investigation. Given the class imbalance present in the dataset, techniques such as SMOTE or class-weight adjustment could be incorporated to improve recall on the minority class.

## GitHub Repository

(Add GitHub repository URL here)

## Streamlit Application

(Add deployed Streamlit URL here)

---

## Project Structure

```text
ML_Assignment_2/
├── app/
│   └── streamlit_app.py
├── artifacts/
├── data/
│   └── bank.csv
├── models/
├── notebooks/
├── reports/
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
├── .gitignore
├── README.md
├── requirements.txt
└── run_training.py
```

## Setup and Installation

1. Create virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Place `bank.csv` inside the `data` folder.

## Training

```powershell
python -m src.train --data-path data/bank.csv
```

Or run:

```powershell
python run_training.py
```

Generated outputs:

- Models in `models/`
- Artifacts in `artifacts/`
  - `preprocessor.joblib`
  - `target_encoder.joblib`
  - `feature_columns.joblib`
- Reports in `reports/`
  - `eda_overview.json`
  - `missing_values_report.csv`
  - `numeric_summary.csv`
  - `categorical_summary.csv`
  - `model_metrics.csv`
  - `model_comparison.csv`
  - `detailed_model_results.json`

## Streamlit Application (Local)

```powershell
streamlit run app/streamlit_app.py
```

## Requirements

All dependencies are listed in `requirements.txt`. Install them using:

```powershell
pip install -r requirements.txt
```

## Notes

- Dataset is expected to be semicolon-separated (`;`) and include target column `y`.
- For consistent behavior, train models before using the Streamlit app.
