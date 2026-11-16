\# Explainable ML for Predictive Maintenance



An Explainable Machine Learning project for predicting industrial machine failures using the AI4I 2020 Predictive Maintenance dataset.



\## Project Overview



Unexpected machine failures can cause production downtime, maintenance costs, and operational disruptions. This project explores machine learning techniques to predict machine failures using operational sensor measurements and applies explainable AI methods to understand the factors influencing predictions.



The project includes data preprocessing, exploratory data analysis, model training, performance evaluation, feature importance analysis, SHAP explanations, and a Streamlit application.



\## Dataset



\- \*\*Dataset:\*\* AI4I 2020 Predictive Maintenance Dataset

\- \*\*Records:\*\* 10,000

\- \*\*Target:\*\* Machine failure

\- \*\*Features:\*\* Air temperature, process temperature, rotational speed, torque, tool wear, and machine type.



The dataset contains an imbalanced target distribution, making precision, recall, and F1-score important evaluation metrics alongside accuracy.



\## Machine Learning Models



The following models were implemented and evaluated:



\- Logistic Regression

\- Decision Tree

\- Random Forest

\- Gradient Boosting



\## Model Performance



Reported evaluation results:



| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |

|---|---:|---:|---:|---:|---:|

| Gradient Boosting | 98.60% | 88.46% | 67.65% | 76.67% | 0.9698 |

| Random Forest | 98.35% | 92.68% | 55.88% | 69.72% | 0.9619 |

| Decision Tree | 95.50% | 41.91% | 83.82% | 55.88% | — |

| Logistic Regression | 82.45% | — | — | — | — |



\*Note: Metrics are from the reported model evaluation results. A dash indicates a metric not included in the available results.\*



\## Explainable AI



The project uses:



\- \*\*Feature Importance:\*\* To identify influential input variables.

\- \*\*SHAP:\*\* To explain feature contributions to model predictions and provide a global view of model behaviour.



The reported Random Forest feature importance ranking includes:



1\. Torque

2\. Rotational speed

3\. Tool wear

4\. Air temperature

5\. Process temperature

6\. Machine type



\## Technology Stack



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- SHAP

\- Matplotlib and Seaborn

\- Streamlit

\- Joblib



\## Project Structure



```text

Explainable-ML-Predictive-Maintenance/

│

├── data/

│   ├── ai4i2020.csv

│   └── processed/

│

├── models/

│   ├── decision\_tree.joblib

│   ├── gradient\_boosting.joblib

│   ├── logistic\_regression.joblib

│   └── random\_forest.joblib

│

├── model\_code/

├── reports/

├── src/

├── app.py

├── inspect\_models.py

└── .gitignore

```



\## Installation



Clone the repository:



```bash

git clone YOUR\_REPOSITORY\_URL

cd Explainable-ML-Predictive-Maintenance

```



Create and activate a virtual environment:



```bash

python -m venv .venv

```



Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install the required packages:



```bash

pip install pandas numpy scikit-learn matplotlib seaborn shap streamlit joblib

```



\## Run the Application



```bash

streamlit run app.py

```



The application provides an interface for entering machine operating measurements and viewing failure predictions. Email alert functionality requires appropriate email configuration.



\## Important Notes



\- The project uses the AI4I 2020 dataset for model development and evaluation.

\- The dataset is imbalanced, so accuracy should not be interpreted in isolation.

\- Predictions from the application should be treated as experimental model outputs, not as a substitute for industrial safety systems or professional maintenance decisions.

\- Sample prediction reports should be understood as demonstration data where applicable.



\## Future Improvements



\- Evaluate additional machine learning and deep learning models.

\- Improve failure detection recall while controlling false alarms.

\- Extend explainability with local prediction explanations.

\- Integrate validated real-time industrial sensor data.



\## License



Add an appropriate open-source license before permitting reuse of this repository.

