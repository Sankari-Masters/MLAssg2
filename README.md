# Online Shoppers Purchasing Intention Prediction

## a. Problem Statement
The objective of this project is to build and evaluate multiple machine learning
classification models to predict whether an online shopper will generate revenue
(i.e., make a purchase) based on their browsing behavior.

---

## b. Dataset Description
- **Dataset Name:** Online Shoppers Purchasing Intention Dataset  
- **Source:** UCI Machine Learning Repository  
- **Number of Instances:** 12,330  
- **Number of Features:** 17 input features and 1 target variable  
- **Target Variable:** Revenue (0 = No Purchase, 1 = Purchase)

The dataset contains information related to user browsing behavior such as page
durations, bounce rates, exit rates, visitor type, and session characteristics.

---

## c. Models Used and Performance Comparison

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1 Score | MCC  |
|---------------------|----------|-------|-----------|--------|----------|------|
| Logistic Regression | 0.8832   | 0.8653| 0.7640    | 0.3560 | 0.4857   | 0.4696 |
| Decision Tree       | 0.8528   | 0.7290| 0.5237    | 0.5497 | 0.5364   | 0.4492 |
| kNN                 | 0.8678   | 0.7888| 0.6217    | 0.3743 | 0.4673   | 0.4138 |
| Naive Bayes         | 0.7794   | 0.8020| 0.3802    | 0.6728 | 0.4858   | 0.3826 |
| Random Forest       | 0.8998   | 0.9179| 0.7320    | 0.5576 | 0.6330   | 0.5834 |
| XGBoost             | 0.8893   | 0.9161| 0.6698    | 0.5628 | 0.6117   | 0.5505 |

---

## Observations on Model Performance

| ML Model Name        | Observation |
|---------------------|-------------|
| Logistic Regression | Provided a strong baseline with high accuracy but lower recall due to class imbalance. |
| Decision Tree       | Captured non-linear patterns but showed moderate variance and lower AUC. |
| kNN                 | Performance was sensitive to feature scaling and showed moderate predictive power. |
| Naive Bayes         | Achieved high recall but low precision, making it suitable for identifying positive cases. |
| Random Forest       | Delivered the best overall performance with high accuracy, AUC, and MCC due to ensemble learning. |
| XGBoost             | Performed close to Random Forest with strong AUC and balanced precision-recall tradeoff. |

---

## Streamlit Application
The Streamlit web application allows users to:
- Select a trained classification model
- Upload a test CSV file
- View evaluation metrics
- Visualize confusion matrix
- View classification report
- See prediction counts

---

## How to Run the Application

### Install dependencies
```
pip install -r requirements.txt
```

### Run the Streamlit app
```
streamlit run app.py
```

The application will open in the browser at:
http://localhost:8501
