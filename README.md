# Online Shoppers Purchasing Intention Prediction

## Project Overview
This project builds a machine learning classification system to predict whether an online shopper will complete a purchase based on their browsing behavior.
The solution includes data preprocessing, training multiple classification models, evaluating their performance, and deploying the best-performing models using a Streamlit web application.

---

## Dataset
- **Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** UCI Machine Learning Repository
- **Records:** 12,330
- **Features:** 17 input features + 1 target variable
- **Target Variable:** `Revenue` (Boolean: Purchase or No Purchase)

---

## Models Implemented
The following six classification models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

## Model Descriptions

- **Logistic Regression**  
  A linear classification model that estimates the probability of a binary outcome. It serves as a strong baseline due to its simplicity and interpretability.

- **Decision Tree**  
  A tree-based model that splits data using decision rules. It can capture non-linear relationships but may overfit if not controlled.

- **K-Nearest Neighbors (KNN)**  
  A distance-based algorithm that classifies data points based on the majority class of their nearest neighbors.

- **Naive Bayes**  
  A probabilistic classifier based on Bayes’ theorem with an assumption of feature independence. It performs well on imbalanced datasets.

- **Random Forest**  
  An ensemble model that combines multiple decision trees to improve generalization and reduce overfitting. It achieved the best overall performance in this project.

- **XGBoost**  
  A gradient boosting algorithm that builds trees sequentially to correct previous errors. It provides high predictive performance and robustness.


---

## Evaluation Metrics
Each model was evaluated using the following metrics:
- Accuracy
- AUC (ROC)
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

Random Forest and XGBoost achieved the best overall performance based on AUC and MCC.

---

## Project Structure
```
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── online_shoppers_intention.csv
│   └── online_shoppers_intention_Test.csv
│
├── model/
│   ├── trainmodel.py
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── model_results.csv

```

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

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

The application will open in the browser at:
```
http://localhost:8501
```

---

## Conclusion
This project demonstrates a complete end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment using Streamlit.
