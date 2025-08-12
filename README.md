#  Precision Agriculture

##  Overview

This project uses **Machine Learning** to enhance agricultural decision-making by predicting crop types, estimating yield, and recommending optimal fertilizers. It also integrates **interactive data visualizations** to help farmers and stakeholders understand and act on the insights.

---

##  Features

* **Crop Recommendation** – Suggests the best crop based on soil nutrients, weather, and rainfall.
* **Crop Yield Prediction** – Estimates yield based on historical and environmental data.
* **Fertilizer Recommendation** – Provides suitable fertilizer suggestions based on soil and crop type.
* **Data Visualization** – Interactive dashboards built in Power BI for trend analysis and insights.

---

##  Modules

1. **Crop Recommendation Model**

   * Algorithm: Random Forest Classifier
   * Input: N, P, K (soil nutrients), temperature, humidity, rainfall
   * Preprocessing: Label mapping, MinMaxScaler normalization
   * Hyperparameter Tuning: GridSearchCV

2. **Crop Yield Prediction Model**

   * Algorithm: Decision Tree Regressor
   * Input: Crop type, area, rainfall
   * Preprocessing: Missing value handling, `pd.to_numeric()` conversion, StandardScaler, OneHotEncoder

3. **Fertilizer Recommendation Model**

   * Algorithm: Random Forest Classifier
   * Input: Soil type, crop type
   * Encoding: LabelEncoder
   * Hyperparameter Tuning: GridSearchCV

4. **Data Visualization Module**

   * Tool: Power BI
   * Features: Interactive charts for yield trends, crop patterns, and fertilizer usage

---

## 🛠 Tech Stack

* **Languages**: Python, DAX (Power BI)
* **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, pickle
* **Tools**: Jupyter Notebook, Power BI, Git
* **Models**: Random Forest, Decision Tree

---

##  Workflow

1. **Data Collection & Cleaning** – Handle missing values, normalize data, encode categorical features.
2. **Data Splitting** – Train-test split for robust evaluation.
3. **Model Training** – Use tuned ML algorithms for optimal performance.
4. **Evaluation** – Accuracy, R² score, MAE, MSE.
5. **Deployment Ready Artifacts** – Save models and scalers with `pickle`.
6. **Visualization** – Generate interactive dashboards.

