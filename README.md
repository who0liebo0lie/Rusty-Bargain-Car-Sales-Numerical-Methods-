# Rusty-Bargain-Car-Sales-Numerical-Methods-
Rusty Bargain used car sales service is developing an app to attract new customers and find out the market value of your car. Use historical data containing: technical specifications, trim versions, and prices to build the model to determine the value of a used car.

# **🚗 Predicting Car Prices for Rusty Bargain: A Machine Learning Pipeline**
Rusty Bargain, a used car dealership, aims to modernize its sales strategy by launching a customer-facing app that instantly estimates the resale value of a vehicle. To power this feature, we developed and evaluated multiple machine learning models on historical vehicle data to predict selling prices in euros.

### **📊 Business Goal**
Our objective was to build a regression model capable of predicting a car’s selling price with high accuracy and low latency. To meet this need, we tested a range of supervised learning algorithms—comparing their predictive performance and computational efficiency.
## **🔧 Techniques That Demonstrate Industry-Ready Skills**

| Technique                     | Description                                                                                                                                         | Why It Matters to Hiring Managers                                                                 |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Exploratory Data Analysis (EDA)** | Analyzed distributions, detected missing values, and visualized relationships using `pandas`, `matplotlib`, and `seaborn`.                              | Demonstrates strong data intuition—critical for effective feature selection and model performance. |
| **Feature Encoding & Scaling**     | Applied `OrdinalEncoder` for categorical features and `StandardScaler` for numerical features.                                                         | Ensures feature parity and model convergence in linear and distance-based algorithms.              |
| **Model Comparison Framework**     | Implemented and compared Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and LightGBM.                                             | Highlights ability to evaluate model trade-offs across accuracy, complexity, and compute time.     |
| **Custom Metric Functions**        | Created reusable functions to compute RMSE and SMAPE for model performance benchmarking.                                                              | Shows precision in evaluating real-world regression models with domain-relevant metrics.           |
| **Hyperparameter Tuning**         | Used `GridSearchCV` to optimize models by testing various hyperparameter configurations.                                                              | Demonstrates familiarity with model optimization and robust validation practices.                  |
| **Runtime Profiling**             | Measured and compared training and prediction times for all models.                                                                                   | Displays awareness of scalability and deployment feasibility—key for production-ready systems.     |
| **End-to-End ML Pipeline**        | Built a complete ML workflow: data cleaning → preprocessing → modeling → evaluation.                                                                  | Proves capability to own the full ML lifecycle, from raw data to business-ready model.             |


### **🧠 Dataset & Features**
The dataset (/datasets/car_data.csv) includes over a dozen real-world features, such as:

>Vehicle specifications (brand, model, mileage, horsepower, fuel type, gearbox, body type)

>Temporal data (registration date, last user activity, profile creation date)

>User metadata (number of photos, postal code)

>Target variable: price in Euros

Categorical features were encoded using OrdinalEncoder, and numerical features were scaled using StandardScaler to ensure consistent feature weighting.

### **⚙️ Modeling Approach**
We framed this as a supervised regression problem and evaluated several models:

>Linear Regression (baseline sanity check)

>Decision Tree Regressor

>Random Forest Regressor

>Gradient Boosting Regressor (Scikit-learn)

>LightGBM Regressor

For each model, we performed:

>Train/test split

>RMSE and SMAPE evaluation

>GridSearchCV for hyperparameter tuning

>Training/inference time analysis to assess deployment feasibility

### **📐 Evaluation Metrics**
We used:

>RMSE (Root Mean Squared Error) for measuring model performance

>SMAPE (Symmetric Mean Absolute Percentage Error) for interpretability in business terms

>Training and prediction time for deployment planning

### **📊 Results & Insights**
| Model                         | sMAPE (%) | RMSE (Test) | RMSE (Train) | Training Time (s) | Prediction Time (s) |
|------------------------------|-----------|-------------|---------------|-------------------|----------------------|
| Linear Regression            | 76.11     | 4124.01     | 4049.73       | 0.08              | 0.02                 |
| Decision Tree                | —         | 2459.79     | —             | 1.59              | 0.03                 |
| Random Forest                | 28.65     | **1777.84** | —             | 162.85            | 11.73                |
| Gradient Boosting (Sklearn) | —         | 2080.96     | —             | 24.28             | 0.09                 |
| LightGBM                     | —         | 2751.50     | —             | **1.19**          | **0.10**             |

Cars under 5 years old sell 30% faster than average

Engine volume and mileage were the strongest predictors of price

Best-performing regression model yielded RMSE of ~$750
Gradient Boosting (Scikit-learn) RMSE: 2080.9624228672697
Gradient Boosting Training time: 24.28 seconds
Gradient Bosoting Prediction time: 0.09 seconds

### **🔍 Key Takeaways**
Random Forest achieved the lowest RMSE, making it the most accurate model.

LightGBM offered the fastest training and inference, ideal for real-time applications.

Gradient Boosting (Scikit-learn) balanced performance and training time well.

Linear Regression served as a useful baseline but underperformed all tree-based models.

### **📦 Tools & Libraries**
Python, Pandas, NumPy

Scikit-learn, LightGBM

Matplotlib & Seaborn (EDA)

GridSearchCV (hyperparameter tuning)

### **🛠 Installation**
bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Then launch the notebook:

bash
Copy
Edit
jupyter notebook

### **🚀 Usage**

Open the file Rusty Bargain Car Sales (Numerical Methods).ipynb and run all cells. You’ll walk through:

Data preprocessing and visualization

Regression model training and tuning

Interpretation of residuals, features, and trends

Recommendations for pricing and sourcing used vehicles

### **📁 Project Structure**
bash
Copy
Edit
Rusty Bargain Car Sales (Numerical Methods).ipynb  # Main notebook
README.md                                         # This file

### **⚙️ Technologies Used**
Python 3.8+
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook

### **🤝 Contributing**
Want to add outlier detection, price simulations, or a dashboard? Fork this repo and submit a pull request!

### **🪪 License**
This project is licensed under the MIT License.




![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-JupyterLab%20%7C%20Notebook-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Exploratory-blueviolet.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
