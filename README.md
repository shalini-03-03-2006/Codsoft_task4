# Codsoft_task4
Perfect 📊 — Sales Prediction is a real-world machine learning problem where we predict sales based on influencing factors like advertising spend, audience, or platform.

We’ll use the Advertising Dataset (commonly used for this project). It contains:

TV – advertising spend on TV

Radio – advertising spend on radio

Newspaper – advertising spend on newspapers

Sales – sales of the product (target variable)



---

📝 Step-by-Step: Sales Prediction with Machine Learning (Regression)

1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


---

2. Load Dataset

(You can download a dataset like advertising.csv from Kaggle or UCI ML repository)

# Load dataset
data = pd.read_csv("advertising.csv")

print(data.head())
print(data.info())


---

3. Data Exploration

# Check correlations
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

# Pairplot
sns.pairplot(data, x_vars=["TV","Radio","Newspaper"], y_vars="Sales", height=4, aspect=1, kind="scatter")
plt.show()


---

4. Split Features & Target

X = data[["TV", "Radio", "Newspaper"]]  # Features
y = data["Sales"]                       # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


---

5. Train Linear Regression Model

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


---

6. Evaluate Model

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Compare actual vs predicted sales
comparison = pd.DataFrame({"Actual Sales": y_test, "Predicted Sales": y_pred})
print(comparison.head())


---

7. Visualization

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Regression line for TV vs Sales
sns.regplot(x="TV", y="Sales", data=data, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title("TV Advertising vs Sales")
plt.show()


---

✅ End Result:

You’ll get a Linear Regression model that predicts sales from advertising data.

R² score (goodness of fit) usually comes around 0.9+, meaning the model explains most of the sales variance.



---

👉 Do you want me to extend this (Task 4 advanced) using a Random Forest Regressor / Gradient Boosting for better accuracy, instead of just Linear Regression?

