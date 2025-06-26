# linear_regression_housing
A machine learning using Simple and Multiple Linear Regression to predict house prices based on features like square footage, bedrooms, bathrooms, and more. Includes data visualization and model evaluation.
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create or load a DataFrame
# Option 1: Load from a file (uncomment and modify path as needed)
# df = pd.read_csv('your_housing_data.csv')

# Option 2: Create a sample DataFrame for demonstration
# Creating a synthetic dataset with housing features
np.random.seed(42)
n_samples = 500

# Generate synthetic data
sqft_living = np.random.randint(500, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.uniform(1, 4, n_samples).round(1)
floors = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples)

# Price will be a function of the features plus some noise
price = (
    100000 +  # base price
    150 * sqft_living +  # price per sqft
    25000 * bedrooms +  # price per bedroom
    40000 * bathrooms +  # price per bathroom
    30000 * floors +  # price per floor
    np.random.normal(0, 50000, n_samples)  # random noise
)

# Create the DataFrame
df = pd.DataFrame({
    'sqft_living': sqft_living,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'floors': floors,
    'price': price
})

print("\nCorrelation between 'sqft_living' and 'price':")
print(df[['sqft_living', 'price']].corr())

# -----------------------------
# Simple Linear Regression
# -----------------------------
X_simple = df[['sqft_living']]
y = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

print("\n=== Simple Linear Regression ===")
print("Intercept:", round(model_simple.intercept_, 2))
print("Coefficient:", round(model_simple.coef_[0], 2))
print("MAE:", round(mean_absolute_error(y_test_s, y_pred_s), 2))
print("MSE:", round(mean_squared_error(y_test_s, y_pred_s), 2))
print("R²:", round(r2_score(y_test_s, y_pred_s), 2))

# Plotting Simple Linear Regression
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual')
plt.plot(X_test_s, y_pred_s, color='red', label='Predicted')
plt.title("Simple Linear Regression: Price vs Sqft Living")
plt.xlabel("Sqft Living")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Multiple Linear Regression
# -----------------------------
features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors']
X_multi = df[features]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

print("\n=== Multiple Linear Regression ===")
print("Intercept:", round(model_multi.intercept_, 2))
print("Coefficients:")
for feature, coef in zip(features, model_multi.coef_):
    print(f"{feature}: {round(coef, 2)}")
print("MAE:", round(mean_absolute_error(y_test_m, y_pred_m), 2))
print("MSE:", round(mean_squared_error(y_test_m, y_pred_m), 2))
print("R²:", round(r2_score(y_test_m, y_pred_m), 2))
