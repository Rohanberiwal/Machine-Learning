# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a larger dataset
# Creating a dataset with one feature (X) and one target variable (Y)
data = {
    'Feature': np.arange(1, 101),  # Features from 1 to 100
    'Target': np.random.normal(5 * np.arange(1, 101), 10)  # Target variable with some noise
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Step 2: Split the dataset into training and testing sets
X = df[['Feature']]  # Feature(s)
y = df['Target']     # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Implement Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model performance metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Step 6: Visualize the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points', alpha=0.5)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression with Larger Dataset')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.grid()
plt.show()
