import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
data = pd.read_csv("uber.csv")

# 1. Pre-process the dataset
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

# Check for missing values 
missing_values = data.isnull().sum() 
print("Missing values in the dataset:") 
print(missing_values)

# Handle missing values 
data.dropna(inplace=True)

# Ensure there are no more missing values 
missing_values = data.isnull().sum() 
print("Missing values after handling:") 
print(missing_values)

# 2. Identify outliers 
sns.boxplot(x=data["fare_amount"])
plt.title('Fare Amount Outliers') 
plt.show()

# Calculate the IQR for the 'fare_amount' column 
Q1 = data["fare_amount"].quantile(0.25)
Q3 = data["fare_amount"].quantile(0.75)
IQR = Q3 - Q1
# Define thresholds for outlier detection 
lower_bound = Q1 - 1.5 * IQR 
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data_no_outliers = data[(data["fare_amount"] >= lower_bound) & (data["fare_amount"] <= upper_bound)]
 
# Visualize the 'fare_amount' distribution without outliers 
sns.boxplot(x=data_no_outliers["fare_amount"]) 
plt.title('Fare Amount Without Outliers')
plt.show()

# 3. Check the correlation
# Exclude non-numeric columns before calculating correlation 
correlation_matrix = data_no_outliers.select_dtypes(include=[np.number]).corr() 
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm') 
plt.title('Correlation Matrix')
plt.show()

# 4. Implement linear regression and random forest regression models 
X = data_no_outliers[['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
y = data_no_outliers['fare_amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model 
lr_model = LinearRegression() 
lr_model.fit(X_train, y_train)

# Create and train the random forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42) 
rf_model.fit(X_train, y_train)

# 5. Evaluate the models
y_pred_lr = lr_model.predict(X_test) 
y_pred_rf = rf_model.predict(X_test)

# Calculate R-squared (R2) and Root Mean Squared Error (RMSE) for both models 
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Display results
print("Linear Regression - R2:", r2_lr) 
print("Linear Regression - RMSE:", rmse_lr)

print("Random Forest Regression - R2:", r2_rf) 
print("Random Forest Regression - RMSE:", rmse_rf)
