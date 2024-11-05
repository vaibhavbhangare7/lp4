import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Display the first few rows of the dataset 
print(data.head())

# Check for missing values 
print("Missing values in the dataset:") 
print(data.isnull().sum())

# Split the data into features and target variable 
X = data.drop('Outcome', axis=1) # Features
y = data['Outcome']	# Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

# Create the KNN model
k = 5 # You can adjust this value
knn_model = KNeighborsClassifier(n_neighbors=k)

# Train the model 
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = knn_model.predict(X_test)

# Evaluate the model 
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred)) 
print("\nClassification Report:") 
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy:.2f}")

# Visualize the confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix') 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.show()
