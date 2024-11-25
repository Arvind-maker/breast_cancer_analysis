from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load the selected features dataset
df = pd.read_csv('selected_data.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the ANN model with the best parameters
model = MLPClassifier(
    hidden_layer_sizes=(50,), 
    activation='tanh', 
    solver='adam', 
    alpha=0.0001, 
    max_iter=500, 
    random_state=42
)

# Train the model
print("Training the ANN model...")
model.fit(X_train, y_train)

# Evaluate the model
print("Model Evaluation:")
y_pred = model.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, 'breast_cancer_model.pkl')
print("Trained model saved as 'breast_cancer_model.pkl'")
