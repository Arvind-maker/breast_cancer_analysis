import pandas as pd

# Load the dataset
df = pd.read_csv('data.csv')

# Display dataset info
print("Dataset Info:")
print(df.info())

# Display the first few rows
print("First 5 Rows:")
print(df.head())

# Handle missing values
print("Checking for missing values...")
print(df.isnull().sum())
df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Save the cleaned dataset
df.to_csv('cleaned_data.csv', index=False)
print("Cleaned dataset saved as 'cleaned_data.csv'.")
