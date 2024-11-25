import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Display dataset columns
print("Columns in dataset:", df.columns.tolist())

# Separate features (X) and target (y)
# Replace 'target' with the actual name of the target column in your dataset
X = df.drop(columns=['diagnosis'])  # Use the actual target column name
y = df['diagnosis']


# Apply SelectKBest to select the top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get the names of the selected features
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features.tolist())

# Save the dataset with selected features
selected_data = pd.DataFrame(X_new, columns=selected_features)
selected_data['target'] = y
selected_data.to_csv('selected_data.csv', index=False)
print("Selected features dataset saved as 'selected_data.csv'.")
