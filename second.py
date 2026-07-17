# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the Wine dataset
wine = load_wine()

# Create a DataFrame
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("First five rows:")
print(X.head())

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 4: Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Step 8: Predict a new sample
sample = [[13.2, 2.5, 2.4, 18.0, 100.0, 2.6, 2.8, 0.3, 1.8, 5.5, 1.0, 3.2, 850]]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nPredicted class:", wine.target_names[prediction[0]])