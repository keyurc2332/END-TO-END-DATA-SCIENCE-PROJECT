import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load the dataset
data_path = os.path.join("data", "iris.csv")
df = pd.read_csv(data_path)

# Preprocess the data
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])  # Drop unnecessary Id column
df['Species'] = df['Species'].astype('category').cat.codes  # Encode target labels

# Split dataset into features and target
X = df.drop(columns=['Species'])
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
os.makedirs("model", exist_ok=True)
model_path = os.path.join("model", "iris_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
