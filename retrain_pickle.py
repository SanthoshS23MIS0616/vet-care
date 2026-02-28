import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("data/dog_data_09032022.csv")

# Example: use all columns except target
X = df.drop("target", axis=1)  # make sure column 'target' exists
y = df["target"]

# Train
model = DecisionTreeClassifier()
model.fit(X, y)

# Save pickle compatible with scikit-learn 1.3.0
joblib.dump(model, "model/dogModel1.pkl")
print("✅ New pickle saved successfully!")
