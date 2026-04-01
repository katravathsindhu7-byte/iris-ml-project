# iris_classification.py

# Import libraries
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names  # For readable labels

# Convert to DataFrame for easy plotting
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [target_names[i] for i in y]

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Print predicted vs actual labels
print("Predicted vs Actual:")
for i in range(len(y_test)):
    print(f"Predicted: {target_names[y_pred[i]]}, Actual: {target_names[y_test[i]]}")

# Print accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.2f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.show()

# --- Pairplot of features vs species ---
sns.pairplot(df, hue='species')
plt.savefig("results/pairplot.png")
plt.show()

# --- Feature Importance plot ---
importances = clf.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(8,5))
sns.barplot(x=feature_names, y=importances)
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.show()