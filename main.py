import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Load the Iris dataset from scikit-learn
iris = load_iris()
X = iris.data  # Features
Y = iris.target  # Target (species labels)

#split the data
IrisFlower_X_train, IrisFlower_X_test, IrisFlower_Y_train, IrisFlower_Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Feature scaling
scaler = StandardScaler()
IrisFlower_X_train = scaler.fit_transform(IrisFlower_X_train)
IrisFlower_X_test = scaler.transform(IrisFlower_X_test)

#Train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(IrisFlower_X_train, IrisFlower_Y_train)

#make predictions
IrisFlower_Y_predict = classifier.predict(IrisFlower_X_test)

#evaluate the model
accuracy = accuracy_score(IrisFlower_Y_test, IrisFlower_Y_predict)
confusion = confusion_matrix(IrisFlower_Y_test, IrisFlower_Y_predict)
classification_rep = classification_report(IrisFlower_Y_test, IrisFlower_Y_predict)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{classification_rep}")
