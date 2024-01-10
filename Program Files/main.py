import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Import the dataset 
df = pd.read_csv("processed.cleveland2.csv")
X = df.iloc[:, :-1].values 
y = np.where(df.iloc[:, -1] > 0, 1, 0) # Change to a binary classification problem 

# Split into training/test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123) 

# Feature Scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create a function to return accuracy and confusion matrix for given model
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, conf_matrix, f1

# List of models
models = [
    LogisticRegression(random_state = 123),
    RandomForestClassifier(n_estimators = 100, criterion = "entropy", random_state = 0),
    SVC(kernel = "linear", random_state = 0)
]

results = [] 

if __name__ == "__main__": 
    # Loop through models
    for model in models:
        accuracy, conf_matrix, f1 = train_evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append((model.__class__.__name__, accuracy, f1))
        print(f"Model: {model}\nConfusion Matrix:\n{conf_matrix}\nAccuracy: {accuracy}\nF1 Score: {f1}")

    # Plotting 
    model_names, accuracies, f1_scores = zip(*results)

    plt.figure(figsize=(10, 5))

    # Accuracy 
    plt.subplot(1, 2, 1)
    plt.bar(model_names, accuracies, color='red')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')

    # F1 Score 
    plt.subplot(1, 2, 2)
    plt.bar(model_names, f1_scores, color='blue')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Model F1 Scores')

    plt.tight_layout()
    plt.show()