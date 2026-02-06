import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset (update path if needed)
df = pd.read_csv(r"C:\Users\omard\OneDrive\Documents\OmarsData\UTD Classwork\CS 4375\Project\Machine Learning Project\kc_house_data.csv")

# Drop irrelevant columns
df.drop(['id', 'date'], axis=1, inplace=True)

# Create price category
bins = [0, 400000, 800000, float('inf')]
labels = ['Low', 'Medium', 'High']
df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)

# Drop original price column (to avoid leakage)
df.drop('price', axis=1, inplace=True)

# Encode target labels
y = LabelEncoder().fit_transform(df['price_category'])
df.drop('price_category', axis=1, inplace=True)

# Feature matrix
X = df.copy()

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True)
}



# Evaluation
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("AUC Score:", auc)

    # ROC Curve (1 vs rest for each class)
    fpr = {}
    tpr = {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
        plt.plot(fpr[i], tpr[i], label=f"Class {i} ({labels[i]})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

