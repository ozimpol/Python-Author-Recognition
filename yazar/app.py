import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_excel('yazar.xlsx')
text_data = data['Kose_Yazisi']
text_data = text_data.apply(lambda x: x.lower())
labels = data['Yazar']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

models = [
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Support Vector Machine", SVC(kernel='linear')),
    ("Random Forest", RandomForestClassifier()),
    ("Neural Network", MLPClassifier(max_iter=1000)),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier())
]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"{name} Model Performansı:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("------")
