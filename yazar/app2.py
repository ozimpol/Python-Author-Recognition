import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data = pd.read_excel('yazar.xlsx')

X_train, X_test, y_train, y_test = train_test_split(data['Kose_Yazisi'], data['Yazar'], test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0],
    'fit_prior': [True, False]
}
grid_nb = GridSearchCV(MultinomialNB(), param_grid_nb, scoring='accuracy', cv=3)
grid_nb.fit(X_train_tfidf, y_train)

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}
grid_svm = GridSearchCV(SVC(), param_grid_svm, scoring='accuracy', cv=3)
grid_svm.fit(X_train_tfidf, y_train)

param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [2000, 3000, 4000] 
}
grid_nn = GridSearchCV(MLPClassifier(), param_grid_nn, scoring='accuracy', cv=3)
grid_nn.fit(X_train_tfidf, y_train)

print("Multinomial Naive Bayes En İyi Parametreler:", grid_nb.best_params_)
print("Multinomial Naive Bayes En İyi Skor:", grid_nb.best_score_)

print("Support Vector Machine En İyi Parametreler:", grid_svm.best_params_)
print("Support Vector Machine En İyi Skor:", grid_svm.best_score_)

print("Neural Network En İyi Parametreler:", grid_nn.best_params_)
print("Neural Network En İyi Skor:", grid_nn.best_score_)
