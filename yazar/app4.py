from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

data = pd.read_excel('yazar.xlsx')

X = data['Kose_Yazisi']
y = data['Yazar']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_text(text):

    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('turkish'))  
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

vectorizer = TfidfVectorizer()
X_train_processed = vectorizer.fit_transform(X_train.apply(preprocess_text))
X_test_processed = vectorizer.transform(X_test.apply(preprocess_text))

model_with_preprocessing = MLPClassifier(max_iter=4000, random_state=42)
model_with_preprocessing.fit(X_train_processed, y_train)
y_pred_with_preprocessing = model_with_preprocessing.predict(X_test_processed)
accuracy_with_preprocessing = accuracy_score(y_test, y_pred_with_preprocessing)

vectorizer_no_preprocessing = TfidfVectorizer()
X_train_no_preprocessing = vectorizer_no_preprocessing.fit_transform(X_train)
X_test_no_preprocessing = vectorizer_no_preprocessing.transform(X_test)

model_no_preprocessing = MLPClassifier(max_iter=4000, random_state=42)
model_no_preprocessing.fit(X_train_no_preprocessing, y_train)
y_pred_no_preprocessing = model_no_preprocessing.predict(X_test_no_preprocessing)
accuracy_no_preprocessing = accuracy_score(y_test, y_pred_no_preprocessing)

print(f"Ön İşleme Yapılmış Verilerle Model Doğruluğu: {accuracy_with_preprocessing:.4f}")
print(f"Ön İşleme Yapılmamış Verilerle Model Doğruluğu: {accuracy_no_preprocessing:.4f}")
