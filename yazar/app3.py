from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QMessageBox
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

data = pd.read_excel('yazar.xlsx')  

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('turkish'))  
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

data['Kose_Yazisi'] = data['Kose_Yazisi'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Kose_Yazisi'])
y = data['Yazar']

best_params = {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (50,), 'max_iter': 4000, 'solver': 'adam'}
model = MLPClassifier(**best_params)
model.fit(X, y)

class TextAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Yazar Tahmini Uygulaması')
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        self.predict_button = QPushButton('Tahmin Yap')
        self.predict_button.clicked.connect(self.predict_author)
        layout.addWidget(self.predict_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def predict_author(self):
        input_text = self.text_edit.toPlainText()
        input_vector = vectorizer.transform([input_text])

        prediction_proba = model.predict_proba(input_vector)
        max_proba = prediction_proba.max()

        threshold = 0.6

        if max_proba >= threshold:
            predicted_class = model.classes_[prediction_proba.argmax()]
            QMessageBox.information(self, 'Tahmin Sonucu', f"Metin, {predicted_class} yazarına ait olabilir. (%{max_proba*100:.2f})")
        else:
            max_class = model.classes_[prediction_proba.argmax()]
            max_class_proba = prediction_proba.max()
            QMessageBox.warning(self, 'Tahmin Sonucu', f"Yeterince yüksek güvenilirlikte bir yazar tespit edilemedi. En olası yazar: {max_class} (%{max_class_proba*100:.2f})")

if __name__ == '__main__':
    app = QApplication([])
    window = TextAnalyzer()
    window.show()
    app.exec_()
