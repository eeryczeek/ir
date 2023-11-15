import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class Preprocessor:
    def __init__(self):
        self.processed_articles = {}

    def preprocess_text(self, text):
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalpha()]

        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words]

        words = [PorterStemmer().stem(WordNetLemmatizer().lemmatize(word))
                 for word in words]
        return " ".join(words)

    def process_articles(self, articles):
        for url, article_text in articles.items():
            self.processed_articles[url] = self.preprocess_text(article_text)
