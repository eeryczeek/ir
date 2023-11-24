import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

for resource_name in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource_name}')
    except:
        nltk.download(resource_name)


class Preprocessor:
    def preprocess_text(self, text):
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalpha()]

        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words]

        words = [PorterStemmer().stem(WordNetLemmatizer().lemmatize(word))
                 for word in words]
        return " ".join(words)

    def process_articles(self, articles):
        processed_articles = {}
        for url, article_text in articles.items():
            processed_articles[url] = self.preprocess_text(article_text)
        return processed_articles
