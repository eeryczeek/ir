from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Evaluator:
    """
    Evaluate the model on the given data.
    Parameters:
        visited_articles (dict): A dictionary of visited articles (in this use case an query), key is a url/article title and value should be a document content.
        random_articles (dict): A dictionary of random articles (in this use case a 'train' set), key is a url/article title and value should be a document content.
    """

    def __init__(self, visited_articles: dict, scrapped_articles: dict):
        self.visited_articles = visited_articles
        self.scrapped_articles = scrapped_articles

    def evaluate(self):
        """
        Evaluate the model on the given data.
        """
        (query_keys, query_vals), (scrap_keys, scrap_vals) = self.tf_idf()
        similarities = cosine_similarity(scrap_vals, query_vals)

        similarities = np.max(similarities, axis=1)
        scrapped_files_ranking = {
            url: similarities[idx] for idx, url in enumerate(scrap_keys)}
        return sorted(scrapped_files_ranking.items(), key=lambda x: x[1], reverse=True)

    def tf_idf(self):
        """
        Implement the TF-IDF algorithm.
        """
        vectorizer = TfidfVectorizer()
        scrapped_articles_keys, scrapped_articles_values = zip(
            *self.scrapped_articles.items())
        scrapped_articles_vectorized = vectorizer.fit_transform(
            scrapped_articles_values)

        visited_articles_keys, visited_articles_values = zip(
            *self.visited_articles.items())
        visited_articles_vectorized = vectorizer.transform(
            visited_articles_values)
        return (visited_articles_keys, visited_articles_vectorized), (scrapped_articles_keys, scrapped_articles_vectorized)
