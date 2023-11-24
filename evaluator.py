from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

from visualizator import Visualizator


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
        self.visualizator = Visualizator()

    def evaluate(self):
        """
        Evaluate the model on the given data.
        """
        vectorizer, (query_keys, query_vals), (scrap_keys,
                                               scrap_vals) = self.tf_idf()
        similarities = cosine_similarity(scrap_vals, query_vals)

        similarities = np.mean(similarities, axis=1)
        scrapped_files_ranking = {
            url: similarities[idx] for idx, url in enumerate(scrap_keys)}
        sorted_ranking = sorted(
            scrapped_files_ranking.items(), key=lambda x: x[1], reverse=True)

        self.visualizator.visualize_results(vectorizer, sorted_ranking,
                                            query_keys, query_vals, scrap_keys, scrap_vals)

        return sorted_ranking

    def tf_idf(self):
        """
        Implements the TF-IDF algorithm.
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

        return vectorizer, (visited_articles_keys, visited_articles_vectorized), (scrapped_articles_keys, scrapped_articles_vectorized)
