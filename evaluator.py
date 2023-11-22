from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
        vectorizer, (query_keys, query_vals), (scrap_keys,
                                               scrap_vals) = self.tf_idf()
        similarities = cosine_similarity(scrap_vals, query_vals)

        similarities = np.max(similarities, axis=1)
        scrapped_files_ranking = {
            url: similarities[idx] for idx, url in enumerate(scrap_keys)}
        sorted_ranking = sorted(
            scrapped_files_ranking.items(), key=lambda x: x[1], reverse=True)

        self.generate_plots(vectorizer, sorted_ranking,
                            query_keys, query_vals, scrap_keys, scrap_vals)

        return sorted_ranking

    def generate_plots(self, vectorizer, sorted_ranking, query_keys, query_vals, scrap_keys, scrap_vals, top_n=5, top_cooccurring_n=10):
        for article in query_keys:
            print(article)
            self.generate_wordcloud(
                vectorizer.get_feature_names_out(),
                article.split(
                    '/')[-1].split('.')[-1].split('title=')[-1].replace('=', '_'),
                query_vals[query_keys.index(article)])
        radar_data = {
            'labels': vectorizer.get_feature_names_out(),
            'query_values': query_vals[query_keys.index(query_keys[0])].toarray().flatten(),
            'article_values': np.zeros(len(vectorizer.get_feature_names_out()))
        }

        top_n_articles = sorted_ranking[:top_n]
        for article, similarity in top_n_articles:
            print(article)
            self.generate_wordcloud(
                vectorizer.get_feature_names_out(),
                article.split(
                    '/')[-1].split('.')[-1].split('title=')[-1].replace('=', '_'),
                scrap_vals[scrap_keys.index(article)])

            article_vectorized = scrap_vals[scrap_keys.index(
                article)].toarray().flatten()
            radar_data['article_values'] += article_vectorized

        radar_data['article_values'] /= len(top_n_articles)

        # Get the indices of the top N co-occurring words
        top_indices = np.argsort(
            radar_data['article_values'])[-top_cooccurring_n:]

        # Extract only the top N co-occurring words and values
        radar_data['labels'] = [radar_data['labels'][i] for i in top_indices]
        radar_data['query_values'] = [radar_data['query_values'][i]
                                      for i in top_indices]
        radar_data['article_values'] = [radar_data['article_values'][i]
                                        for i in top_indices]

        self.generate_radar_chart(
            radar_data['labels'],
            radar_data['query_values'],
            radar_data['article_values'],
            'stacked_radar_chart'
        )

        return None

    def generate_radar_chart(self, labels, query_values, article_values, chart_name):
        """
        Generate a radar chart with stacked values based on TF-IDF vectors.
        """
        num_features = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_features,
                             endpoint=False).tolist()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot the Query
        ax.fill(angles, query_values, color='b', alpha=0.25, label='Query')
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)

        # Plot the Stacked Article values
        ax.fill(angles, article_values, color='r',
                alpha=0.25, label='Stacked Articles')
        ax.set_yticklabels([])

        plt.title(chart_name)
        plt.legend(loc='upper right')
        plt.savefig(f'plots/{chart_name}.png')

    def tf_idf(self):
        """
        Implement the TF-IDF algorithm and generate plots.
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

    def generate_wordcloud(self, feature_names, article_name, vectorized_data):
        """
        Generate a word cloud based on TF-IDF vectors.
        """
        word_weights = {feature_names[j]: vectorized_data[0, j]
                        for j in range(vectorized_data.shape[1])}
        wordcloud = WordCloud(
            width=800, height=400, background_color='white').generate_from_frequencies(word_weights)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f'{article_name}_wordcloud')
        plt.savefig(f'plots/wordcloud_{article_name}.png')
