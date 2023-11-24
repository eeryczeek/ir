

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from scipy.sparse import vstack


class Visualizator:

    def generate_combined_wordcloud(self, feature_names, article_name, all_article_vectorized_data):
        """
        Generates a word cloud based on TF-IDF vectors.
        """

        all_data = vstack(all_article_vectorized_data)
        sums = np.array(all_data.sum(axis=0)).ravel()

        word_weights = {feature_names[j]: sums[j]
                        for j in range(sums.shape[0])}

        wordcloud = WordCloud(
            width=800, height=400, background_color='white').generate_from_frequencies(word_weights)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f'{article_name}')
        plt.savefig(f'plots/{article_name}.png')

    def generate_wordcloud(self, feature_names, article_name, vectorized_data):
        """
        Generates a word cloud based on TF-IDF vectors.
        """
        word_weights = {feature_names[j]: vectorized_data[0, j]
                        for j in range(vectorized_data.shape[1])}
        wordcloud = WordCloud(
            width=800, height=400, background_color='white').generate_from_frequencies(word_weights)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f'{article_name} wordcloud')
        plt.savefig(f'plots/wordclouds/{article_name}.png')

    def generate_combined_heatmap(self, feature_names, article_name, vectorized_data):
        """
        Generates a heatmap based on similarity matrix. make it a bit more readable.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(vectorized_data, cmap='hot', interpolation='nearest')
        plt.title(f'{article_name} heatmap')
        plt.savefig(f'plots/{article_name}.png')

    def generate_scores_histogram(self, scores, plot_name, log=False):
        """
        Generates a histogram based on similarity scores.
        """
        plt.figure(figsize=(10, 5))

        plt.hist(scores, bins='fd', log=log)
        plt.title(f'{plot_name} histogram')
        plt.savefig(f'plots/{plot_name}.png')

    def generate_radar_chart(self, labels, query_values, article_values, chart_name):
        """
        Generate a radar chart with stacked values based on TF-IDF vectors.
        """
        num_features = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_features,
                             endpoint=False).tolist()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.fill(angles, query_values, color='b', alpha=0.25, label='Query')
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)

        ax.fill(angles, article_values, color='r',
                alpha=0.25, label='Stacked Articles')
        ax.set_yticklabels([])

        plt.title(chart_name)
        plt.legend(loc='upper right')
        plt.savefig(f'plots/{chart_name}.png')

    def visualize_results(self, vectorizer, sorted_ranking, query_keys, query_vals, scrap_keys, scrap_vals, top_n=5, top_cooccurring_n=10):

        self.generate_combined_wordcloud(
            vectorizer.get_feature_names_out(),
            'query_combined_wordcloud',
            query_vals)

        self.generate_combined_wordcloud(
            vectorizer.get_feature_names_out(),
            'scrap_combined_wordcloud',
            scrap_vals)

        self.generate_combined_heatmap(
            vectorizer.get_feature_names_out(),
            'query_combined_heatmap',
            cosine_similarity(query_vals))

        self.generate_combined_heatmap(
            vectorizer.get_feature_names_out(),
            'scrap_combined_heatmap',
            cosine_similarity(scrap_vals))

        query_word_counts = np.array(query_vals.sum(axis=0)).ravel()
        scrap_word_counts = np.array(scrap_vals.sum(axis=0)).ravel()

        self.generate_scores_histogram(
            query_word_counts,
            'query_word_counts_histogram',
            log=True
        )

        self.generate_scores_histogram(
            scrap_word_counts,
            'scrap_word_counts_histogram',
            log=True
        )

        similarities = cosine_similarity(scrap_vals, query_vals)
        similarities = np.mean(similarities, axis=1)
        unique, counts = np.unique(similarities, return_counts=True)
        self.generate_scores_histogram(
            similarities, 'similarity_scores_histogram')

        for article in query_keys:
            self.generate_wordcloud(
                vectorizer.get_feature_names_out(),
                article.split(
                    '/')[-1].split('.')[-1].split('title=')[-1].replace('=', '_').split('&')[:-1][0],
                query_vals[query_keys.index(article)])

        top_n_articles = sorted_ranking[:top_n]
        for article, similarity in top_n_articles:
            self.generate_wordcloud(
                vectorizer.get_feature_names_out(),
                article.split(
                    '/')[-1].split('.')[-1].split('title=')[-1].replace('=', '_').split('&')[:-1][0],
                scrap_vals[scrap_keys.index(article)])

            article_vectorized = scrap_vals[scrap_keys.index(
                article)].toarray().flatten()

        return None
