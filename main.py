from scraper import WikiArticleGetter
from text_processor import Preprocessor
from utils import save_wiki_articles, load_wiki_articles, read_query_links, save_ranking
from evaluator import Evaluator
import os
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rank wikipedia articles according to given set of visited articles.')
    parser.add_argument('--visited_articles', type=str,
                        default="inputs.csv", help='Path to file with visited articles.')
    parser.add_argument('--wiki_articles_to_scrap', type=int, default=1000,
                        help='Number of articles to scrap from wikipedia.')
    args = parser.parse_args()
    wikiScraper = WikiArticleGetter()
    preprocessor = Preprocessor()

    if not os.path.exists("wiki_articles.csv"):
        dict_of_articles = wikiScraper.retrieve_wiki_articles(
            args.wiki_articles_to_scrap)
        processed_articles = preprocessor.process_articles(dict_of_articles)
        save_wiki_articles("wiki_articles.csv", processed_articles)

    wiki_articles = load_wiki_articles("wiki_articles.csv")

    if args.visited_articles is not None and os.path.exists(args.visited_articles):
        query_links = read_query_links(args.visited_articles)
        query = wikiScraper.retrieve_wiki_articles(
            len(query_links), query_links)
    else:
        query = wikiScraper.retrieve_wiki_articles(3)
    query = preprocessor.process_articles(query)

    evaluator = Evaluator(query, wiki_articles)
    ranked_scrapped_articles = evaluator.evaluate()
    save_ranking("ranking.csv", ranked_scrapped_articles)

    # print(json.dumps(ranked_scrapped_articles, indent=2))
