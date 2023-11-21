from scraper import WikiArticleGetter
from text_processor import Preprocessor
from utils import save_wiki_articles, load_wiki_articles, read_query_links
from evaluator import Evaluator
import os
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rank wikipedia articles according to given set of visited articles.')
    parser.add_argument('--visited_articles', type=str, default=None, help='Path to file with visited articles.')
    parser.add_argument('--wiki_articles_to_scrap', type=int, default=1000, help='Number of articles to scrap from wikipedia.')
    args = parser.parse_args()

    if not os.path.exists("wiki_articles.csv"):
        wikiScraper = WikiArticleGetter()
        wikiScraper.retrieve_wiki_articles(args.wiki_articles_to_scrap)

        preprocessor = Preprocessor()
        preprocessor.process_articles(wikiScraper.dict_of_pages)

        save_wiki_articles("wiki_articles.csv", preprocessor.processed_articles)

    wiki_articles = load_wiki_articles("wiki_articles.csv")

    wikiScraper = WikiArticleGetter()
    if args.visited_articles is not None and os.path.exists(args.visited_articles):
        query_links = read_query_links(args.visited_articles)
        query = wikiScraper.retrive_given_wiki_articles(query_links)
    else:
        wikiScraper.retrieve_wiki_articles(2)
    preprocessor = Preprocessor()
    preprocessor.process_articles(wikiScraper.dict_of_pages)
    query = preprocessor.processed_articles

    evaluator = Evaluator(query, wiki_articles)
    ranked_scrapped_articles = evaluator.evaluate()
    print(json.dumps(ranked_scrapped_articles, indent=2))
