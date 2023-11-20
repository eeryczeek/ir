from scraper import WikiArticleGetter
from text_processor import Preprocessor
from utils import save_wiki_articles, load_wiki_articles

if __name__ == "__main__":
    wikiScraper = WikiArticleGetter()
    wikiScraper.retrieve_wiki_articles(4)

    preprocessor = Preprocessor()
    preprocessor.process_articles(wikiScraper.dict_of_pages)

    save_wiki_articles("wiki_articles.csv", preprocessor.processed_articles)

    wiki_articles = load_wiki_articles("wiki_articles.csv")
