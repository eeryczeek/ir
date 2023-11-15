from scraper import WikiArticleGetter
from text_processor import Preprocessor

if __name__ == "__main__":
    wikiScraper = WikiArticleGetter()
    wikiScraper.retrieve_wiki_articles(3)

    preprocessor = Preprocessor()
    preprocessor.process_articles(wikiScraper.dict_of_pages)

    print(preprocessor.processed_articles)
