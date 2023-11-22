import pandas as pd


def save_wiki_articles(filename: str, pages: dict) -> None:
    dataframe = pd.DataFrame.from_dict(pages, orient='index')
    dataframe.index.name = 'link'
    dataframe.columns = ['wiki_text']
    dataframe.to_csv(filename)
    return None


def load_wiki_articles(filename: str) -> dict:
    wiki_dict = {}
    dataframe = pd.read_csv(filename)
    for index, row in dataframe.iterrows():
        wiki_dict[row['link']] = row['wiki_text']
    return wiki_dict


def read_query_links(filename: str) -> list:
    return pd.read_csv(filename)['links'].tolist()
