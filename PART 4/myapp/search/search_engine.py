import random

from myapp.search.objects import ResultItem, Document
from myapp.search.algorithms import search_in_corpus


class SearchEngine:
    """educational search engine"""
    
    def __init__(self):
        self._last_query = ''
        self._results = []

    def search(self, search_query, search_id, corpus):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        #results = build_demo_results(corpus, search_id)  # replace with call to search algorithm
        
        # Si la consulta es idéntica a la última, devolvemos resultados almacenados en caché.
        if self._last_query == search_query:
            return self._results
        
        else:
            doc_scores = search_in_corpus(search_query, corpus)
            ##### your code here #####
            for score, doc_id in doc_scores:
                document = corpus[doc_id]
                results.append(ResultItem(
                    id=doc_id,
                    title=document.title,
                    tweet=document.tweet,
                    username=document.username,
                    date=document.date,
                    likes=document.likes,
                    retweets=document.retweets,
                    url=document.url,
                    search_id=search_id,
                    ranking=score
                ))
                
        self._last_query = search_query
        self._results = results
        return results
