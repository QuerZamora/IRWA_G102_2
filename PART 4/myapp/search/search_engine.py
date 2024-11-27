import random

from myapp.search.objects import ResultItem, Document
from myapp.search.algorithms import search_in_corpus


def build_demo_results(corpus: dict, search_id,search_query):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    size = len(corpus)

    ll = list(corpus.values())

    for index in range(random.randint(0, 40)):
        item: Document = ll[random.randint(0, size)] #getting a random object in the list of documents

        #if you want to search if the search query is in the quote
        if search_query in item.quote.split():
            res.append(ResultItem(item.id, item.author, item.quote,"doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), random.random()))

        #generate total random results
        #res.append(ResultItem(item.id, item.author, item.quote,random.randint(0, size)))

    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


class SearchEngine:
    """educational search engine"""

    def search(self, search_query, search_id, corpus):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        #results = build_demo_results(corpus, search_id)  # replace with call to search algorithm

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

        return results