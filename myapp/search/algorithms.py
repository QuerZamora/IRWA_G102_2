'''
from collections import defaultdict, Counter
from array import array
import math
import numpy as np
from numpy import linalg as la
import collections
from myapp.search.load_corpus import build_terms

def create_index_tfidf(documents):
    index = defaultdict(list)
    tf = defaultdict(list)
    df = Counter()
    idf = defaultdict(float)

    total_documents = len(documents)

    for document_id, document in documents.items():
        pp_tt = ' '.join(document.preprocessed_tweet)
        terms = build_terms(pp_tt)

        unique_terms = set(terms)
        term_positions = {term: [document_id, np.array([pos for pos, t in enumerate(terms) if t == term], 'I')] for term in unique_terms}

        doc_length = sum(len(posting[1]) for posting in term_positions.values())
        doc_length_sqrt = math.sqrt(doc_length)

        for term, posting in term_positions.items():
            tf_value = np.round(len(posting[1]) / doc_length_sqrt, 4)
            tf[term].append(tf_value)
            df[term] += 1
            index[term].append(posting)

    for term in df:
        idf[term] = np.round(np.log(total_documents / df[term]), 4)

    return index, tf, df, idf


def rank_documents(terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title

    Returns:
    Print the list of ranked documents
    """

    doc_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)

    query_terms_count = collections.Counter(terms)

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):
        if term not in index:
            continue

        query_vector[termIndex]=(query_terms_count[term] / query_norm) * idf[term]

        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)

    return doc_scores

#s'ha de canviar pel nostre
def search_tf_idf(query, index, idf, tf):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = build_terms(query)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs=[posting[0] for posting in index[term]]

            # docs = docs Union term_docs
            docs = docs.union(set(term_docs))
        except:
            #term is not in index
            pass
    docs = list(docs)
    doc_scores = rank_documents(query, docs, index, idf, tf)
    return doc_scores

def search_in_corpus(query, corpus):
    # 1. create create_tfidf_index
    index, tf, df, idf = create_index_tfidf(corpus)

    # 2. apply ranking
    doc_scores = search_tf_idf(query, index, idf, tf)
    
    return doc_scores 
'''
# Imported and adapted the algorithms elaborated in the prepvious parts of the lab already delivered, stored in the .ipynb file IRWA_2024_PART_3.ipynb


from collections import defaultdict, Counter
from array import array
import math
import numpy as np
from numpy import linalg as la
import collections
from myapp.search.load_corpus import build_terms
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def create_index_tfidf(documents):
    """
    Builds an inverted index, computes term frequencies (TF), 
    document frequencies (DF), and inverse document frequencies (IDF).

    Arguments:
    documents -- dictionary of {document_id: document_content}

    Returns:
    index -- inverted index structure
    tf -- term frequencies
    df -- document frequencies
    idf -- inverse document frequencies
    title_index -- mapping of document_id to its content
    """
    index = defaultdict(list)
    tf = defaultdict(list)  # Term frequencies
    df = defaultdict(int)  # Document frequencies
    title_index = {}  # Mapping document_id to content
    term_count_per_doc = defaultdict(int)  # Term count per document

    total_documents = len(documents)

    # Create index and compute TF and DF
    for document_id, document in documents.items():
        title_index[document_id] = ' '.join(document.preprocessed_tweet)
        terms = build_terms(title_index[document_id])

        current_page_index = defaultdict(list)
        term_frequencies = defaultdict(int)

        # Count term frequencies and positions
        for position, term in enumerate(terms):
            current_page_index[term].append(position)
            term_frequencies[term] += 1

        # Update global index and document frequency
        for term, positions in current_page_index.items():
            index[term].append([document_id, positions])
            df[term] += 1
            term_count_per_doc[document_id] += len(positions)

        # Normalize term frequencies
        norm = math.sqrt(sum(f ** 2 for f in term_frequencies.values()))
        for term, freq in term_frequencies.items():
            tf[term].append(np.round(freq / norm, 4))

    # Compute IDF
    idf = {term: np.round(math.log(total_documents / df[term]), 4) for term in df}

    return index, tf, df, idf #, title_index

def rank_documents(terms, docs, index, idf, tf):
    """
    Ranks documents based on TF-IDF and cosine similarity.

    Arguments:
    terms -- list of query terms
    docs -- list of documents to rank
    index -- inverted index structure
    idf -- inverse document frequencies
    tf -- term frequencies
    title_index -- mapping of document_id to content

    Returns:
    result_docs -- list of ranked document IDs
    """
    doc_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)

    # Query normalization and vector creation
    query_terms_count = Counter(terms)
    query_norm = la.norm(list(query_terms_count.values()))

    for term_index, term in enumerate(terms):
        if term not in index:
            continue

        # Calculate TF-IDF for query
        query_vector[term_index] = (query_terms_count[term] / query_norm) * idf.get(term, 0)

        # Generate document vectors
        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                doc_vectors[doc][term_index] = tf[term][doc_index] * idf[term]

    # Calculate cosine similarity scores
    doc_scores = [(np.dot(cur_doc_vec, query_vector), doc) for doc, cur_doc_vec in doc_vectors.items()]
    doc_scores.sort(reverse=True, key=lambda x: x[0])  # Sort by score

    # Return ranked document IDs
    return doc_scores


def search_tf_idf(query, index, idf, tf):
    """
    Searches for documents containing the query terms and ranks them by TF-IDF.

    Arguments:
    query -- search query string
    index -- inverted index structure
    idf -- inverse document frequencies
    tf -- term frequencies
    title_index -- mapping of document_id to content

    Returns:
    Ranked list of tuples (score, document_id) based on TF-IDF scores.
    """
    query_terms = build_terms(query)
    if not query_terms:
        return []  # No valid query terms

    matching_docs = None

    for term in query_terms:
        try:
            # Get documents containing the term
            current_term_docs = {posting[0] for posting in index[term]}

            if matching_docs is None:
                matching_docs = current_term_docs
            else:
                matching_docs &= current_term_docs  # Intersection of documents
        except KeyError:
            # Term not in index
            return []

    if not matching_docs:
        return []

    # Rank the matching documents
    ranked_docs = rank_documents(query_terms, list(matching_docs), index, idf, tf)
    return ranked_docs


def search_in_corpus(query, corpus):
    # 1. create create_tfidf_index
    index, tf, df, idf = create_index_tfidf(corpus)

    # 2. apply ranking
    ranked_docs = search_tf_idf(query, index, idf, tf)

    return ranked_docs

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove newline characters
    text = text.replace('\\n', '')
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Delete URLs on the tweet because we won't be able to access to them
    text = re.sub(r'\S*https?:\S*', '', text)
    # Remove spaces at first and at the end of a message
    text.strip()
    # Remove punctuation
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    # Tokenize the text
    words = text.split()
    # Remove stopwords and apply stemming
    processed_words = [stemmer.stem(word) for word in words if word not in custom_stopwords]
    return ' '.join(processed_words)
'''

import numpy as np
import collections
from collections import defaultdict
from array import array
import math
from numpy import linalg as la
from myapp.search.objects import ResultItem, Document
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

stemmer = nltk.stem.SnowballStemmer('english')
custom_stopwords = set(stopwords.words('english'))

def search_in_corpus(corpus: dict, query, index, tf, idf, search_option):
    # 1. create create_tfidf_index
    # DONE IN web_app FOR FASTER EXECUTION

    # 2. apply ranking
    if (search_option == "tf-idf"):
        ranked_tweets = search_tf_idf(query, index, tf, idf)
    else:
        corpus_df = pd.DataFrame(columns=["Id", "Title", "Tweet", "Date", "Likes", "Retweets", "Url"])
        for i in range(len(list(corpus.values()))):
            data_to_append = {
                "Id": list(corpus.values())[i].id,
                "Title": list(corpus.values())[i].title,
                "Tweet": list(corpus.values())[i].description,
                "Date": list(corpus.values())[i].doc_date,
                "Likes": list(corpus.values())[i].likes,
                "Retweets": list(corpus.values())[i].retweets,
                "Url": list(corpus.values())[i].url
            }
            corpus_df.loc[len(corpus_df)] = data_to_append
        
        ranked_tweets = our_search(corpus_df, query, index, tf, idf)
        
    return ranked_tweets

def create_index(corpus):
    num_tweets = len(corpus)
    index = defaultdict(list)
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = defaultdict(float)
    for doc_id, row in corpus.items():
        full_tweet = preprocess(row.description)
        tweet_index = {}
        for position, tweet in enumerate(full_tweet.split(" ")):
            try:
                tweet_index[tweet][1].append(position)
            except:
                tweet_index[tweet]=[doc_id, array('I',[position])]            
        norm = 0
        for tweet, posting in tweet_index.items():
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)
        for tweet, posting in tweet_index.items():
            tf[tweet].append(np.round(len(posting[1])/norm,4))
            df[tweet] += 1
        for doc, position in tweet_index.items():
            index[doc].append(position)
        # Compute IDF:
        for term in df:
            idf[term] = np.round(np.log(float(num_tweets/df[term])), 4)
    return index, tf, idf

def search_tf_idf(query, index, tf, idf):

    query_terms = preprocess(query)
    if not query_terms:
        return []  # No valid query terms

    matching_docs = set()
    for each_query in query.split(" "):
        try:
            query_tweet=[posting[0] for posting in index[each_query]]
            if matching_docs is None:
                matching_docs = query_tweet
            else:
                matching_docs &= query_tweet  # Intersection of documents
        except KeyError:
            # Term not in index
            return []
    if not matching_docs:
        return []
    tweets = list(matching_docs)
    ranked_tweets = rank_tweets(query, tweets, index, idf, tf)
    return ranked_tweets

def rank_tweets(terms, tweets, ourindex, idf, tf):
    terms = terms.split(" ")
    tweet_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)
    query_terms_count = collections.Counter(terms)
    query_norm = la.norm(list(query_terms_count.values()))
    for termIndex, term_i in enumerate(terms):
        if term_i not in ourindex:
            continue
        query_vector[termIndex] = query_terms_count[term_i] / query_norm * idf[term_i]
        for row_tweet, (tweet, postings) in enumerate(ourindex[term_i]):
            if tweet in tweets:
                tweet_vectors[tweet][termIndex] = tf[term_i][row_tweet] * idf[term_i]
    tweet_scores=[[np.dot(curTweetVec, query_vector), tweet] for tweet, curTweetVec in tweet_vectors.items() ]
    tweet_scores.sort(reverse=True)
    result_tweets = [x[1] for x in tweet_scores]
    if len(result_tweets) == 0:
        print("No results found, try again")
    return tweet_scores

def our_search(corpus, query, index, tf, idf):
    query_terms = preprocess(query)
    if not query_terms:
        return []  # No valid query terms

    matching_docs = set()
    for each_query in query.split(" "):
        try:
            query_tweet=[posting[0] for posting in index[each_query]]
            if matching_docs is None:
                matching_docs = query_tweet
            else:
                matching_docs &= query_tweet  # Intersection of documents
        except KeyError:
            # Term not in index
            return []
    if not matching_docs:
        return []
    tweets = list(matching_docs)
    ranked_tweets = rank_tweets2(corpus, query, tweets, index, idf, tf)
    return ranked_tweets

def rank_tweets2(df, terms, tweets, index, idf, tf):
    terms = terms.split(" ")
    tweet_vectors = defaultdict(lambda: [0] * len(terms)) 
    query_vector = [0] * len(terms)
    query_terms_count = collections.Counter(terms)
    query_norm = la.norm(list(query_terms_count.values()))
    min_likes = min(np.log(df["Likes"]+1))
    den_likes = max(np.log(df["Likes"]+1)) - min(np.log(df["Likes"]+1))
    min_rt = min(np.log(df["Retweets"]+1))
    den_rt = max(np.log(df["Retweets"]+1)) - min(np.log(df["Retweets"]+1))
    for termIndex, term in enumerate(terms):
        if term not in index:
            continue
        query_vector[termIndex] =  query_terms_count[term] / query_norm * idf[term]
        for row_tweet, (tweet, postings) in enumerate(index[term]):
            if tweet in tweets:
                likes = np.log(df[df["Id"]==tweet]["Likes"]+1).values[0]
                rt = np.log(df[df["Id"]==tweet]["Retweets"]+1).values[0]
                tweet_vectors[tweet][termIndex] = tf[term][row_tweet] * idf[term] + (((likes-min_likes)/den_likes*0.15) + ((rt-min_rt)/den_rt* 0.3))
    tweet_scores=[[np.dot(curTweetVec, query_vector), tweet] for tweet, curTweetVec in tweet_vectors.items() ]
    tweet_scores.sort(reverse=True)
    result_tweets = [x[1] for x in tweet_scores]
    if len(result_tweets) == 0:
        print("No results found, try again")
    return tweet_scores

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove newline characters
    text = text.replace('\\n', '')
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Delete URLs on the tweet because we won't be able to access to them
    text = re.sub(r'\S*https?:\S*', '', text)
    # Remove spaces at first and at the end of a message
    text.strip()
    # Remove punctuation
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    # Tokenize the text
    words = text.split()
    # Remove stopwords and apply stemming
    processed_words = [stemmer.stem(word) for word in words if word not in custom_stopwords]
    return ' '.join(processed_words)
'''