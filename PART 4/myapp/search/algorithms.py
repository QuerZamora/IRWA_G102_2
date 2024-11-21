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

    return index, tf, df, idf, title_index

def rank_documents(terms, docs, index, idf, tf, title_index):
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


def search_tf_idf(query, index, idf, tf, title_index):
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
    ranked_docs = rank_documents(query_terms, list(matching_docs), index, idf, tf, title_index)
    return ranked_docs


def search_in_corpus(query, corpus):
    # 1. create create_tfidf_index
    index, tf, df, idf, title_index = create_index_tfidf(corpus)

    # 2. apply ranking
    ranked_docs = search_tf_idf(query, index, idf, tf, title_index)

    return ranked_docs
