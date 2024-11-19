import pandas as pd
import json
from myapp.search.objects import Document
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from string import punctuation
from myapp.core.utils import load_json_file

_corpus = {}

# Preprocessing function
def preprocess_text(text):
    """
    Preprocess the tweet text by:
    - Lowercasing
    - Removing URLs
    - Removing hashtags
    - Removing extra spaces and punctuation
    - Tokenizing, removing stopwords, and stemming
    """
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    hashtags = re.findall(r'#\S+', text)  # Extract hashtags
    text = re.sub(r'#\w+', '', text)  # Remove hashtags from main text
    text = text.strip()  # Remove leading and trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]+', '', text)

    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words("english"))  # Get stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  # Apply stemming

    return stemmed_tokens, hashtags

def build_terms(line):
    """
    Preprocess the text by removing stop words, stemming, and tokenizing.

    Arguments:
    line -- string (text) to be preprocessed

    Returns:
    A list of tokens after preprocessing
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    line = line.lower()
    line = re.findall(r'\b\w+\b', line)  # Tokenize the text
    line = [word for word in line if word not in stop_words]  # Remove stopwords
    line = [stemmer.stem(word) for word in line]  # Perform stemming
    return line

import json

def load_corpus(path) -> [Document]:
    preprocessed_tweets = []  
    with open(path, 'r') as file:
        for line in file:
            try:
                tweet = json.loads(line)

                tweet_id = tweet.get('id')
                content = tweet.get('content')
                date = tweet.get('date')
                hashtags = preprocess_text(content)[1]  
                preprocessed_content = preprocess_text(content)[0]
                
                username = tweet.get('user', {}).get('username', '')
                likes = tweet.get('likeCount', 0)
                retweets = tweet.get('retweetCount', 0)
                url = tweet.get('url', '')
                
                # Crear una instancia de la clase Document
                document = Document(
                    tweet_id=tweet_id,
                    preprocessed_content=preprocessed_content,
                    date=date,
                    hashtags=hashtags,
                    likes=likes,
                    retweets=retweets,
                    url=url,
                    username='@' + username
                )
                
                preprocessed_tweets.append(document)  # Agregar el documento a la lista

            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                
    return preprocessed_tweets

