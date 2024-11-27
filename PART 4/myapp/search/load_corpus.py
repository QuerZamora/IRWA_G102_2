###############
import pandas as pd
import datetime
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from myapp.core.utils import load_json_file
from myapp.search.objects import Document

_corpus = {}
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove newline characters
    text = text.replace('\\n', '')
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Delete URLs on the tweet because we won't be able to access to them
    text = re.sub(r'\S*https?:\S*', '', text)
    # Remove spaces at first and at the end of a message
    text = text.strip()
    # Remove punctuation
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    # Tokenize the text
    words = text.split()
    # Remove stopwords and apply stemming
    processed_words = [_stemmer.stem(word) for word in words if word not in _stop_words]
    return ' '.join(processed_words)


def build_terms(text):
  # Clean text
  text = clean(text)

  # Tokenize the text to get a list of terms
  text = text.split()

  # Eliminate the stopwords (HINT: use List Comprehension)
  text = [word for word in text if word not in _stop_words]

  # Perform stemming (HINT: use List Comprehension)
  text = [_stemmer.stem(word) for word in text]

  return text

def separate_by_words(input_string):
    words = re.findall(r'[A-Z][a-z]*', input_string)
    if not words:
        return input_string
    return ' '.join(words).lower()


def getHashtagsFromTweet(row):
    """
    Extract hashtags from the content of a tweet.

    Args:
        row (dict): A dictionary representing a tweet row, including the `content` field.

    Returns:
        list: A list of hashtags extracted from the content.
    """
    content = row.get('content', '')
    hashtags = re.findall(r'#\w+', content)
    return [tag[1:] for tag in hashtags]  # Remove the '#' for consistency


def prepare_hashtag_for_text(list_processed_hashtags):
    if not list_processed_hashtags:
        return []
    return ' '.join(list_processed_hashtags).lower().split()


def load_corpus(path) -> [Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    # Load the dataset into a pandas DataFrame
    df = _load_corpus_as_dataframe(path)
    
    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        # Create a Document object and append it to the corpus

        document = Document(
            id=row['id'],
            title=row['content'][0:100],
            tweet=row['content'],
            preprocessed_tweet=build_terms(row['content']) + prepare_hashtag_for_text(getHashtagsFromTweet(row)),
            username='@' + row['user']['displayname'],
            date=row['date'].strftime('%d/%m/%Y %H:%M:%S'),
            hashtags=['#' + tag for tag in getHashtagsFromTweet(row)],  # Use the extracted hashtags
            likes=row['likeCount'],
            retweets=row['retweetCount'],
            url=row['url']
        )

        _corpus[row['id']] = document

    return _corpus

def _load_corpus_as_dataframe(path):
    # Read the JSON file into a DataFrame
    df = pd.read_json(path, lines=True)
    # Return the DataFrame
    return df


