'''

import pandas as pd
import json
from myapp.search.objects import Document
from pandas import json_normalize

_corpus = {}

def load_corpus(path) -> [Document]:
    #read all lines
    with open(path,encoding='utf-8') as fp:
        lines = fp.readlines()

    json_string = ''.join(lines)
    # Parse the string into a JSON object
    json_data = json.loads(json_string)
    df = json_normalize(json_data['Tweet'])
    df['id'] = df.index.values
    df.apply(_row_to_doc_dict, axis=1)

    return _corpus


def _row_to_doc_dict(row: pd.Series):
    _corpus[row['Id']] = Document(row['Id'], row['Tweet'][0:100], row['Tweet'], row['Date'], row['Likes'], row['Retweets'],row['Url'], row['Hashtags'])

'''

import pandas as pd
import datetime
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

from myapp.core.utils import load_json_file
from myapp.search.objects import Document

_corpus = {}


def clean(text):
  # Transform to lowercase
  cleanTxt = text.lower()

  # Removing the urls from tweets, starts with https
  cleanTxt = re.sub('https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
  # Removing the entire hashtags, starts with '#'
  cleanTxt = re.sub(r'#\w+\s*', '', cleanTxt)

  # Removing nonalphanumeric
  cleanTxt = re.sub(r'[\W]+', ' ', cleanTxt)
  cleanTxt = re.sub(r'[\_]+', '', cleanTxt)

  return cleanTxt

def build_terms(text):
  stemmer = PorterStemmer()
  stop_words = set(stopwords.words("english"))

  # Clean text
  text = clean(text)

  # Tokenize the text to get a list of terms
  text = text.split()

  # Eliminate the stopwords (HINT: use List Comprehension)
  text = [word for word in text if word not in stop_words]

  # Perform stemming (HINT: use List Comprehension)
  text = [stemmer.stem(word) for word in text]

  return text

def separate_by_words(input_string):
    words = re.findall(r'[A-Z][a-z]*', input_string)

    # If there is only one or none words
    if (words == None or len(words) == 0):
      return input_string

    # If there are 2 or more identified words
    else:
      return ' '.join(words)

'''
def getHashtagsFromTweet(tweet):
  length_hashtags = len(tweet["entities"]["hashtags"]) # number of hashtags used in the tweet
  hashtags = [] #empty hashtag list

  # Loop the list of hashtags used in the tweet
  for i in range(length_hashtags):
    hash = separate_by_words(tweet["entities"]["hashtags"][i]["text"])
    hashtags.append(hash)

  # Return the list of hashtags used
  return hashtags
'''


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


def prepare_hashtag_for_text(list_processed_hasthags):
  # Join the hashtags into a single string and convert to lowercase
  all_hashtags_text = ' '.join(list_processed_hasthags).lower()

  # Split the text into individual words
  words = all_hashtags_text.split()
  return words

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

