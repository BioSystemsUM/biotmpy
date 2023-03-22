from src.biotmpy.data_structures import Document
from src.biotmpy.data_structures.sentence import Sentence
from src.biotmpy.data_structures import Token
from src.biotmpy.data_structures.relevance import Relevance
import nltk
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import re
import string
from nltk.stem import WordNetLemmatizer, PorterStemmer

from wrappers_utils import *


def tsv_to_docs(file, stop_words=None, lower=False, remove_punctuation=False, split_by_hyphen=True, lemmatization=False,
                stems=False, config=None, sep='\t'):
    """
    Converts a tsv file to a list of Document objects. The tsv file must contain an id column, a title column, an abstract column and a label column.

    :param file: path to the file
    :param stop_words: list of stop words
    :param lower: boolean, if True, all words are converted to lower case
    :param remove_punctuation: boolean, if True, all punctuation is removed
    :param split_by_hyphen: boolean, if True, words are split by hyphen and the hyphen is removed
    :param lemmatization: boolean, if True, words are lemmatized. For more details see https://www.nltk.org/book/ch03.html
    :param stems: boolean, if True, words are stemmed. For more details see https://www.nltk.org/book/ch03.html
    :param config: Deep Learning Configuration object
    :param sep: separator of the tsv file

    :return: list of Document objects
    """
    if config:
        stop_words = config.stop_words
        lower = config.lower
        remove_punctuation = config.lower
        split_by_hyphen = config.split_by_hyphen
        lemmatization = config.lemmatization
        stems = config.stems

    dataframe = tsv_file_reader(file, sep=sep)
    docs = []
    for i, df_row in dataframe.iterrows():
        document = get_document(df_row, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems)
        docs.append(document)
    return docs


def tsv_to_relevances(file, sep='\t', description=None):
    """
    Converts a tsv file to a list of Relevance objects.

    :param file: path to the file
    :param sep: separator of the tsv file
    :param description: description of the relevance
    """
    try:
        dataframe = tsv_file_reader(file)
        relevances = []
        for i, df_row in dataframe.iterrows():
            relevance = Relevance(df_row["label"], df_row['pmid'], description)
            relevances.append(relevance)
    except:
        print('ERROR: Your file %s does not contain documents with an associated relevance.' % file)
    else:
        return relevances


def tsv_file_reader(file, sep='\t'):
    """
    Reads a tsv file and returns a dataframe.

    :param file: path to the file
    :param sep: separator of the tsv file

    :return: dataframe
    """
    dataframe = pd.read_csv(file, sep=sep)
    return standardize_headers(dataframe)


def standardize_headers(dataframe):
    """
    Standardizes the headers of a dataframe by converting them to lower case.

    :param dataframe: dataframe

    :return: dataframe
    """
    dataframe.columns = dataframe.columns.str.lower()
    return dataframe


if __name__ == '__main__':
    import sys

    sys.append('../')