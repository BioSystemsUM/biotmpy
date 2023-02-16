from src.biotmpy.data_structures import Document
from src.biotmpy.data_structures.sentence import Sentence
from src.biotmpy.data_structures import Token
from src.biotmpy.data_structures.relevance import Relevance
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import re
import string
from nltk.stem import WordNetLemmatizer, PorterStemmer

from wrappers_utils import *


def dictionary_to_docs(file, stop_words=None, lower=False, remove_punctuation=False, split_by_hyphen=True,
                       lemmatization=False, stems=False, dl_config=None):
    """
    Converts a dictionary file to a list of Document objects with the sentences and tokens.

    :param file: path to the file
    :param stop_words: list of stop words. For more details see https://www.nltk.org/book/ch02.html
    :param lower: boolean, if True, all words are converted to lower case
    :param remove_punctuation: boolean, if True, all punctuation is removed
    :param split_by_hyphen: boolean, if True, words are split by hyphen and the hyphen is removed
    :param lemmatization: boolean, if True, words are lemmatized. For more details see https://www.nltk.org/book/ch03.html
    :param stems: boolean, if True, words are stemmed. For more details see https://www.nltk.org/book/ch03.html
    :param dl_config: DLConfig object with the configuration for the deep learning model

    :return: list of documents
    """
    if dl_config:
        stop_words = dl_config.stop_words
        lower = dl_config.lower
        remove_punctuation = dl_config.lower
        split_by_hyphen = dl_config.split_by_hyphen
        lemmatization = dl_config.lemmatization
        stems = dl_config.stems

    data_dict = txt_file_reader(file)
    docs = []
    for doc_id in data_dict.keys():
        document = get_document_dictionary(data_dict[doc_id],
                                           doc_id=doc_id,
                                           stop_words=stop_words,
                                           lower=lower,
                                           remove_punctuation=remove_punctuation,
                                           split_by_hyphen=split_by_hyphen,
                                           lemmatization=lemmatization,
                                           stems=stems)
        docs.append(document)
    return docs


def dictionary_to_relevances(file, topic=None):
    """
    Converts a dictionary file to a list of Relevance objects. The dictionary must have the following structure: {'Label': 'relevance', 'title': 'title_text', 'abstract': 'abstract_text'} where relevance is a string and title_text and abstract_text are strings.

    :param file: path to the file
    :param topic: topic of the relevance

    :return: list of relevances
    """
    try:
        data_dict = txt_file_reader(file)
        relevances = []
        for doc_id in data_dict.keys():
            relevance = Relevance(data_dict[doc_id]["Label"], doc_id, topic)
            relevances.append(relevance)
    except:
        print('ERROR: Your file %s does not contain documents with an associated relevance.' % file)
    else:
        return relevances



