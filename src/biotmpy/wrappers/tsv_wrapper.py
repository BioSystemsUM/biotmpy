
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

def tsv_to_docs(file, stop_words=None, lower=False, remove_punctuation=False, split_by_hyphen=True, lemmatization=False, stems=False, dl_config=None, sep='\t'):
    if dl_config:
        stop_words = dl_config.stop_words
        lower = dl_config.lower
        remove_punctuation = dl_config.lower
        split_by_hyphen = dl_config.split_by_hyphen
        lemmatization = dl_config.lemmatization
        stems = dl_config.stems

    dataframe = tsv_file_reader(file, sep=sep)
    docs = []
    for i, df_row in dataframe.iterrows():
        document = get_document(df_row, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems)
        docs.append(document)
    return docs


def tsv_to_relevances(file, sep='\t', description=None):
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
    dataframe = pd.read_csv(file, sep=sep)
    return standardize_headers(dataframe)
    

def standardize_headers(dataframe):
    dataframe.columns = dataframe.columns.str.lower()
    return dataframe


def get_document(df_row, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
    sentences = []
    if df_row['title'] != '':
        sentences.extend(get_sentences_dictionary(df_row['title'], passage_type = 't', 
                                        doc_id=df_row['pmid'], stop_words=stop_words,
                                        lower=lower,
                                        remove_punctuation=remove_punctuation, 
                                        split_by_hyphen=split_by_hyphen,
                                        lemmatization=lemmatization,
                                        stems=stems))
    if df_row['abstract'] != '':
        sentences.extend(get_sentences_dictionary(df_row['abstract'], passage_type = 'a', 
                                        doc_id=df_row['pmid'], stop_words=stop_words,
                                        lower=lower,
                                        remove_punctuation=remove_punctuation, 
                                        split_by_hyphen=split_by_hyphen,
                                        lemmatization=lemmatization,
                                        stems=stems))
    document = Document(sentences=sentences)
    document.raw_title = df_row['title']
    document.raw_abstract = df_row['abstract']

    return document


def get_sentences_dictionary(text, passage_type, doc_id, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
    sent = nltk.sent_tokenize(text)  
    sentences = []
    for s in sent:
        tokens = get_tokens(s, passage_type=passage_type, doc_id=doc_id, stop_words=stop_words, lower=lower,
                                    remove_punctuation=remove_punctuation, split_by_hyphen=split_by_hyphen, 
                                    lemmatization=lemmatization, stems=stems)
        s = ''
        for i in range(len(tokens)):
            if i != len(tokens) - 1: 
                s += tokens[i].string + ' '
            else:
                s += tokens[i].string
        sentence = Sentence(s, tokens, passage_type=passage_type)
        sentences.append(sentence)
    return sentences


if __name__=='__main__':
    import sys
    sys.append('../')