from data_structures.document import Document
from data_structures.sentence import Sentence
from data_structures.token import Token
from data_structures.relevance import Relevance
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import re
import string
from nltk.stem import WordNetLemmatizer, PorterStemmer


def dictionary_to_docs(file, stop_words=None, lower=False, remove_punctuation=False, split_by_hyphen=True, lemmatization=False, stems=False, dl_config=None):
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
        document = get_document(data_dict[doc_id], doc_id=doc_id, stop_words=stop_words, lower=lower, remove_punctuation=remove_punctuation,
                                 split_by_hyphen=split_by_hyphen, lemmatization=lemmatization, stems=stems)
        docs.append(document)
    return docs

def dictionary_to_relevances(file, description=None):
    try:
        data_dict = txt_file_reader(file)
        relevances = []
        for doc_id in data_dict.keys():
            relevance = Relevance(data_dict[doc_id]["Label"], doc_id, description)
            relevances.append(relevance)
    except:
        print('ERROR: Your file %s does not contain documents with an associated relevance.' % file)
    else:
        return relevances


def txt_file_reader(file):
    with open(file, 'r') as fp:
        text = fp.readlines()
    return text


def get_document(text_dict, doc_id, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
    sentences = []
    if text_dict['title'] != '':
        sentences.extend(get_sentences_dictionary(text_dict['title'], passage_type = 't', 
                                        doc_id=doc_id, stop_words=stop_words,
                                        lower=lower,
                                        remove_punctuation=remove_punctuation, 
                                        split_by_hyphen=split_by_hyphen,
                                        lemmatization=lemmatization,
                                        stems=stems))
    if text_dict['abstract'] != '':
        sentences.extend(get_sentences_dictionary(text_dict['abstract'], passage_type = 'a', 
                                        doc_id=doc_id, stop_words=stop_words,
                                        lower=lower,
                                        remove_punctuation=remove_punctuation, 
                                        split_by_hyphen=split_by_hyphen,
                                        lemmatization=lemmatization,
                                        stems=stems))
    document = Document(sentences=sentences)
    document.raw_title = text_dict['title']
    document.raw_abstract = text_dict['abstract']

    return document


def get_sentences_dictionary(text, passage_type, doc_id, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
    sent = nltk.sent_tokenize(text)  
    sentences = []
    for s in sent:
        tokens = get_tokens_dictionary(s, passage_type=passage_type, doc_id=doc_id, stop_words=stop_words, lower=lower, 
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


def get_tokens_dictionary(text, passage_type, doc_id, stop_words, 
                lower, remove_punctuation, split_by_hyphen, 
                lemmatization, stems):
    tokens = []
    tokenized = nltk.word_tokenize(text)
    start = 0
    punctuation = string.punctuation
    for t in tokenized:
        if stop_words:
            if t.lower() in stop_words:
                continue
        if lower:
            t = t.lower()
        if remove_punctuation:
            if t in punctuation:
                continue
        if lemmatization:
            lemmatizer = WordNetLemmatizer()
            t = lemmatizer.lemmatize(t)
        if stems:
            stemmer = PorterStemmer()
            t = stemmer.stem(t)
        pos = text.find(t, start)
        if '-' in t and split_by_hyphen:
            splitted = t.split('-')
            for i in range(len(splitted)):
                if i == 0 and splitted[i] == '':
                    pos = text.find('-', pos)
                    start = pos + 1
                    token = Token('-', passage_type, doc_id = doc_id)
                    tokens.append(token)
                else:
                    if splitted[i] != '':
                        pos = text.find(splitted[i], start)
                        start = pos+len(splitted[i])
                        token = Token(splitted[i], passage_type=passage_type, doc_id = doc_id)
                        tokens.append(token)
                    if i+1 < len(splitted):
                        pos = text.find('-', start)
                        start = pos + 1
                        token = Token('-', passage_type, doc_id = doc_id)
                        tokens.append(token)
        else:
            start = pos + len(t)
            token = Token(t, passage_type, doc_id = doc_id)
            tokens.append(token)
    return tokens