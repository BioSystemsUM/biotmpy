import string
from typing import List

import nltk
from nltk import WordNetLemmatizer, PorterStemmer

from biotmpy.data_structures import Document, Sentence, Token


def get_document(dict_or_pandas_row, doc_id, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization,
                 stems):
    """
    Converts  a dictionary or a pandas row to a Document object with the sentences and tokens. Examples of the structure of both dictionary and pandas can be seen on the data folder.

    :param dict_or_pandas_row: dictionary or pandas row
    :param doc_id: document id
    :param stop_words: list of stop words. For more details see https://www.nltk.org/book/ch02.html
    :param lower: boolean, if True, all words are converted to lower case
    :param remove_punctuation: boolean, if True, all punctuation is removed
    :param split_by_hyphen: boolean, if True, words are split by hyphen and the hyphen is removed
    :param lemmatization: boolean, if True, words are lemmatized. For more details see https://www.nltk.org/book/ch03.html
    :param stems: boolean, if True, words are stemmed. For more details see https://www.nltk.org/book/ch03.html

    :return: Document object
    """
    sentences = []
    if dict_or_pandas_row['title'] != '':
        sentences.extend(get_sentences(dict_or_pandas_row['title'], passage_type='t',
                                       doc_id=doc_id, stop_words=stop_words,
                                       lower=lower,
                                       remove_punctuation=remove_punctuation,
                                       split_by_hyphen=split_by_hyphen,
                                       lemmatization=lemmatization,
                                       stems=stems))
    if dict_or_pandas_row['abstract'] != '':
        sentences.extend(get_sentences(dict_or_pandas_row['abstract'], passage_type='a',
                                       doc_id=doc_id, stop_words=stop_words,
                                       lower=lower,
                                       remove_punctuation=remove_punctuation,
                                       split_by_hyphen=split_by_hyphen,
                                       lemmatization=lemmatization,
                                       stems=stems))
    document = Document(sentences=sentences)
    document.raw_title = dict_or_pandas_row['title']
    document.raw_abstract = dict_or_pandas_row['abstract']

    return document


def get_sentences(text, passage_type, doc_id, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization,
                  stems):
    """
    Converts text to a list of Sentence objects. Each sentence is a list of Token objects. Each token has a string, a start position, an end position, an offset and a document id.

    :param text: text to be converted
    :param passage_type: type of the passage. It can be 't' for title and 'a' for abstract
    :param doc_id: document id
    :param stop_words: list of stop words. For more details see https://www.nltk.org/book/ch02.html
    :param lower: boolean, if True, all words are converted to lower case
    :param remove_punctuation: boolean, if True, all punctuation is removed
    :param split_by_hyphen: boolean, if True, words are split by hyphen and the hyphen is removed
    :param lemmatization: boolean, if True, words are lemmatized. For more details see https://www.nltk.org/book/ch03.html
    :param stems: boolean, if True, words are stemmed. For more details see https://www.nltk.org/book/ch03.html

    :return: list of Sentence objects
    """
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


def get_tokens(text, doc_id=None, offset=0, passage_type=None, stop_words=None,
               lower=None, remove_punctuation=None, split_by_hyphen=None,
               lemmatization=None, stems=None):
    """
    Converts text to a list of Token objects. Each token has a string, a start position, an end position, an offset and a document id.

    :param text: text to be converted
    :param doc_id: document id
    :param offset: offset. It is used identify the position of the token in the original text
    :param passage_type: type of the passage. It can be 't' for title and 'a' for abstract
    :param stop_words: list of stop words. For more details see https://www.nltk.org/book/ch02.html
    :param lower: boolean, if True, all words are converted to lower case
    :param remove_punctuation: boolean, if True, all punctuation is removed
    :param split_by_hyphen: boolean, if True, words are split by hyphen and the hyphen is removed
    :param lemmatization: boolean, if True, words are lemmatized. For more details see https://www.nltk.org/book/ch03.html
    :param stems: boolean, if True, words are stemmed. For more details see https://www.nltk.org/book/ch03.html

    :return: list of Token objects
    """
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
                    token = Token('-', passage_type, pos + offset, start + offset, offset, doc_id)
                    tokens.append(token)
                else:
                    if splitted[i] != '':
                        pos = text.find(splitted[i], start)
                        start = pos + len(splitted[i])
                        token = Token(splitted[i], passage_type, pos + offset, start + offset, offset, doc_id)
                        tokens.append(token)
                    if i + 1 < len(splitted):
                        pos = text.find('-', start)
                        start = pos + 1
                        token = Token('-', passage_type, pos + offset, start + offset, offset, doc_id)
                        tokens.append(token)
        else:
            start = pos + len(t)
            token = Token(t, passage_type, pos + offset, start + offset, offset, doc_id)
            tokens.append(token)
    return tokens



def sort_list(l):
    """
    Sorts a list.

    :param l: list to sort

    :return: sorted list
    """
    l.sort()
    return l


def txt_file_reader(file: str) -> List[str]:
    """
    Reads a txt file

    :param file: path to the txt file

    :return: list of strings, each string is a line of the txt file
    """
    with open(file, 'r') as fp:
        text = fp.readlines()
    return text


def get_doc_ids(docs):
    """
    Gets the Document IDs from a list of Document objects.

    :param docs: list of Document objects

    :return: list of Document IDs
    """
    indexes = []
    for d in docs:
        indexes.append(d.id)
    return indexes
