import nltk

from wrappers_utils import *

def get_document(dict_or_pandas, doc_id, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization,
                            stems):
    """
    Converts  a dictionary or a pandas row to a Document object with the sentences and tokens. Examples of the structure of both dictionary and pandas can be seen on the data folder.

    :param dict_or_pandas: dictionary or pandas row
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
    if dict_or_pandas['title'] != '':
        sentences.extend(get_sentences(text_dict['title'], passage_type='t',
                                       doc_id=doc_id, stop_words=stop_words,
                                       lower=lower,
                                       remove_punctuation=remove_punctuation,
                                       split_by_hyphen=split_by_hyphen,
                                       lemmatization=lemmatization,
                                       stems=stems))
    if dict_or_pandas['abstract'] != '':
        sentences.extend(get_sentences(text_dict['abstract'], passage_type='a',
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
def get_sentences(text, passage_type, doc_id, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
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
    return

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


def txt_file_reader(file):
    """
    :param file: path to the file

    :return: dictionary with the documents
    """
    with open(file, 'r') as fp:
        text = fp.readlines()
    return text


def sort_list(l):
    """
    Sorts a list.

    :param l: list to sort

    :return: sorted list
    """
    l.sort()
    return l