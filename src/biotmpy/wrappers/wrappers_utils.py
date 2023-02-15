import nltk

from wrappers_utils import *


def get_sentences(text, passage_type, doc_id, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
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
