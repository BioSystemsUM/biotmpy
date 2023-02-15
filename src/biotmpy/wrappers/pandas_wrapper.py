from src.biotmpy.data_structures import Document
from src.biotmpy.data_structures.sentence import Sentence
from src.biotmpy.data_structures import Token
from src.biotmpy.data_structures.relevance import Relevance
import pandas

from wrappers_utils import *

def docs_to_pandasdocs(docs):
    """
    Converts a list of Document objects to a pandas DataFrame.

    :param docs: list of Document objects

    :return: pandas DataFrame
    """
    indexes = get_doc_ids(docs)
    dataframe = pandas.DataFrame(data=docs, index=indexes, columns=['Document'])
    return dataframe


def relevances_to_pandas(dataframe, relevances, merge=True):
    """
    Converts a list of Relevance objects to a pandas DataFrame.

    :param dataframe: pandas DataFrame
    :param relevances: list of Relevance objects
    :param merge: boolean, if True, the returned DataFrame is merged with the input DataFrame

    :return: pandas DataFrame
    """
    indexes = []
    labels = []
    for r in relevances:
        indexes.append(r.id)
        if r.label == 'no' or r.label == 0 or r.label == 'non-relevant':
            labels.append(0)
        elif r.label == 'yes' or r.label == 1 or r.label == 'relevant':
            labels.append(1)
    relevances_df = pandas.DataFrame(data={'Label': labels}, index=indexes)
    if merge:
        return pandas.merge(dataframe, relevances_df, left_index=True, right_index=True)
    else:
        return relevances_df


def pandasdocs_to_docs(dataframe):
    """
    Converts a pandas DataFrame to a list of Document objects.

    :param dataframe: pandas DataFrame

    :return: list of Document objects
    """
    return list(dataframe['Document'])


def pandas_to_relevances(dataframe):
    pass


def pandas_to_sentences(dataframe):
    """
    Converts a pandas DataFrame to a list of Sentence objects.

    :param dataframe: pandas DataFrame

    :return: list of Sentence objects
    """
    tokens = pandas_to_tokens(dataframe)
    sentences = []
    sent_tokens = []
    s_id = -1
    doc_id = -1
    for t in tokens:
        if s_id != t.sent_id or doc_id != t.doc_id:
            if sent_tokens:
                sentences.append(Sentence(sent, sent_tokens, sent_tokens[0].pos_i, sent_tokens[-1].pos_f))
            sent = t.string
            sent_tokens = [t]
            s_id = t.sent_id
            doc_id = t.doc_id
            prev_tk_pos_f = t.pos_f
        else:
            spaces = ''
            count_spaces = t.pos_i - prev_tk_pos_f
            for i in range(count_spaces):
                spaces += ' '
            sent += spaces + t.string
            sent_tokens.append(t)
            prev_tk_pos_f = t.pos_f
    if sent_tokens:
        sentences.append(Sentence(sent, sent_tokens, sent_tokens[0].pos_i, sent_tokens[-1].pos_f))
    return sentences


def pandas_to_tokens(dataframe):
    tokens = indexes_to_tokens(dataframe)
    return sort_list(tokens)


def sort_list(l):
    l.sort()
    return l


def indexing_series_tokens(dataframe):
    tokens = list(dataframe['Token'])
    indexes = []
    ind = ''
    for t in tokens:
        ind += t.string + ' ' + str(t.pos_i) + ' ' + str(t.pos_f) + ' ' + t.passage_type + ' ' + str(
            t.sent_id) + ' ' + t.doc_id
        indexes.append(ind)
        ind = ''
    return indexes


def get_doc_ids(docs):
    indexes = []
    for d in docs:
        indexes.append(d.id)
    return indexes


def indexes_to_tokens(dataframe):
    indexes = list(dataframe.index.values)
    tokens = []
    for ind in indexes:
        tok = ind.split(' ')
        tokens.append(Token(tok[0], tok[3], int(tok[1]), int(tok[2]), int(tok[4]), tok[5]))
    return tokens


def indexes_to_annots(dataframe):
    indexes = list(dataframe.index.values)
    labels = list(dataframe['Label'])
    annots = []
    for i in range(len(indexes)):
        tok = indexes[i].split(' ')
        annots.append(Annotation(tok[0], tok[3], int(tok[1]), int(tok[2]), labels[i], tok[5]))
    return annots


def docs_to_pandastokens(docs):
    tokens = []
    for d in docs:
        for s in d.sentences:
            tokens += s.tokens
    dataframe = pandas.DataFrame(tokens, columns=['Token'])
    indexes = indexing_series_tokens(dataframe)
    dataframe = pandas.DataFrame(data=dataframe.values, index=indexes, columns=list(dataframe.columns))
    return dataframe


def annot_to_pandas(dataframe, annotations):
    indexes = list(dataframe.index.values)
    labels = []
    for a in annotations:
        labels.append(a.label)
    dataframe['Label'] = pandas.Series(labels, index=indexes)
    return dataframe


def pandas_to_annot(dataframe):
    annots = indexes_to_annots(dataframe)
    return sort_list(annots)


def pandastokens_to_docs(dataframe):
    sentences = pandas_to_sentences(dataframe)
    doc_sentences, docs = [], []
    d_id = -1
    for s in sentences:
        s_doc_id = s.getDocId()
        if s_doc_id != d_id:
            d_id = s_doc_id
            if doc_sentences:
                docs.append(Document(doc_sentences))
            doc_sentences = []
        doc_sentences.append(s)
    if doc_sentences:
        docs.append(Document(doc_sentences))
    return docs
