from bioc.biocxml.decoder import load
from bioc.biocxml.encoder import dump
from bioc import BioCCollection, BioCDocument, BioCPassage, BioCAnnotation, BioCLocation
from biotmpy.data_structures import Document
from biotmpy.data_structures.sentence import Sentence

from biotmpy.data_structures.relevance import Relevance

import nltk

from biotmpy.wrappers.wrappers_utils import get_tokens


def bioc_to_docs(file, stop_words=None, lower=False, remove_punctuation=False, split_by_hyphen=True,
                 lemmatization=False, stems=False, config=None):
    """
    Converts a bioc file to a list of Document objects with the following structure: Document -> Sentence -> Token.

    :param file: path to the bioc file
    :param stop_words: list of stop words. For more details see https://www.nltk.org/book/ch02.html
    :param lower: boolean, if True, all words are converted to lower case
    :param remove_punctuation: boolean, if True, all punctuation is removed
    :param split_by_hyphen: boolean, if True, words are split by hyphen and the hyphen is removed
    :param lemmatization: boolean, if True, words are lemmatized. For more details see https://www.nltk.org/book/ch03.html
    :param stems: boolean, if True, words are stemmed. For more details see https://www.nltk.org/book/ch03.html
    :param config: DLConfig object with the configuration for the deep learning model

    :return: list of documents
    """
    if config:
        stop_words = config.stop_words
        lower = config.lower
        remove_punctuation = config.lower
        split_by_hyphen = config.split_by_hyphen
        lemmatization = config.lemmatization
        stems = config.stems

    collection = bioc_file_reader(file)
    docs = []
    for doc in collection.documents:
        document = get_document_bioc(doc, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems)
        docs.append(document)
    return docs


def docs_to_bioc(docs, file):
    """
    Converts a list of documents to a bioc file

    :param docs: list of documents to be converted to bioc file
    :param file: path to the bioc file to be created or overwritten

    :return: None
    """
    collection = BioCCollection()
    for d in docs:
        bioc_document = BioCDocument()
        bioc_document.id = d.id
        title = ''
        abstract = ''
        offset_a = d.getSectionOffset('a')
        for s in d.sentences:
            if s.pos_f < offset_a:
                title += s.string + ' '
            else:
                abstract += s.string + ' '
        bioc_passage_t = BioCPassage()
        bioc_passage_t.infons['type'] = 'title'
        bioc_passage_t.offset = 0
        bioc_passage_t.text = title.rstrip()
        bioc_document.add_passage(bioc_passage_t)
        bioc_passage_a = BioCPassage()
        bioc_passage_a.infons['type'] = 'abstract'
        bioc_passage_a.offset = offset_a
        bioc_passage_a.text = abstract.rstrip()
        bioc_document.add_passage(bioc_passage_a)
        collection.add_document(bioc_document)
        bioc_file_writer(file, collection)


def bioc_to_relevances(file, topic=None):
    """
    Converts a bioc file to a list of relevances (one relevance per document)

    :param file: path to the bioc file
    :param topic: topic of the documents in the bioc file
    """
    try:
        collection = bioc_file_reader(file)
        relevances = []
        for d in collection.documents:
            relevance = Relevance(d.infons["relevant"], d.id, topic)
            relevances.append(relevance)
    except KeyError:
        raise ValueError('Your file %s does not contain documents with an associated relevance.' % file)
    else:
        return relevances


def bioc_file_reader(file):
    """
    Reads a bioc file

    :param file: path to the bioc file

    :return: BioCCollection object
    """
    with open(file, 'r') as fp:
        collection = load(fp)
    return collection


def bioc_file_writer(file, collection):
    """
    Writes a bioc file

    :param file: path to the bioc file to be created or overwritten
    :param collection: BioCCollection object to be written to the bioc file

    :return: None
    """

    with open(file, 'w') as fp:
        dump(collection, fp)


def get_document_bioc(biocdoc, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
    """
    Converts a bioc document to a Document object

    :param biocdoc: BioCDocument object
    :param stop_words: list of stop words. For more details see https://www.nltk.org/book/ch02.html
    :param lower: boolean, if True, all words are converted to lower case
    :param remove_punctuation: boolean, if True, all punctuation is removed
    :param split_by_hyphen: boolean, if True, words are split by hyphen and the hyphen is removed
    :param lemmatization: boolean, if True, words are lemmatized. For more details see https://www.nltk.org/book/ch03.html
    :param stems: boolean, if True, words are stemmed. For more details see https://www.nltk.org/book/ch03.html

    :return: Document object
    """

    sentences, raw_title, raw_abstract = get_sentences_bioc(biocdoc, stop_words, lower, remove_punctuation,
                                                            split_by_hyphen,
                                                            lemmatization,
                                                            stems)
    document = Document(sentences)
    document.raw_title = raw_title
    document.raw_abstract = raw_abstract
    return document


def get_sentences_bioc(biocdoc, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
    """
    Converts a bioc document to a list of Sentence objects

    :param biocdoc: BioCDocument object
    :param stop_words: list of stop words. For more details see https://www.nltk.org/book/ch02.html
    :param lower: boolean, if True, all words are converted to lower case
    :param remove_punctuation: boolean, if True, all punctuation is removed
    :param split_by_hyphen: boolean, if True, words are split by hyphen and the hyphen is removed
    :param lemmatization: boolean, if True, words are lemmatized. For more details see https://www.nltk.org/book/ch03.html
    :param stems: boolean, if True, words are stemmed. For more details see https://www.nltk.org/book/ch03.html

    :return: list of Sentence objects
    """
    raw_title = ''
    raw_abstract = ''
    offset_a = 0
    sentences = []
    for p in biocdoc.passages:
        if p.infons['type'] == 'title':
            raw_title = p.text
        elif p.infons['type'] == 'abstract':
            raw_abstract = p.text
            offset_a = p.offset
    text = raw_title + ' ' + raw_abstract
    sent = nltk.sent_tokenize(text)
    offset = 0
    for s in sent:
        if offset < offset_a or offset_a == 0:
            passage_type = 't'
        else:
            passage_type = 'a'
        tokens = get_tokens(s, biocdoc.id, offset, passage_type, stop_words, lower, remove_punctuation, split_by_hyphen,
                            lemmatization, stems)
        s = ''
        for i in range(len(tokens)):
            if i != len(tokens) - 1:
                s += tokens[i].string + ' '
            else:
                s += tokens[i].string
        sentence = Sentence(s, tokens, passage_type, offset, offset + len(s))
        sentences.append(sentence)
        offset += len(s) + 1
    return sentences, raw_title, raw_abstract

# def get_annotations_tokens(biocdoc):
#     sentences = get_sentences(biocdoc)
#     toks = []
#     for s in sentences:
#         toks += s.tokens
#     annots = []
#     for t in toks:
#         label = 'O'
#         text = t.string
#         pos_f = t.pos_f
#         for p in biocdoc.passages:
#             for a in p.annotations:
#                 pos = a.locations[0].offset
#                 l = a.locations[0].length
#                 if t.pos_i == pos:
#                     label = 'B'
#                     if t.pos_f > pos + l:
#                         text = t.string[:l]
#                         pos_f = pos + l
#                     if a.infons['type'] == 'Disease':
#                         label += '-d'
#                     elif a.infons['type'] == 'Chemical':
#                         label += '-c'
#                     break
#                 elif t.pos_i > pos and t.pos_i < pos + l:
#                     label = 'I'
#                     if t.pos_f > pos + l:
#                         text = t.string[:l]
#                         pos_f = pos + l
#                     if a.infons['type'] == 'Disease':
#                         label += '-d'
#                     elif a.infons['type'] == 'Chemical':
#                         label += '-c'
#                     break
#         annotation = Annotation(text, t.passage_type, t.pos_i, pos_f, label, biocdoc.id)
#         annots.append(annotation)
#     return annots
#
#
# def annots_tokens_to_bioc(annotations, file):
#     annots = {}
#     for a in annotations:
#         if a.doc_id not in annots:
#             annots[a.doc_id] = []
#             i = 0
#         if a.label[0] == 'B':
#             annots[a.doc_id].append([i, a.label[2], a.pos_i, a.pos_f, a.string, a.passage_type])
#             i += 1
#         elif a.label[0] == 'I':
#             if a.pos_i == annots[a.doc_id][-1][3]:
#                 annots[a.doc_id][-1][4] += a.string
#             else:
#                 annots[a.doc_id][-1][4] += ' ' + a.string
#             annots[a.doc_id][-1][3] = a.pos_f
#
#     collection = bioc_file_reader(file)
#     for doc_id in annots:
#         for a in annots[doc_id]:
#             bioc_annotation = BioCAnnotation()
#             bioc_annotation.id = str(a[0])
#             bioc_annotation.text = a[4]
#             if a[1] == 'd':
#                 bioc_annotation.infons['type'] = 'Disease'
#             elif a[1] == 'c':
#                 bioc_annotation.infons['type'] = 'Chemical'
#             offset = a[2]
#             length = a[3] - a[2]
#             bioc_location = BioCLocation(offset, length)
#             bioc_annotation.add_location(bioc_location)
#             for d in collection.documents:
#                 if doc_id == d.id:
#                     for p in d.passages:
#                         if p.infons['type'] == 'title' and a[5] == 't':
#                             p.add_annotation(bioc_annotation)
#                         elif p.infons['type'] == 'abstract' and a[5] == 'a':
#                             p.add_annotation(bioc_annotation)
#     bioc_file_writer(file, collection)
#
#
# def bioc_to_annots_tokens(file):
#     """
#     Converts a bioc file to a list of annotations (one annotation per token)
#
#     :param file: path to the bioc file
#
#     :return: list of annotations
#     """
#     try:
#         collection = bioc_file_reader(file)
#         annots = []
#         for d in collection.documents:
#             annotations = get_annotations_tokens(d)
#             annots += annotations
#     except:
#         print('ERROR: Your file %s does not contain annotated documents.' % file)
#     else:
#         return annots
