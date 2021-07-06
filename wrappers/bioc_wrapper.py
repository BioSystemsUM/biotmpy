from bioc.biocxml.decoder import load
from bioc.biocxml.encoder import dump
from bioc import BioCCollection, BioCDocument, BioCPassage, BioCAnnotation, BioCLocation
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


def bioc_to_docs(file, stop_words=None, lower=False, remove_punctuation=False, split_by_hyphen=True, lemmatization=False, stems=False, dl_config=None):
    """
    
    :param file:
    :param stop_words:
    :param lower:
    :param remove_punctuation:
    :param split_by_hyphen:
    :param lemmatization:
    :param stems:
    :param dl_config:
    :return:
    """
    if dl_config:
        stop_words = dl_config.stop_words
        lower = dl_config.lower
        remove_punctuation = dl_config.lower
        split_by_hyphen = dl_config.split_by_hyphen
        lemmatization = dl_config.lemmatization
        stems = dl_config.stems

    collection = bioc_file_reader(file)
    docs = []
    for d in collection.documents:
        document = get_document(d, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems)
        docs.append(document)
    return docs


def docs_to_bioc(docs, file):
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


def bioc_to_annots_tokens(file):
    try:
        collection = bioc_file_reader(file)
        annots = []
        for d in collection.documents:
            annotations = get_annotations_tokens(d)
            annots += annotations        
    except:
        print('ERROR: Your file %s does not contain annotated documents.' % file)
    else:
        return annots

def bioc_to_relevances(file, description):
    try:
        collection = bioc_file_reader(file)
        relevances = []
        for d in collection.documents:
            relevance = Relevance(d.infons["relevant"], d.id, description)
            relevances.append(relevance)
    except:
        print('ERROR: Your file %s does not contain documents with an associated relevance.' % file)
    else:
        return relevances


def annots_tokens_to_bioc(annotations, file):
    annots = {}
    for a in annotations:
        if a.doc_id not in annots:
            annots[a.doc_id] = []
            i = 0
        if a.label[0] == 'B':
            annots[a.doc_id].append([i, a.label[2], a.pos_i, a.pos_f, a.string, a.passage_type])
            i += 1
        elif a.label[0] == 'I':
            if a.pos_i == annots[a.doc_id][-1][3]:
                annots[a.doc_id][-1][4] += a.string
            else:                
                annots[a.doc_id][-1][4] += ' ' + a.string
            annots[a.doc_id][-1][3] = a.pos_f

    collection = bioc_file_reader(file)
    for doc_id in annots:
        for a in annots[doc_id]:
            bioc_annotation = BioCAnnotation()
            bioc_annotation.id = str(a[0])
            bioc_annotation.text = a[4]
            if a[1] == 'd':
                bioc_annotation.infons['type'] = 'Disease'
            elif a[1] == 'c':
                bioc_annotation.infons['type'] = 'Chemical'
            offset = a[2]
            length = a[3] - a[2]
            bioc_location = BioCLocation(offset, length)
            bioc_annotation.add_location(bioc_location)
            for d in collection.documents:
                if doc_id == d.id:
                    for p in d.passages:
                        if p.infons['type'] == 'title' and a[5] == 't':
                            p.add_annotation(bioc_annotation)
                        elif p.infons['type'] == 'abstract' and a[5] == 'a':
                            p.add_annotation(bioc_annotation)
    bioc_file_writer(file, collection)


def bioc_file_reader(file):
    with open(file, 'r') as fp:
        collection = load(fp)
    return collection


def bioc_file_writer(file, collection):
    with open(file, 'w') as fp:
        dump(collection, fp)


def txt_file_reader(file):
    with open(file, 'r') as fp:
        text = fp.readlines()
    return text


def get_document(biocdoc, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
    sentences, raw_title, raw_abstract = get_sentences(biocdoc, 
    													stop_words, 
    													lower, 
    													remove_punctuation, 
    													split_by_hyphen, 
    													lemmatization, 
    													stems)
    document = Document(sentences)
    document.raw_title = raw_title
    document.raw_abstract = raw_abstract
    return document


def get_sentences(biocdoc, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems):
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
        tokens = get_tokens(s, biocdoc.id, offset, passage_type, stop_words, lower, remove_punctuation, split_by_hyphen, lemmatization, stems)
        s = ''
        for i in range(len(tokens)):
            if i != len(tokens) - 1: 
                s += tokens[i].string + ' '
            else:
                s += tokens[i].string
        sentence = Sentence(s, tokens, passage_type, offset, offset+len(s))
        sentences.append(sentence)
        offset += len(s)+1
    return sentences, raw_title, raw_abstract


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
                        start = pos+len(splitted[i])
                        token = Token(splitted[i], passage_type, pos + offset, start + offset, offset, doc_id)
                        tokens.append(token)
                    if i+1 < len(splitted):
                        pos = text.find('-', start)
                        start = pos + 1
                        token = Token('-', passage_type, pos + offset, start + offset, offset, doc_id)
                        tokens.append(token)
        else:
            start = pos + len(t)
            token = Token(t, passage_type, pos + offset, start + offset, offset, doc_id)
            tokens.append(token)
    return tokens


def get_annotations_tokens(biocdoc):
    sentences = get_sentences(biocdoc)
    toks = []
    for s in sentences:
        toks += s.tokens
    annots = []
    for t in toks:
        label = 'O'
        text = t.string
        pos_f = t.pos_f
        for p in biocdoc.passages:
            for a in p.annotations:
                pos = a.locations[0].offset
                l = a.locations[0].length
                if t.pos_i == pos:
                    label = 'B'
                    if t.pos_f > pos+l:
                        text = t.string[:l]
                        pos_f = pos+l
                    if a.infons['type'] == 'Disease':
                        label += '-d'
                    elif a.infons['type'] == 'Chemical':
                        label += '-c'
                    break
                elif t.pos_i > pos and t.pos_i < pos+l:
                    label = 'I'
                    if t.pos_f > pos+l:
                        text = t.string[:l]
                        pos_f = pos+l
                    if a.infons['type'] == 'Disease':
                        label += '-d'
                    elif a.infons['type'] == 'Chemical':
                        label += '-c'
                    break
        annotation = Annotation(text, t.passage_type, t.pos_i, pos_f, label, biocdoc.id)
        annots.append(annotation)
    return annots
