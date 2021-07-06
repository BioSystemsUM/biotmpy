from Bio.Entrez import efetch, read
from Bio import Entrez
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import nltk
from data_structures.document import Document
from data_structures.sentence import Sentence
from data_structures.token import Token
from wrappers.dictionary_wrapper import get_sentences_dictionary, get_tokens_dictionary
import string
from pdftitle import get_title_from_io

        
def pmids_to_docs(pmids, email, dl_config):
    docs = []
    pmids_not_found = []
    for pmid in pmids:
        sentences = []
        data = get_data_from_pmid(pmid, email)
        if data is None:
            pmids_not_found.append(pmid)
        else:
            if data['Title'] != '':
                sentences.extend(get_sentences_dictionary(data['Title'], passage_type = 't', 
                                                doc_id=pmid, stop_words=dl_config.stop_words,
                                                lower=dl_config.lower,
                                                remove_punctuation=dl_config.remove_punctuation, 
                                                split_by_hyphen=dl_config.split_by_hyphen,
                                                lemmatization=dl_config.lemmatization,
                                                stems=dl_config.stems))
            if data['Abstract'] != '':
                sentences.extend(get_sentences_dictionary(data['Abstract'], passage_type = 'a', 
                                                doc_id=pmid, stop_words=dl_config.stop_words,
                                                lower=dl_config.lower,
                                                remove_punctuation=dl_config.remove_punctuation, 
                                                split_by_hyphen=dl_config.split_by_hyphen,
                                                lemmatization=dl_config.lemmatization,
                                                stems=dl_config.stems))
            if sentences:  
                doc = Document(sentences=sentences)
                doc.raw_title = data['Title']
                docs.append(doc)
            else:
                pmids_not_found.append(pmid)
    return docs, pmids_not_found



#using PMID
def get_data_from_pmid(pmid, email):
    Entrez.email = email
    Entrez.api_key = '0a925fca3ee9689bd778607f10e438fbfa09' 
    
    handle = efetch(db='pubmed', id=int(pmid), retmode='xml')
    xml_data = read(handle)
    handle.close()
    try:
        article = xml_data['PubmedArticle'][0]['MedlineCitation']['Article']
        title = article['ArticleTitle']
        try:
            abstract = article['Abstract']['AbstractText'][0]
            return {'Title': title, 'Abstract': abstract}
        except:
            return {'Title': title, 'Abstract': ''}
    except:
        article = xml_data['PubmedBookArticle'][0]['BookDocument']
        title = article['ArticleTitle']
        try:
            abstract = article['Abstract']['AbstractText'][0]
            return {'Title': title, 'Abstract': abstract}
        except:
            return {'Title': title, 'Abstract': ''}




#Using Term
def term_to_docs(term, email, retmax, dl_config):
    pmids = get_data_from_term(term, email, retmax)
    if pmids is None:
        return None
    else:
        docs, pmids_not_found = pmids_to_docs(pmids, email, dl_config)
        return docs


def get_data_from_term(term, email, retmax):
    Entrez.email = email
    Entrez.api_key = '0a925fca3ee9689bd778607f10e438fbfa09' 

    handle = Entrez.esearch(db="pubmed", retmax=retmax, term=term, idtype="acc", sort='relevance')
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

#Using pdfs
def pdf_paths_to_titles(pdf_paths, email):
    titles = []
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as f:
            titles.append(get_data_from_pdf(f, email))
    return titles

def pdfs_to_docs(files, email, dl_config):
    pmids = []
    docs_not_found = []
    try:
        for file in files:
            pmid = get_data_from_pdf(file, email)
            if pmid:
                pmids.append(pmid)
            else:
                docs_not_found.append(file.filename)
        docs, pmids_not_found = pmids_to_docs(pmids, email, dl_config)
        return docs, docs_not_found
    except:
        return None, None
        
def get_data_from_pdf(file, email):
    Entrez.email = email
    Entrez.api_key = '0a925fca3ee9689bd778607f10e438fbfa09' 

    try:
        title = get_title_from_io(file)
        pmid = get_data_from_term(term=title, email='21nunoalves21@gmail.com', retmax=1)[0]
        return pmid
    except:
        return None


if __name__ == '__main__':
    # path = 'D:/Desktop/artigos/rdml.pdf'
    # #id = 20367574
    # print(get_data_from_pdf_path(path, '21nunoalves21@gmail.com'))
    # term = "Predicting commercially available antiviral drugs that may act on the novel coronavirus (SARS-CoV-2) through a drug-target interaction deep learning model"
    # print(get_data_from_term(term, '21nunoalves21@gmail.com', 1)['IdList'])

    paths = ['D:/Desktop/artigos/Burns.pdf',
             'D:/Desktop/artigos/Fergadis.pdf',
             'D:/Desktop/artigos/Luo.pdf',
             'D:/Desktop/artigos/Mohan.pdf',
             'D:/Desktop/artigos/rdml.pdf',
             'D:/Desktop/artigos/Yan.pdf']
    
    pdf_paths_to_titles(paths, '21nunoalves21@gmail.com')