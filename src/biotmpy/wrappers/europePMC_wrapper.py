import requests
import xml.etree.ElementTree as ET
import re
import pickle
#from nltk.tokenize import sent_tokenize
from  classes_do_europePMC_wrapper import Paper
#import nltk
#nltk.download('punkt')
#nltk.download("stopwords")
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer

## API

def pmc_to_list(term): 
    results = []
    
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(TITLE:"{term}" OR ABSTRACT:"{term}")&resultType=core&format=json' 

    nextPage = True
    while nextPage and len(results) < 10:
        r = requests.get(url=url) 
        data = r.json()

        results += data['resultList']['result']
        try:
            url = data['nextPageUrl']
        except:
            nextPage = False

    return results


def list_to_paper_instances(paper_list, keyword=None):  
    paper_instances = []
    for paper_dict in paper_list:
        pmid = paper_dict['id']
        title = paper_dict['title']
        abstract = paper_dict.get('abstractText', '')  
        source = paper_dict.get('source')
        paper_obj = Paper(pmid, title, abstract, source, query_keyword=keyword)
        paper_instances.append(paper_obj)
    #print(paper_instances)
    # with open('paper_instances.pickle', 'wb') as file:
    #     pickle.dump(paper_instances, file)
    return paper_instances



def get_annotations_paper_instances(paper_instances):
    instances_annotations = []
    for paper_instance in paper_instances:
        instances_annotations.append(get_annotations(paper_instance))

    return instances_annotations

# #API Annotations

def get_annotations(paper_instance): 
    source = paper_instance.source
    id = paper_instance.id

    annotations_url= f'https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds={source}%3A{id}&type=Organisms%2CGene_Proteins%2CDiseases%2CChemicals'
    
    #annotations_url= f'https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds=MED%3A37158067&type=Organisms%2CGene_Proteins%2CDiseases%2CChemicals'
    try:
        response = requests.get(annotations_url)
        annotations_data = response.json()
        paper_instance = annotations_to_dict(annotations_data[0], paper_instance)
        # with open('annotations.pickle', 'wb') as file:
        #     pickle.dump(paper_instance, file)
        return paper_instance


    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)

    except requests.exceptions.ConnectionError as errc:
        print("Connection Error:", errc)

    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)

    except requests.exceptions.RequestException as err:
        print("Something went wrong:", err)

def annotations_to_dict(json_anotations, paper_instance):
    print(json_anotations)
    for annotation in json_anotations['annotations']:
        category = annotation['type']
        exact = annotation['exact'].lower()
        if exact not in paper_instance.annotations[category]:
            paper_instance.annotations[category].append(exact)
        print(exact)
    return paper_instance
         
def pmc_to_papers(keyword):
    pmc_results = pmc_to_list(keyword)
    paper_instances = list_to_paper_instances(pmc_results, keyword)
    return paper_instances


if __name__=="__main__": #confirmar se está bem TypeError: list_to_paper_instances() takes 1 positional argument but 3 were given
    articles = pmc_to_list('tuberculosis')
    #articles2 = pmc_to_list('Mycobacterium tuberculosis')
    #articles3= pmc_to_list('targets')
    papers = list_to_paper_instances(articles)
    