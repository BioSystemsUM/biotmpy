import requests
import xml.etree.ElementTree as ET
import re
import pickle
from paper import Paper
import os
import pathlib
import tqdm
import time
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed

## API

def pmc_to_list(term): 
    """
    Function that returns a list of articles from Europe PMC that match the provided search term (keyword).

    Args:
    - term: search keyword to extract the articles from Europe PMC.

    Returns:
    A list of dictionaries representing the Europe PMC articles that match the search term.
    Each dictionary contains information about an article, such as ID, title, abstract, etc."""

    results = []
    print('getting papers')
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(TITLE:"{term}" OR ABSTRACT:"{term}")&pageSize=1000&resultType=core&format=json' 

    nextPage = True
    while nextPage:
        r = requests.get(url=url) 
        data = r.json()

        results += data['resultList']['result']
        try:
            url = data['nextPageUrl']
        except:
            nextPage = False

    return results


def list_to_paper_instances(paper_list, keyword=None):  
    """
    Function that converts a list of dictionaries representing articles into instances of the Paper class.
    
    Args:
    - paper_list: A list of dictionaries representing articles.
    - keyword: Keyword associated with the articles.

    Returns:
    A list of Paper instances created from the provided article dictionaries.
    Only articles containing the keyword "target" in their title or abstract will be included in the result."""

    paper_instances = []
    for paper_dict in paper_list:
        pmid = paper_dict['id']
        title = paper_dict['title']
        abstract = paper_dict.get('abstractText', '')  
        source = paper_dict.get('source')
        date_publication = paper_dict.get('dateOfPublication', None)
        paper_obj = Paper(pmid, title, abstract, source, date_publication, query_keyword=keyword)

        if 'target' in paper_obj.title_abstract:
            paper_instances.append(paper_obj)

    return paper_instances


def get_annotations_paper_instances(paper_instances, n_workers=10, max_attempts=3):
    """ 
    Function that retrieves annotations for a list of Paper instances using multiple threads.

    Args:
    - paper_instances: A list of Paper instances for which to retrieve annotations.
    - n_workers: The number of worker threads to use for concurrent annotation retrieval.
    - max_attempts: The maximum number of attempts to retrieve annotations for each paper.

    Returns:
    A list of annotations corresponding to the provided Paper instances.
    The annotations are retrieved using multiple threads for improved efficiency."""

    instances_annotations = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for paper_instance in paper_instances:
                future = executor.submit(get_annotations, paper_instance, session, max_attempts)
                futures.append(future)

            number_of_papers = len(paper_instances)
            start_time = time.time()

            for i, future in enumerate(as_completed(futures), 1):
                annotations = future.result()
                instances_annotations.append(annotations)
                elapsed_time = time.time() - start_time
                print(f'Retrieved: {i}/{number_of_papers} papers | Duration: {elapsed_time:.2f} seconds', end='\r')

    print()  
    return instances_annotations

#API ANNOTATIONS 

def get_annotations(paper_instance, session, max_attempts=5):
    """
    Function that retrieves annotations for a specific Paper instance from the Europe PMC annotations API.

    Args:
    - paper_instance: A Paper instance for which to retrieve annotations.
    - session: A requests.Session object for making HTTP requests.
    - max_attempts: The maximum number of attempts to retrieve annotations for the paper.

    Returns:
    The updated Paper instance with annotations added, if annotations were successfully retrieved.
    Otherwise, returns the original Paper instance without any modifications.
    """
    source = paper_instance.source
    paper_id = paper_instance.id
    annotations_url = f'https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds={source}%3A{paper_id}&pageSize=1000&type=Organisms%2CGene_Proteins%2CDiseases%2CChemicals'

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(annotations_url)
            response.raise_for_status()
            annotations_data = response.json()
            if annotations_data:
                paper_instance = annotations_to_dict(annotations_data[0], paper_instance)
            return paper_instance

        except RequestException as err:
            print(f"Attempt {attempt} failed. Retrying... Error: {err}")
            time.sleep(5)

    print(f"Maximum number of attempts reached for paper {paper_id}. Skipping...")
    return paper_instance


def annotations_to_dict(json_anotations, paper_instance):
    """
    Function that converts annotations from a JSON format to a dictionary and adds them to a Paper instance.

    Args:
    - json_annotations: JSON data containing the annotations for a paper.
    -  paper_instance: The Paper instance to which the annotations should be added.


    Returns:
    The updated Paper instance with the annotations added.
    """

    for annotation in json_anotations['annotations']:
        category = annotation['type']
        exact = annotation['exact'].lower().rstrip()
        section = annotation.get('section', '')
        if 'Title' in section or 'Abstract' in section:
            if exact not in paper_instance.annotations[category]:
                paper_instance.annotations[category].append(exact)
    return paper_instance



         
def pmc_to_papers(keyword):
    """
    Function that retrieves a list of Paper instances from Europe PMC based on a given keyword.

    Args:
    - keyword: Keyword used to search for papers on Europe PMC.

    Returns:
        A list of Paper instances representing the papers obtained from Europe PMC. Each Paper instance contains information such as ID, title, abstract, etc.
    """
    pmc_results = pmc_to_list(keyword)
    paper_instances = list_to_paper_instances(pmc_results, keyword)
    return paper_instances


# if __name__=="__main__": 
#    articles = pmc_to_list('tuberculosis')
#    papers = list_to_paper_instances(articles)


    