import pickle
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from  classes_do_europePMC_wrapper import Paper

with open('paper_instances1.pickle', 'rb') as file:
    paper_instances = pickle.load(file)

#print(paper_instances)

import scispacy
import spacy 

nlp = spacy.load("en_ner_bc5cdr_md")

results= []
for paper in paper_instances:
    doc = nlp(paper.title_abstract)
    
    for entity in doc.ents:
        




        print(entity.text, entity.label_)
        
        paper_results = {
        'title': paper.title,
        'abstract': paper.abstract,
        'entities': []
    }
    for entity in doc.ents:
        paper_results['entities'].append({
            'text': entity.text,
            'label': entity.label_
        })

results.append(paper_results)


with open('results.pickle', 'wb') as file:
    pickle.dump(results, file)


doc = nlp(paper_instances)



