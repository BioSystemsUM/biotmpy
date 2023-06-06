import pickle
import scispacy
import spacy 

# path_file = "C:\\Users\\catarina\\OneDrive - Universidade do Minho\\Mestrado\\PROJETO\\biotmpy\\instances_annotated.pkl"
# with open(path_file, 'rb') as file:
    paper_instances = pickle.load(file)

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
file_path = 'annotations.pickle'
paper_instances = load_paper_instances(file_path)

# Extrair entidades dos papers usando spaCy
entity_results = extract_entities(paper_instances)

# Imprimir resultados
for result in entity_results:
    print('Title:', result['title'])
    print('Abstract:', result['abstract'])
    print('Entities:')
    for entity in result['entities']:
        print(entity['text'], entity['label'])
    print()


    





