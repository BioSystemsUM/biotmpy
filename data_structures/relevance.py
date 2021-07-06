

class Relevance:
    
    def __init__(self, label, doc_id, description, conf_score=None):
        self.label = label
        self.id = doc_id
        self.description = description
        self.conf_score = conf_score