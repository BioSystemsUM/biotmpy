class Paper:
    def __init__(self, id, title, abstract, source=None, query_keyword=None): 
        self.id = id
        self.title = title
        self.abstract = abstract
        self.source=source
        self.annotations = {'Diseases':[],
                            'Chemicals':[],
                            'Gene_Proteins':[],
                            'Organisms':[]}
        if query_keyword:
            self.query_keyword = query_keyword
