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
        
        self.annotations_scispacy = {'Diseases':[],
                                        'Chemicals':[],
                                        'Gene_Proteins':[],
                                        'Organisms':[]}
        
     
        if query_keyword:
            self.query_keyword = query_keyword

        self.title_abstract = self.title + self.abstract
        

    def compare_annotations(self):
        for category in self.annotations:
            annotations1 = set(self.annotations[category])
            annotations2 = set(self.annotations_scispacy[category])
            
            if annotations1 == annotations2:
                print(f"Annotations for '{category}' are the same.")
            else:
                different_annotations = annotations1.symmetric_difference(annotations2)
                print(f"Different annotations for '{category}': {different_annotations}")
  
