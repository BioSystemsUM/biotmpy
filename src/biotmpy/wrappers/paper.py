from typing import List, Dict

class Paper:
    def __init__(self, id:str, title:str, abstract:str, source:str=None, date_publication:str = None, query_keyword:str=None):
        """
        Initialize a Paper instance.

        Args:
            id (str): The ID of the paper.
            title (str): The title of the paper.
            abstract (str): The abstract of the paper.
            source (str, optional): The source of the paper. Defaults to None.
            date_publication (str, optional): The publication date of the paper. Defaults to None.
            query_keyword (str, optional): The query keyword associated with the paper. Defaults to None.
        """
        self.id:str = id
        self.title:str = title
        self.abstract:str = abstract
        self.source:str=source
        self.annotations:Dict[str, List[str]] = {
                            'Diseases':[],
                            'Chemicals':[],
                            'Gene_Proteins':[],
                            'Organisms':[]}
        
        self.annotations_scispacy:Dict[str, List[str]] = {
                                        'Diseases':[],
                                        'Chemicals':[],
                                        'Gene_Proteins':[],
                                        'Organisms':[]}
        
        #if date_publication:
        self.date_publication:str = date_publication
        
     
        if query_keyword:
            self.query_keyword = query_keyword

        self.title_abstract:str = self.title + self.abstract
        

    def compare_annotations(self)->None:
        """
        Compare the annotations of the Paper instance and its SciSpacy annotations.
        Prints the differences between the annotations of each category.
        """
        for category in self.annotations:
            annotations1 = set(self.annotations[category])
            annotations2 = set(self.annotations_scispacy[category])
            
            if annotations1 == annotations2:
                print(f"Annotations for '{category}' are the same.")
            else:
                different_annotations = annotations1.symmetric_difference(annotations2)
                print(f"Different annotations for '{category}': {different_annotations}")
  
    