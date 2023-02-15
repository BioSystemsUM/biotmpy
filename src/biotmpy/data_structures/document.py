class Document:

    def __init__(self, sentences):
        self.sentences = sentences
        self.title_string = ''
        self.title_tokens = []
        self.abstract_string = ''
        self.abstract_tokens = []
        self.fulltext_string = ''
        self.fulltext_tokens = []
        self.raw_title = ''
        self.raw_abstract = ''
        self.compute_attributes()
        self.tokens_string = self.get_all_tokens_strings()
        self.id = self.sentences[0].get_doc_id()

    def compute_attributes(self):
        if self.sentences:
            for s in self.sentences:
                self.fulltext_string += s.string + ' '
                self.fulltext_tokens += s.tokens
                if s.passage_type == 't':
                    self.title_string += s.string + ' '
                    self.title_tokens += s.tokens
                else:
                    self.abstract_string += s.string + ' '
                    self.abstract_tokens += s.tokens
            self.title_string = self.title_string.strip()
            self.abstract_string = self.abstract_string.strip()
            self.fulltext_string = self.fulltext_string.strip()

    def get_section_offset(self, passage_type):
        for t in self.fulltext_tokens:
            if t.passage_type == passage_type:
                return t.pos_i
        return -1

    def get_all_tokens_strings(self) -> object:
        string_tokens = []
        for token in self.fulltext_tokens:
            string_tokens.append(token.string)
        return string_tokens
