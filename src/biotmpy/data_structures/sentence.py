class Sentence:

    def __init__(self, string, tokens, passage_type=None, pos_i=-1, pos_f=-1):
        self.string = string
        self.tokens = tokens
        self.passage_type = passage_type
        self.pos_i = pos_i
        self.pos_f = pos_f
        self.doc_id = self.tokens[0].doc_id

    def __str__(self):
        return self.string

    def get_doc_id(self):
        return self.tokens[0].doc_id
