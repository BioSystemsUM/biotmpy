

class Token:

    def __init__(self, string, passage_type=None, pos_i=-1, pos_f=-1, sent_id=None, doc_id=None):
        self.string = string
        self.passage_type = passage_type
        self.pos_i = pos_i
        self.pos_f = pos_f
        self.sent_id = sent_id
        self.doc_id = doc_id
        
    def __str__(self):
        return self.string

    def __eq__(self, other):
        if self.pos_i == other.pos_i and self.sent_id == other.sent_id and self.doc_id == other.doc_id:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
         
    def __lt__(self, other):
        if self.doc_id < other.doc_id:
            return True
        elif self.doc_id == other.doc_id:
            if self.sent_id < other.sent_id:
                return True
            elif self.sent_id == other.sent_id:
                if self.pos_i < other.pos_i:
                    return True
        return False

    def __gt__(self, other):
        if self.doc_id > other.doc_id:
            return True
        elif self.doc_id == other.doc_id:
            if self.sent_id > other.sent_id:
                return True
            elif self.sent_id == other.sent_id:
                if self.pos_i > other.pos_i:
                    return True
        return False

    def __le__(self, other):
        if self.__lt__(other) or self.__eq__(other):
            return True
        return False

    def __ge__(self, other):
        if self.__gt__(other) or self.__eq__(other):
            return True
        return False
