

class Token:
    """
    Token object. It contains the string of the token, the type of the passage to which it belongs, the initial and
    final position of the token in the sentence, the ID of the sentence to which it belongs and the ID of the
    document to which it belongs.
    """

    def __init__(self, string, passage_type=None, pos_i=-1, pos_f=-1, sent_id=None, doc_id=None):
        """
        Token object constructor.

        :param string: string of the token
        :param passage_type: type of the passage. Can be 't' for title or 'a' for abstract.
        :param pos_i: initial position of the token in the sentence
        :param pos_f: final position of the token in the sentence
        :param sent_id: ID of the sentence to which the token belongs
        :param doc_id: ID of the document to which the token belongs

        :return: Token object
        """
        self.string = string
        self.passage_type = passage_type
        self.pos_i = pos_i
        self.pos_f = pos_f
        self.sent_id = sent_id
        self.doc_id = doc_id
        
    def __str__(self):
        """
        String representation of the Token object.

        :return: string representation of the Token object
        """
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
