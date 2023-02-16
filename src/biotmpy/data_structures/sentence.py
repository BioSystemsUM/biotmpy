class Sentence:
    """
    Sentence object. It contains a list of Token objects.
    """
    def __init__(self, string, tokens, passage_type=None, pos_i=-1, pos_f=-1):
        """
        Sentence object constructor.

        :param string: string of the sentence
        :param tokens: list of Token objects
        :param passage_type: type of the passage. Can be 't' for title or 'a' for abstract.
        :param pos_i: initial position of the sentence in the document
        :param pos_f: final position of the sentence in the document

        :return: Sentence object
        """
        self.string = string
        self.tokens = tokens
        self.passage_type = passage_type
        self.pos_i = pos_i
        self.pos_f = pos_f
        self.doc_id = self.tokens[0].doc_id

    def __str__(self):
        """
        String representation of the Sentence object.

        :return: string representation of the Sentence object
        """
        return self.string

    def get_doc_id(self):
        """
        Gets the ID of the document to which the sentence belongs.

        :return: ID of the document to which the sentence belongs
        """
        return self.tokens[0].doc_id
