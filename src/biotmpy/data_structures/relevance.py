class Relevance:
    """
    Class to represent a relevance of a document to a topic.
    """

    def __init__(self, label, doc_id, topic=None, conf_score=None):
        """
        Constructor of the class.

        :param label: Label of the relevance (0, 1, 2, 3, 4, 5)
        :param doc_id: ID of the document
        :param topic: Topic of the document
        :param conf_score: Confidence score of relevance predicted by a model
        """
        self.label = label
        self.id = doc_id
        self.topic = topic
        self.conf_score = conf_score

