from unittest import TestCase

from biotmpy.wrappers.bioc_wrapper import bioc_to_docs


class TestBiocWrapper(TestCase):
    def setUp(self):
        self.bioc_file = './tests/data/PMtask_Triage_TrainingSet.xml'

    def test_bioc_to_docs(self):
        docs = bioc_to_docs(self.bioc_file)
