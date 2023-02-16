class Report:
    """
    Class to create a report for a ML classifier. The report is saved in a .txt file by Default. The report contains the
    following information: the classifier used, the cross validation results, the training prediction results,
    the test prediction results and the features used.

    :param classifier: classifier used
    :param file_name: name of the file where the report will be saved
    :param file_type: type of the file where the report will be saved

    :return: Report object
    """
    def __init__(self, classifier, file_name, file_type='.txt'):
        """
        Constructor of the class.

        :param classifier: classifier used
        :param file_name: name of the file where the report will be saved
        :param file_type: type of the file where the report will be saved
        """
        self.file_name = file_name
        self.file_type = file_type
        self.path = '../tests/reports/report_' + self.file_name + self.file_type
        self.to_report = {'Classifier': str(classifier) + '\n' + '_' * 100 + '\n', 'Cross Validation': '',
                          'Training Prediction': '', 'Test Prediction': '', 'Features Used': ''}

    def append_to_file(self, text):
        """
        Appends a text to the report file.

        :param text: text to append

        :return: None
        """
        with open(self.path, 'a') as file:
            file.write(text)

    def file_contains(self, text):
        """
        Checks if the report file contains a text.

        :param text: text to check

        :return: True if the text is in the file, False otherwise
        """
        with open(self.path) as file:
            if str(text) in file.read():
                return True
        return False

    def write_report(self):
        """
        Writes the report into a file.

        :return: None
        """
        with open(self.path, 'w') as file:
            for string, attribute in self.to_report.items():
                file.write(string + '\n' + attribute + '\n')
