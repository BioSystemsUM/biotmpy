class Report:

    def __init__(self, classifier, file_name, file_type='.txt'):
        self.file_name = file_name
        self.file_type = file_type
        self.path = '../tests/reports/report_' + self.file_name + self.file_type
        self.to_report = {'Classifier': str(classifier) + '\n' + '_' * 100 + '\n', 'Cross Validation': '',
                          'Training Prediction': '', 'Test Prediction': '', 'Features Used': ''}

    def append_to_file(self, text):
        with open(self.path, 'a') as file:
            file.write(text)

    def file_contains(self, text):
        with open(self.path) as file:
            if str(text) in file.read():
                return True
        return False

    def write_report(self):
        with open(self.path, 'w') as file:
            for string, attribute in self.to_report.items():
                file.write(string + '\n' + attribute + '\n')
