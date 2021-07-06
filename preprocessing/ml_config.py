import sys
sys.path.append('../')
from data_structures.report import Report
from pathlib import Path

class MLConfig:
    
    def __init__(self, classifier, path, scaler=None, feature_selector=None, label_encoder=None, features_cols_names=None):
        self.classifier = classifier
        self.path = path
        self.scaler = scaler
        self.feature_selector = feature_selector
        self.label_encoder = label_encoder
        self.features_cols_names = features_cols_names
        self.report = Report(classifier, file_name=Path(str(self.path)).stem)
    
    def write_report(self):
        self.report.write_report()