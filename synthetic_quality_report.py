from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
import pandas as pd
import json
import os

class SyntheticQualityAssurance:
    def __init__(self, original_file_path, synthetic_file_path, model="ctgan") -> None:
        self.data_df = pd.read_csv(original_file_path)
        self.synthetic_data_df = pd.read_csv(synthetic_file_path)
        if model == "dgan":        # Metadata not happy with 'example_id' column
            if 'example_id' in self.synthetic_data_df.columns:
                self.synthetic_data_df.drop('example_id', axis = 1, inplace = True) 
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.data_df)
        self.metadata = self.metadata.to_dict()
        self.report = QualityReport()

    def generate_report(self, save_dir_path=None):
        self.report.generate(self.data_df, self.synthetic_data_df, self.metadata)
        report = self.report.get_info()
        report["overall_score"] = self.report.get_score()
        report["properties_info"] = self.report.get_properties().to_dict()
        if save_dir_path != None:
            report_path = os.path.join(save_dir_path, "synthetic_data_quality_report.json")
            with open(report_path, 'w') as fp:
                json.dump(report, fp, indent=4)
        return report
