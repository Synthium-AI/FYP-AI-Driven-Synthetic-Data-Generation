from sdv.metadata import SingleTableMetadata
from dateutil.parser import parse
import pandas as pd
from ctgan_model import CTGANER
from dgan_model import DGANER


def synthetic_model_trainer(data_artifact_file_path, model_config, model_type, save_model_file_path, save_model_encoding_mappings_path=None):
    """## Train a synthetic model
    - model_config: dict() or json() object
    - model_type: "ctgan" | "dgan"
    ## Model Requirements:-
    ### CTGAN:
    - save_model_file_path (.pkl)
    ### DGAN:
    - save_model_file_path (.pt)
    - save_model_encoding_mappings_path (.pkl)
    """
    if model_type == "ctgan":
        model_trainer = CTGANER(data_artifact_file_path, model_config)
        model_trainer.train()
        model_trainer.save(save_model_file_path)
    elif model_type == "dgan":
        model_trainer = DGANER(data_artifact_file_path, model_config)
        model_trainer.train()
        model_trainer.save(save_model_file_path, save_model_encoding_mappings_path)

def synthetic_model_data_generator(num_examples, save_synthetic_data_artifact_file_path, model_file_path, model_config, model_type, model_encoding_mappings_path=None):
    if model_type == "ctgan":
        model_loader = CTGANER(model_file_path, model_config, load_mode=True)
        model_loader.generate_synthetic_data_csv(save_synthetic_data_artifact_file_path, num_examples)
    elif model_type == "dgan":
        model_loader = DGANER(model_file_path, model_config, load_mode=True, model_encoding_mappings_path=model_encoding_mappings_path)
        model_loader.generate_synthetic_data_csv(save_synthetic_data_artifact_file_path, num_examples)

class AutoSyntheticConfigurator:
    def __init__(self, file_path):
        self.data_df = pd.read_csv(file_path)

    def get_ctgan_config(self):
        ctgan_main_config = {
            "metadata": None,
            "metadata_is_valid": None,
            "enforce_min_max_values": True,
            "enforce_rounding": True,
            "locales": None,
            "embedding_dim": 128,
            "generator_dim": (256, 256),
            "discriminator_dim": (256, 256),
            "generator_lr": 0.0002,
            "generator_decay": 0.000001,
            "discriminator_lr": 0.0002,
            "discriminator_decay": 0.000001,
            "batch_size": 500,
            "discriminator_steps": 1,
            "log_frequency": True,
            "verbose": True,
            "epochs": 300,
            "pac": 10,
            "cuda": True
        }

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.data_df)
        ctgan_main_config["metadata"] = metadata.to_dict()
        try:
            metadata.validate()
            ctgan_main_config["metadata_is_valid"] = True
        except Exception as e:
            ctgan_main_config["metadata_is_valid"] = False

        return ctgan_main_config

    def get_dgan_config(self):
        dgan_main_config = {
            # Training Configs
            "df_style": "long",
            "time_column": None,
            "feature_columns": None,
            "discrete_columns": None,
            "encodable_columns": None,
            "attribute_columns": None,
            "example_id_column": None,
            # Model Configs
            "max_sequence_len": None,
            "sample_len" : 1,
            "batch_size": None,
            "apply_feature_scaling" : True,
            "apply_example_scaling": False,
            "use_attribute_discriminator": False,
            "generator_learning_rate": 1e-4,
            "discriminator_learning_rate": 1e-4,
            "epochs": 500,
            "cuda": True
        }
        # Set Model Configs
        dgan_main_config["max_sequence_len"] = self.data_df.shape[0]//4
        dgan_main_config["batch_size"] = min(100, self.data_df.shape[1])
        # First check the datetime column
        datetime_candidate = self.detect_datetime_column()
        if datetime_candidate:
            datetime_candidate_name, success_score = datetime_candidate
            dgan_main_config["time_column"] = datetime_candidate_name

        # Now check for feature cols
        features_candidates = self.detect_numeric_columns()
        if features_candidates:
            dgan_main_config["feature_columns"] = features_candidates

        # Now check for encodable cols
        encodable_candidates = self.detect_string_columns()
        if encodable_candidates:
            if dgan_main_config["time_column"] in encodable_candidates:
                encodable_candidates.remove(dgan_main_config["time_column"])
            dgan_main_config["encodable_columns"] = encodable_candidates
            dgan_main_config["feature_columns"] += dgan_main_config["encodable_columns"]

        return dgan_main_config

    def detect_datetime_column(self):
        # Load the CSV file
        df = self.data_df
        # Dictionary to hold the success rate of date parsing for each column
        datetime_success_rate = {}
        # Iterate over each column in the DataFrame
        for column in df.columns:
            total_count = 0
            success_count = 0
            # Attempt to parse each value in the column
            for value in df[column].dropna().unique():
                try:
                    # Try parsing the value as a date
                    parsed_date = parse(str(value), fuzzy=False)
                    success_count += 1
                except (ValueError, TypeError):
                    # If parsing fails, continue to the next value
                    continue
                finally:
                    total_count += 1
            # Calculate the success rate of parsing for the current column
            if total_count > 0:
                success_rate = success_count / total_count
                datetime_success_rate[column] = success_rate
        # Find the column with the highest success rate of date parsing
        datetime_column = max(datetime_success_rate, key=datetime_success_rate.get, default=None)
        # Return the name of the column that most likely represents a datetime
        return datetime_column, datetime_success_rate[datetime_column] if datetime_column else None

    def detect_numeric_columns(self):
        # Load the CSV file
        df = self.data_df

        # List to hold the names of numeric and non-alphabetic columns
        numeric_columns = []

        # Iterate over each column in the DataFrame
        for column in df.columns:
            # Use to_numeric to attempt converting the column, errors='coerce' replaces non-convertible values with NaN
            numeric_series = pd.to_numeric(df[column], errors='coerce')

            # After conversion, if there are no NaN values, it means all values were numeric
            if not numeric_series.isnull().any():
                # Additional check: Ensure there are no alphabetic characters
                # This is a revised approach where we directly analyze the content without replacing
                if not df[column].astype(str).str.contains('[a-zA-Z]').any():
                    numeric_columns.append(column)

        # Return the list of column names that are numeric and non-alphabetic
        return numeric_columns if numeric_columns != [] else None

    def detect_string_columns(self):
        # Load the CSV file
        df = self.data_df
        # List to hold the names of string-like columns
        string_columns = []
        # Iterate over each column in the DataFrame
        for column in df.columns:
            # Initially assume the column is string-like
            is_string_like = True
            # Check if the column is of object type (commonly used for strings in pandas)
            if df[column].dtype == 'object':
                # Check if the majority of the values in the column are non-numeric
                non_numeric_count = 0
                for value in df[column].dropna().unique():
                    if isinstance(value, str) and not value.replace('.', '', 1).isdigit():
                        non_numeric_count += 1
                # Determine if the majority of the column's values are non-numeric
                if non_numeric_count / len(df[column].dropna().unique()) < 0.5:
                    is_string_like = False
            else:
                # If the column's dtype is not 'object', it's less likely to be string-like
                is_string_like = False

            # If the column is determined to be string-like, add it to the list
            if is_string_like:
                string_columns.append(column)
        # Return the list of column names that are string-like
        return string_columns if string_columns != [] else None