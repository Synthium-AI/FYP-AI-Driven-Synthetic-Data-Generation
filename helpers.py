from dateutil.parser import parse
import pandas as pd

class AutoSyntheticConfigurator:
    def __init__(self, file_path):
        self.data_df = pd.read_csv(file_path)

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
            "max_sequence_len": 'default',
            "sample_len" : 'default',
            "batch_size": 'default',
            "apply_feature_scaling" : True,
            "apply_example_scaling": False,
            "use_attribute_discriminator": False,
            "generator_learning_rate": 1e-4,
            "discriminator_learning_rate": 1e-4,
            "epochs": 500,
            "cuda": True
        }
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