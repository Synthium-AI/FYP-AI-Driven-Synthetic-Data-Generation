from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType
from gretel_synthetics.timeseries_dgan.structures import ProgressInfo
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
import numpy as np
import torch
import pickle
import json
import os


class DGANER:
    def __init__(self, file_path, main_config="load_mode", project_directory_path=None) -> None:
        if main_config != "load_mode":
            self.main_config = main_config
            self.encodable_encoding_mappings = {}
        else:
            dgan_config_path = os.path.join(project_directory_path, "dgan_config.json")
            model_encoding_mappings_path = os.path.join(project_directory_path, "encoding_mappings.pkl")
            with open(dgan_config_path, "r") as json_file:
                self.main_config = json.load(json_file)
            with open(model_encoding_mappings_path, "rb") as pickle_file:
                self.encodable_encoding_mappings = pickle.load(pickle_file)

        self.df_style = self.main_config["df_style"]
        self.example_id_column = self.main_config["example_id_column"]
        self.feature_columns = self.main_config["feature_columns"]
        self.attribute_columns = self.main_config["attribute_columns"]
        self.discrete_columns = self.main_config["discrete_columns"]
        self.encodable_columns = self.main_config["encodable_columns"]
        self.time_column = self.main_config["time_column"]

        self.data_df = pd.read_csv(file_path)
        self.model = DGAN(DGANConfig(
            # max_sequence_len = self.data_df.shape[0] if self.main_config["max_sequence_len"] == 'default' else self.main_config["max_sequence_len"],
            max_sequence_len = self.data_df.shape[0]//2 if self.main_config["max_sequence_len"] == 'default' else self.main_config["max_sequence_len"],
            # max_sequence_len = 10 if self.main_config["max_sequence_len"] == 'default' else self.main_config["max_sequence_len"],
            sample_len = 1 if self.main_config["sample_len"] == 'default' else self.main_config["sample_len"],
            batch_size = min(100, self.data_df.shape[1]) if self.main_config["batch_size"] == 'default' else self.main_config["batch_size"],
            apply_feature_scaling = self.main_config["apply_feature_scaling"],
            apply_example_scaling = self.main_config["apply_example_scaling"],
            use_attribute_discriminator = self.main_config["use_attribute_discriminator"],
            generator_learning_rate = self.main_config["generator_learning_rate"],
            discriminator_learning_rate = self.main_config["discriminator_learning_rate"],
            epochs = self.main_config["epochs"],
            cuda = self.main_config["cuda"]
            ))

        if main_config == "load_mode":
            model_path = os.path.join(project_directory_path, "model.pt")
            self.model = self.model.load(model_path)

    def train(self):
        encoder = OrdinalEncoder()
        self.encodable_encoding_mappings = {}
        for column in self.encodable_columns:
            # Encode the column
            self.data_df[column] = encoder.fit_transform(self.data_df[[column]])
            # Store the mapping (encoder.categories_ contains the original values)
            self.encodable_encoding_mappings[column] = encoder.categories_[0]

        self.model.train_dataframe(
            self.data_df,
            df_style = self.df_style,
            example_id_column = self.example_id_column,
            feature_columns = self.feature_columns,
            attribute_columns = self.attribute_columns,
            discrete_columns = self.discrete_columns,
            time_column = self.time_column,
            progress_callback = self.progress_callbacker
            )

    def generate_synthetic_data_df(self, num_examples):
        actual_num_examples = 0
        while (actual_num_examples*self.main_config["max_sequence_len"]) < num_examples:
            print(actual_num_examples)
            actual_num_examples+=1
        if self.encodable_columns:
            # Create a copy to avoid modifying the original encoded_df
            reverted_df = self.model.generate_dataframe(actual_num_examples)
            print(reverted_df)
            # Iterate over the encoding mappings and revert each column
            for column, mapping in self.encodable_encoding_mappings.items():
                reverted_df[column] = reverted_df[column].astype(int)
                print(reverted_df)
                # Create a mapping from encoded value back to original value
                inverse_mapping = {i: val for i, val in enumerate(mapping)}
                # Replace encoded values with original values using the inverse mapping
                reverted_df[column] = reverted_df[column].map(inverse_mapping)

            # Return the DataFrame with reverted encoding
            return reverted_df
        else:
            return self.model.generate_dataframe(actual_num_examples)

    def generate_synthetic_data_csv(self, filename, num_examples, index=False, encoding='utf-8'):
        self.generate_synthetic_data_df(num_examples).to_csv(filename, index = index, encoding=encoding)

    def progress_callbacker(self, progress_callback:ProgressInfo):
        progress = f"Epoch {progress_callback.epoch}/{progress_callback.total_epochs}, Batch {progress_callback.batch}/{progress_callback.total_batches}: {int(progress_callback.frac_completed * 100)}%"
        print(progress)
        return progress

    def show_df(self):
        return self.data_df

    def save(self, directory_path):
        model_path = os.path.join(directory_path, "model.pt")
        dgan_config_path = os.path.join(directory_path, "dgan_config.json")
        model_encoding_mappings_path = os.path.join(directory_path, "encoding_mappings.pkl")
        self.model.save(model_path)
        with open(dgan_config_path, 'w') as fp:
            json.dump(self.main_config, fp)
        with open(model_encoding_mappings_path, 'wb') as handle:
            pickle.dump(self.encodable_encoding_mappings, handle, protocol=pickle.HIGHEST_PROTOCOL)

