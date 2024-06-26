from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd
import json
import os


class CTGANER:
    """
    Initialize CTGANER instance with given file path
    ### Note: 
    - If in "load_mode" file_path will be assumed to be data_artifact path (.csv file)
    - If not in "load_mode" file_path will be assumed to be model path (.pkl file)
    """
    def __init__(self, file_path, main_config, load_mode=False) -> None:
        if not load_mode:
            self.data_df = pd.read_csv(file_path)
            self.main_config = main_config
            self.metadata = SingleTableMetadata.load_from_dict(main_config["metadata"])
            self.model = CTGANSynthesizer(
                metadata = self.metadata,
                enforce_min_max_values = main_config["enforce_min_max_values"],
                enforce_rounding = main_config["enforce_rounding"],
                locales = main_config["locales"],
                embedding_dim = main_config["embedding_dim"],
                generator_dim = main_config["generator_dim"],
                discriminator_dim = main_config["discriminator_dim"],
                generator_lr = main_config["generator_lr"],
                generator_decay = main_config["generator_decay"],
                discriminator_lr = main_config["discriminator_lr"],
                discriminator_decay = main_config["discriminator_decay"],
                batch_size = main_config["batch_size"],
                discriminator_steps = main_config["discriminator_steps"],
                log_frequency = main_config["log_frequency"],
                verbose = main_config["verbose"],
                epochs = main_config["epochs"],
                pac = main_config["pac"],
                cuda = main_config["cuda"]
            )      
        else:
            model_path = file_path
            # ctgan_config_path = os.path.join(project_directory_path, "ctgan_config.json")
            self.model = CTGANSynthesizer.load(model_path)
            # with open(ctgan_config_path, "r") as json_file:
            #     self.main_config = json.load(json_file)
            # self.metadata = SingleTableMetadata.load_from_dict(main_config["metadata"])

    def train(self):
        self.model.fit(self.data_df)

    def generate_synthetic_data_df(self, num_examples):
        return self.model.sample(num_examples)

    def generate_synthetic_data_csv(self, filename, num_examples, index=False, encoding='utf-8'):
        self.generate_synthetic_data_df(num_examples).to_csv(filename, index = index, encoding=encoding)

    def show_df(self):
        return self.data_df

    def save(self, save_model_file_path):
        self.model.save(save_model_file_path)


if __name__ == "__main__":
    pass
    # model = CTGANER("client/ctgan_model_7af9e00f-c1b6-44bb-bf30-206efc9413f9.pkl",None,True)
    # model.generate_synthetic_data_csv("synthetic_data.csv", 5000)