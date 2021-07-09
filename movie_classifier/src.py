import argparse
import json
import pickle
from keras.models import load_model
import numpy as np


class CliInterface:
    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--title", help="Movie title", required=True)
        parser.add_argument("--description", help="Movie description", required=True)
        return parser


class InputParameters:

    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description


class Reader:

    def __init__(self, input_folder_path: str):
        self.input_folder_path = input_folder_path

    def read_pkl(self, file_path: str):
        with open(self.input_folder_path + file_path, "rb") as fp:
            x = pickle.load(fp)
        return x

    def read_model(self, model_folder):
        return load_model(self.input_folder_path + model_folder)


class Executor:

    def __init__(self, input_parameters: InputParameters, reader: Reader):
        self.input_parameters = input_parameters
        self.reader = reader

    def predict(self, genres_subset, tokenizer, model):
        tokenized_description = tokenizer.texts_to_sequences([self.input_parameters.description.split()])
        num_word = tokenizer.__dict__["num_words"]
        vectorized_description = np.zeros((1, num_word), bool)
        vectorized_description[0, tokenized_description] = True

        probabilities = model.predict(vectorized_description.reshape(1, num_word))

        estimated_genre = genres_subset[np.argmax(probabilities)]
        return estimated_genre

    def exe(self, print_result: bool = True):
        genres_subset = self.reader.read_pkl("genres_subset.txt")
        tokenizer = self.reader.read_pkl("tokenizer.txt")
        num_word = tokenizer.__dict__["num_words"]
        model = self.reader.read_model("nn/")

        estimated_genre = self.predict(genres_subset, tokenizer, model)

        output = {
            "title": self.input_parameters.title,
            "description": self.input_parameters.description,
            "genre": estimated_genre,
        }

        if print_result:
            print(json.dumps(output, indent=4))
        return output
