"""
Script containing all the classes to interface with CLI, read input data and execute the logics
"""
import argparse
import json
import pickle
from keras.models import load_model
import numpy as np


class CliInterface:
    """
    Class to interact with the command line inputs
    """

    @staticmethod
    def get_parser():
        """
        Define the parser from argparse package.
        :return: parser object
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--title", help="Movie title", required=True)
        parser.add_argument("--description", help="Movie description", required=True)
        return parser


class InputParameters:
    """
    Class to contain and make available everywhere the parameter used as input by the model
    """

    def __init__(self, title: str, description: str):
        """
        Saving params in class attributes
        :param title: title of the movie
        :param description: movie description
        """
        self.title = title
        self.description = description


class Reader:
    """
    Class defining a simple reader to load local files
    """

    def __init__(self, input_folder_path: str):
        """
        Saving local folder path in a class attribute
        :param input_folder_path: root folder containing the model, tokenizer and genres list
        """
        self.input_folder_path = input_folder_path

    def read_pkl(self, file_path: str):
        """
        Load a Python object saved as pickle file.
        :param file_path: file relative path, including the name and extension
        :return: the object read
        """
        with open(self.input_folder_path + file_path, "rb") as fp:
            x = pickle.load(fp)
        return x

    def read_model(self, model_folder):
        """
        Load Keras model.
        :param model_folder: relative path of the folder in which the model is saved
        :return: the model
        """
        return load_model(self.input_folder_path + model_folder)


class Executor:
    """
    The executor is the object responsible for triggering the run
    """

    def __init__(self, input_parameters: InputParameters, reader: Reader):
        """
        To initialize the executor, we need to define a way to read the model and to have its input values
        :param input_parameters:
        :param reader:
        """
        self.input_parameters = input_parameters
        self.reader = reader

    def predict(self, genres_subset, tokenizer, model):
        """
        Core function: computes the probability associated to each possible genre and predict the most probable
        :param genres_subset: list of possible outcome
        :param tokenizer: tokenizer object to encode movie description string
        :param model: Keras NN to compute the probabilities
        :return: the most probable genre for the given movie description
        """
        tokenized_description = tokenizer.texts_to_sequences(
            [self.input_parameters.description.split()]
        )
        num_word = tokenizer.__dict__["num_words"]
        vectorized_description = np.zeros((1, num_word), bool)
        vectorized_description[0, tokenized_description] = True

        probabilities = model.predict(vectorized_description.reshape(1, num_word))

        estimated_genre = genres_subset[np.argmax(probabilities)]
        return estimated_genre

    def exe(self, print_result: bool = True):
        """
        Run the program.
        :param print_result: boolean variable to print the outcome
        :return: the output of predict function
        """
        genres_subset = self.reader.read_pkl("genres_subset.txt")
        tokenizer = self.reader.read_pkl("tokenizer.txt")
        model = self.reader.read_model("nn/")

        estimated_genre = self.predict(genres_subset, tokenizer, model)

        output = {
            "title": self.input_parameters.title,
            "description": self.input_parameters.description,
            "genre": estimated_genre,
        }

        if print_result:
            print(json.dumps(output, indent=4))
        return estimated_genre
