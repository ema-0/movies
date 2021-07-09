"""
    Movie classifier is a multiclass classifier for movies genre.
    Based on the plot description, it infers the genres between 10 different labels.
    The logic is based on encoding the movies description using work vectorization and a
    subsequent fully connected neural network.
"""
import argparse
import os
import pathlib
from src import InputParameters, Reader, Executor

if __name__ == "__main__":
    # removing Keras warning about training performances
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # reading input passed via CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", help="Movie title", required=True)
    parser.add_argument("--description", help="Movie description", required=True)
    args = parser.parse_args()

    # saving it in a specific class instance
    input_parameters = InputParameters(title=args.title, description=args.description)

    # initializing reader class to load the trained model, genres list and tokenizer to encode the texts
    reader = Reader(input_folder_path=str(pathlib.Path(__file__).parent.parent.resolve()) + "/trained_model/")

    # execute
    Executor(input_parameters=input_parameters, reader=reader).exe()
