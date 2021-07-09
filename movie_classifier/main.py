"""
    Movie classifier is a multiclass classifier for movies genre.
    Based on the plot description, it infers the genres between 10 different labels.
    The logic is based on encoding the movies description using work vectorization and a
    subsequent fully connected neural network.
"""
import os
import pathlib
from src import InputParameters, Reader, Executor, CliInterface

if __name__ == "__main__":
    # removing Keras warning about training performances
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # managing of the CLI interface, defining a parser and getting the input values from cli
    parser = CliInterface.get_parser()
    args = parser.parse_args()

    # saving external input values in a specific class instance
    input_parameters = InputParameters(title=args.title, description=args.description)

    # initializing reader class to load the trained model, genres list and tokenizer to encode the texts
    reader = Reader(input_folder_path=str(pathlib.Path(__file__).parent.parent.resolve()) + "/trained_model/")

    # execute
    Executor(input_parameters=input_parameters, reader=reader).exe()
