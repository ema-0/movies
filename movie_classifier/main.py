import pickle
from keras.models import load_model
import numpy as np
import argparse
import os
import json
import pathlib


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", help="Movie title", required=True)
    parser.add_argument("--description", help="Movie description", required=True)
    args = parser.parse_args()

    folder_path = str(pathlib.Path(__file__).parent.resolve()) + '/model/'

    with open(folder_path + 'genres_subset.txt', "rb") as fp:
        genres_subset = pickle.load(fp)

    with open(folder_path + 'tokenizer.txt', "rb") as fp:
        tokenizer = pickle.load(fp)
    num_word = tokenizer.__dict__['num_words']

    model = load_model(folder_path + 'nn/')

    tokenized_description = tokenizer.texts_to_sequences([args.description.split()])

    vectorized_description = np.zeros((1, num_word), bool)
    vectorized_description[0, tokenized_description] = True

    probabilities = model.predict(vectorized_description.reshape(1, num_word))

    estimated_genre = genres_subset[np.argmax(probabilities)]

    output = {"title": args.title, "description": args.description, "genre": estimated_genre}

    print(json.dumps(output, indent=4))
