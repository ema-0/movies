"""
Unit tests
"""
import os
import pathlib
import unittest

from movie_classifier.src import InputParameters, Reader, Executor


class MoviesTests(unittest.TestCase):
    """
    Class to define and easily run all unit tests
    """

    def test_accepted_value(self):
        """
        Tests if the output genre is among the possible ones
        :return:
        """
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        input_parameters = InputParameters(
            title="""Toy Story""",
            description="""
        Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. 
        Afraid of losing his place in Andy's heart, Woody plots against Buzz.
         But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their 
         differences.""",
        )
        reader = Reader(
            input_folder_path=str(pathlib.Path(__file__).parent.parent.resolve())
            + "/trained_model/"
        )
        self.assertIn(
            Executor(input_parameters=input_parameters, reader=reader).exe(
                print_result=False
            ),
            reader.read_pkl("genres_subset.txt"),
        )

    def test_correct_value(self):
        """
        Tests if the output genre is the expected one
        :return:
        """
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        input_parameters = InputParameters(
            title="""Jumanji""",
            description='''
        When siblings Judy and Peter discover an enchanted board game that opens the door to a magical world, 
        they unwittingly invite Alan -- an adult who's been trapped inside the game for 26 years -- 
        into their living room. Alan's only hope for freedom is to finish the game, which proves risky as all three
         find themselves running from giant rhinoceroses, evil monkeys and other terrifying creatures."''',
        )
        reader = Reader(
            input_folder_path=str(pathlib.Path(__file__).parent.parent.resolve())
            + "/trained_model/"
        )
        self.assertEqual(
            Executor(input_parameters=input_parameters, reader=reader).exe(
                print_result=False
            ),
            "Adventure",
        )


if __name__ == "__main__":
    unittest.main()
