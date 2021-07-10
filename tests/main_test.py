import os
import pathlib
import unittest

from movie_classifier.src import InputParameters, Reader, Executor


class MyTestCase(unittest.TestCase):
    def test1(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        input_parameters = InputParameters(title='''Toy Story''',
                                           description='''
        Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. 
        Afraid of losing his place in Andy's heart, Woody plots against Buzz.
         But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their 
         differences.''')
        reader = Reader(input_folder_path=str(pathlib.Path(__file__).parent.parent.resolve()) + "/trained_model/")
        #print(Executor(input_parameters=input_parameters, reader=reader).exe(print_result=False))
        self.assertIn(Executor(input_parameters=input_parameters, reader=reader).exe(print_result=False),
                      reader.read_pkl("genres_subset.txt"))


if __name__ == '__main__':
    unittest.main()
