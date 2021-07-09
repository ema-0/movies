import unittest

from movie_classifier.main import sum_first_n


class MyTestCase(unittest.TestCase):
    def test_sumg(self):

        self.assertEqual(sum_first_n(3), 6)
        self.assertEqual(sum_first_n(5), 20)


if __name__ == '__main__':
    unittest.main()
