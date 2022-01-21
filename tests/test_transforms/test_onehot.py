import unittest
import numpy as np
from dataset4ms.transforms import OneHot

class TestOneHot(unittest.TestCase):
    def test_onehot(self):
        onehot = OneHot(3)
        labels = np.array([0, 2, 1])
        
        onehot_labels = onehot(labels)
        print(onehot_labels)
        expected = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0]], np.int64)
        
        assert np.array_equal(onehot_labels, expected)