import numpy as np
import unittest
from dataset4ms.core import Dataset, DataProcess
from dataset4ms.transforms.basic import OneHot

class TestDataProcess(unittest.TestCase):
    def setUp(self) -> None:
        class TestDataset(Dataset):
            def __init__(self) -> None:
                super().__init__()
                self.labels = [0, 1, 0, 0, 1]
                self.data = np.random.randn(5, 10)
            
            def __getitem__(self, index):
                return self.data[index], self.labels[index]
            
            def __len__(self):
                return len(self.labels)
        
        self.test_data = TestDataset()
        return super().setUp()

    def test_dataprocess(self):
        dataprocess = DataProcess(self.test_data, column_names=['data', 'label'])
        dataprocess.map(OneHot(2), 'label')
        print(dataprocess[0])