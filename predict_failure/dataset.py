"""dataset.py
Representation of a dataset used to train a predictive network.
"""

import torch

from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, labeled_data):
        """Stores the labeled data that make up this dataset

        :param labeled_data: List of tuples [(X1,y1), (X2,y2), ...]
        """
        self.labeled_data = labeled_data

    def __len__(self):
        """Returns the length of the dataset

        :return: The length of the dataset
            :rtype: int
        """
        return len(self.labeled_data)

    def __getitem__(self, item):
        """Gets an item from the dataset

        :param item: Index of the example to get
        :return: Tuple of the form (X, y)
            :rtype: Two element tuple
        """
        return (self.labeled_data[item][0],
                torch.tensor(self.labeled_data[item][1]))
