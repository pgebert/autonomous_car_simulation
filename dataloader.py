from __future__ import print_function, division
import os
import torch
import random as rn
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms

import pandas as pd
from PIL import Image

class SimulationDataset(Dataset):
    """Dataset wrapping input and target tensors for the driving simulation dataset.

    Arguments:
        set (String):  Dataset - train, test
        path (String): Path to the csv file with the image paths and the target values
    """

    def __init__(self, set, csv_path='driving_log.csv', transforms=None):

        self.transforms = transforms

        self.data = pd.read_csv(csv_path, header=None)
        # First column contains the middle image paths
        piv = int(4 / 5 * len(self.data))
        if (set == "test"):
            self.image_paths = np.asarray(self.data.iloc[:piv, 0:3])
        else:
            self.image_paths = np.asarray(self.data.iloc[piv:, 0:3])
        # Fourth column contains the steering angle
        self.targets = np.asarray(self.data.iloc[:, 3])

    def __getitem__(self, index):

         # Get image name from the pandas df
        image_paths = self.image_paths[index]
        # Open image
        image = [Image.open(image_paths[i]) for i in range(3)]
        target = self.targets[index]     

        sample = {'image': image, 'target': target}

        # Get target value
        # target = torch.tensor(float(self.targets[index]))

        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample['image'], sample['target']

    def __len__(self):
        return len(self.image_paths)


if  __name__ =='__main__':
    dataset = SimulationDataset()
    print(dataset.__len__())
    print(dataset.__getitem__(0))
    # print(dataset.__getitem__(0))
    # print(len(dataset.__get_annotations__()))