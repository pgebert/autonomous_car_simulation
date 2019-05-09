from __future__ import print_function, division
import os
import torch
import random as rn
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms

import torchvision.transforms.functional as F

import pandas as pd
from PIL import Image

import utils as utils

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
        # Fourth column contains the steering angle
        start = 0
        end = int(4 / 5 * len(self.data))

        if (set == "test"):
            start = end 
            end = len(self.data)

        self.image_paths = np.array(self.data.iloc[start:end, 0:3])
        self.targets = np.array(self.data.iloc[start:end, 3]) 

        # Preprocess and filter data
        self.targets = gaussian_filter1d(self.targets, 2)      
        
        bias = 0.03
        self.image_paths = [image_path for image_path, target in zip(self.image_paths, self.targets) if abs(target) > bias]
        self.targets = [target for target in self.targets if abs(target) > bias]

    def __getitem__(self, index):

         # Get image name from the pandas df
        image_paths = self.image_paths[index]
        # Open image
        image = [Image.open(image_paths[i]) for i in range(3)]
        target = self.targets[index]     

        sample = {'image': image, 'target': target}

        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        if self.transforms is not None:
            sample = self.transforms(sample)

        # plt.imshow(F.to_pil_image(sample['image']))
        # plt.title(str(sample['target']))
        # plt.show()
        
        return sample['image'], sample['target']

    def __len__(self):
        return len(self.image_paths)


if  __name__ =='__main__':

    input_shape = (utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH)
    dataset = SimulationDataset("train", transforms=transforms.Compose([                 
                utils.RandomCoose(['center']),          
                utils.Preprocess(input_shape),
                utils.RandomHorizontalFlip(),
                utils.ToTensor(),
                utils.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
    print(dataset.__len__())
    print(dataset.__getitem__(0)[0].size())

    for c in range(3):
        for i in range(dataset.__len__()):
            print(dataset.__getitem__(i)[c].mean())
            print(dataset.__getitem__(i)[c].std())
    # print(dataset.__getitem__(0))
    # print(len(dataset.__get_annotations__()))