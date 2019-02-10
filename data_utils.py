from __future__ import print_function, division
import torch
from torchvision import transforms
import torchvision.transforms.functional as tf
import random

def get_weights(dataset):

    # Get targets
    targets = [dataset.__getitem__(i)[1] for i in range(dataset.__len__())]
    # Count zero value appearance
    count_zeros = len(list(filter(lambda x: x == 0, targets))) 
    # Weights - inverted possibilities
    weights_zeros = float(len(targets) - count_zeros)/len(targets)
    weights_others = float(count_zeros)/len(targets)
    # Weight for each sample                               
    weights = [ weights_zeros if target == 0 else weights_others for target in targets]
    # weights = [ 0.01 if target == 0 else 1.0 for target in targets]

    return weights

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']        
        image = tf.resize(image, self.output_size)
        # Ensure the output size
        image = tf.center_crop(image, self.output_size)
        return {'image': image, 'target': target}

class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']    
        resizeCrop = transforms.RandomResizedCrop(self.output_size[1])    
        image = resizeCrop(image)
        return {'image': image, 'target': target}

class RandomHorizontalFlip(object):
    """Random horizontal flip of image and target control vector"""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        if random.random() > 0.5:
            image = tf.hflip(image)
            target = - target  
        return {'image': image, 'target': target}

class Normalize(object):
    """Normalize the image in a sample: y = (x - mean) / std
    Args:
        mean (list od float): Means of each channel
        std (list of float): Standard deviation of the channels
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = tf.normalize(image, self.mean, self.std)
        return {'image': image, 'target': target}

class ToTensor(object):
    """Convert image and target control vector to tensor."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        target = torch.tensor(float(target))
        image = tf.to_tensor(image)
        return {'image': image, 'target': target}