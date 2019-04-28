from dataloader import SimulationDataset
import matplotlib.pyplot as plt
from PIL import Image

import utils as utils
import torch
from torchvision import transforms
import torchvision.transforms.functional as F


input_shape = (utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH)
dataset = SimulationDataset("train", transforms=transforms.Compose([                 
    utils.RandomCoose(['center']),          
    utils.Preprocess(input_shape),
    utils.ToTensor(),
    utils.Normalize([0.1, 0.4, 0.4], [0.9, 0.6, 0.5])
]))

targets = []


for i in range(dataset.__len__()):
    image, target = dataset.__getitem__(i)
    targets.append(target)
    # plt.imshow(F.to_pil_image(image))
    # plt.title(str(target))
    # plt.show()

plt.hist(targets, 50)
plt.show()

    