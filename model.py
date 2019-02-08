from __future__ import print_function, division

import torch
import torchvision
from torchvision import datasets, models, transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import sys
import os
import copy
from PIL import Image

from dataloader import SimulationDataset
from data_utils import get_weights
from logger import Logger

# Surpress traceback in case of user interrupt
signal.signal(signal.SIGINT, lambda x,y: sys.exit(0))


########################################################################
# Define the network
# ^^^^^^^^^^^^^^^^^^^^

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5)
        self.conv2 = nn.Conv2d(24, 36, 5)
        self.conv3 = nn.Conv2d(36, 48, 5)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 100 * 100, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = self.drop(x)
        x = x.view(-1, 64 * 100 * 100)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

class Model():

    ########################################################################
    # Define configuration, log and network instance
    # ^^^^^^^^^^^^^^^^^^^^

    def __init__(self):
        
        cfg = type('', (), {})()
        cfg.log_dir = "."
        cfg.log_file = "log.json"
        cfg.plot_file = "plot.png"
        cfg.auto_plot = True
        cfg.batch_size = 5
        cfg.test_rate = 1
        cfg.test_epochs = 1
        cfg.train_epochs = 200
        cfg.optimizer = 'sgd'
        cfg.cuda = True

        self.cfg = cfg
        self.log = Logger(cfg)

        self.net = Net()
        if (self.cfg.cuda):
            self.net.cuda()

    ########################################################################
    # Load data
    # ^^^^^^^^^^^^^^^^^^^^

    def loadData(self):       
        
        trainset = SimulationDataset("train", transforms=transforms.Compose([
                transforms.Resize(116),
                transforms.CenterCrop(116),
                # transforms.RandomResizedCrop(200),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
        weights = get_weights(trainset)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.cfg.batch_size, sampler=sampler, num_workers=4)

        testset = SimulationDataset("test", transforms=transforms.Compose([
                transforms.Resize(116),
                transforms.CenterCrop(116),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4)

        # Assert trainset and testset are different
        # assert(not bool(set(trainset.__get_samples__()).intersection(testset.__get_samples__())))

    ########################################################################
    # Helper methods
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Save model in file system
    def saveModel(self):
        print('Saving Model            ')
        torch.save(self.net.state_dict(), 'model.pth')

    # Load model from file system
    def loadModel(self):
        self.net.load_state_dict(torch.load('model.pth'))


    ########################################################################
    # Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    def train(self):

        test_res, tmp_res, best_epoch = 0, 0, 0

        #set train mode
        self.net.train()

        criterion = nn.MSELoss()

        if self.cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        elif self.cfg.optimizer == 'adadelta':
            optimizer = optim.Adadelta(selfnet.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        else:
            optimizer = optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0, dampening=0.0)


        for epoch in range(self.cfg.train_epochs):  # loop over the dataset multiple times

            train_loss, running_loss = 0, 0

            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if (self.cfg.cuda):
                    inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if (self.cfg.cuda):
                    outputs = self.net(inputs).cuda(async=True)
                else:
                    outputs = self.net(inputs)

                # Remove one dimension
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # print statistics
                if i % 100 == 99:    # print every 100 mini-batches
                    self.log.logLoss((epoch+1, running_loss / (i+1)))
                    print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / (i+1)))

            train_loss = running_loss / len(self.trainloader) 
            print('MSE of the network on the traintset: %.6f' % (train_loss))

            if ((epoch + 1) % self.cfg.test_rate == 0):
                    tmp_res = self.test()
                    self.log.logTest((epoch+1, tmp_res))
                    # Check test result over all splits to save best model
                    if (tmp_res < test_res):
                        self.saveModel()
                        test_res = tmp_res
                        best_epoch = epoch+1

        print('Finished Training')
        print('Lowest model MSE: %.6f - in epoch: %d' % (test_res, best_epoch))

    ########################################################################
    # Test the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def test(self):
        # set test mode
        self.net.eval()

        criterion = nn.MSELoss()
        test_loss, running_loss = 0, 0

        for epoch in range(self.cfg.test_epochs):  # loop over the dataset multiple times
            for data in self.testloader:
                inputs, labels = data
                if (self.cfg.cuda):
                    inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                if (self.cfg.cuda):
                    outputs = self.net(inputs).cuda(async=True)
                else:
                    outputs = self.net(inputs)

                # Compute mean squared error
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        if (self.cfg.test_epochs > 0):
            test_loss = running_loss / (len(self.testloader) * self.cfg.test_epochs) 

        print('MSE of the network on the testset: %.6f' % (test_loss))
        # set train mode
        self.net.train()

        return test_loss

    ########################################################################
    # Predict control tensor from image
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def predict(self, image, preloaded=False):
        # set test mode
        self.net.eval()

        if (not preloaded):
            loadModel()
            print('Loaded Model')

        print('Starting Prediction')

        composed=transforms.Compose([
            transforms.Resize(116),
            transforms.CenterCrop(116),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        inputs = composed(image)
        # Add single batch diemension
        inputs = inputs.unsqueeze(0)

        if (self.cfg.cuda):
            inputs = Variable(inputs.cuda(async=True))
        else:
            inputs = Variable(inputs)

        if (self.cfg.cuda):
            outputs = self.net(inputs).cuda(async=True)
        else:
            outputs = self.net(inputs)

        print('Finished Prediction')
        print('Control tensor: %.6f ' % (outputs.item()))

        # set train mode
        self.net.train()

        return outputs.item()

########################################################################
# Main method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

if  __name__ =='__main__':
    model = Model()
    model.loadData()
    model.train()
    # image_path = r'C:\Users\patri\Documents\Python Workspace\autonomous_car_simulation\IMG\center_2019_01_23_19_09_22_763.jpg'
    # image = Image.open(image_path)    
    # model.predict(image)