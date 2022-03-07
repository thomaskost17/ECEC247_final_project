'''
  File: solver.py
 
  Author: Thomas Kost, Mark Schelbe, Zichao Xian, Trishala Chari
  
  Date: 02 March 2022
  
  @brief This file will define our solver object for training, testing, and validation
'''
from xmlrpc.client import boolean
import torch, torchaudio, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 

class Solver():
    def __init__(self, num_epocs:int, NN:nn.module, optimizer, LR_scheduler:optim.lr_scheduler,
                         criterion, verbose:bool=True)->None:
        '''
           @breif Instantaition function for a solver, will store all relevant huper parameters the network is being trained with
           @return None
           @param num_epocs number of epocs the solver will train the network for
           @param NN network architecture
           @param optimizer pytorch optimizer
           @param LR_scheduler LR scheculer for pytorch optimizer
           @param criterion pytorch loss function for classification
        '''
        self.optimizer = optimizer
        self.LR_scheduler = LR_scheduler
        self.criterion = criterion
        self.num_epocs = num_epocs
        self.net = NN
        self.verbose = verbose
        self.best_validation_accuracy = 0.0

    def train(self, trainloader, validloader):
        '''
           @breif train the provided nerual net
           @return None
           @param trainloader: itterable training set dataloader
           @param validloader: itterable validation set dataloader
        '''
        for epoch in range(self.num_epochs):
            for i, data in enumerate(trainloader,0):
                inputs,labels = data
                outputs = self.NN.forward(inputs) #forward pass
                self.optimizer.zero_grad() #calculate the gradient, manually setting to 0
                loss = self.criterion(outputs, labels.reshape(labels.size(0),).type(torch.long))
            
                loss.backward() #calculates the loss of the loss function
            
                self.optimizer.step() #improve from loss, i.e backprop
            self.scheduler.step()
            if(self.verbose):
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            self.validate(validloader)
    def validate(self, validloader):
        '''
           @breif validate the provided nerual net
           @return None
           @param validloader: itterable valiation set dataloader
        '''

        correct = 0
        total = 0
        with torch.no_grad():
            for data in validloader:
                inputs, labels = data
                outputs = self.net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.reshape(labels.size(0),)).sum().item()
            accuracy = float(correct) / float(total);
            if(self.verbose):
                 print("  Val Accuracy: %1.5f"% (accuracy))
            self.best_validation_accuracy = accuracy if (accuracy > self.best_validation_accuracy) else self.best_validation_accuracy

    def test(self, testloader):
        '''
           @breif Test the provided nerual net
           @return None
           @param testloader: itterable testing set dataloader
        '''
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = self.net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.reshape(labels.size(0),)).sum().item()
            if(self.verbose):
                print("Test Accuracy: %1.5f"% (float(correct) / float(total)))
             