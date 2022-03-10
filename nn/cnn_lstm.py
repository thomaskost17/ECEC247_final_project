'''
  File: cnn_lstm.py
 
  Author: Thomas Kost, Mark Schelbe, Zichao Xian, Trishala Chari
  
  Date: 09 March 2022
  
  @brief This file will define a CNN object
'''
import torch, torchaudio, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(CNN_LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_len =seq_length

        self.L1 = nn.Sequential(
            nn.Conv1d(in_channels=self.seq_len, out_channels = 25, kernel_size = (9,1), padding='same' ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = (3,1), padding=(1,0)),
            nn.BatchNorm2d(25),
            nn.Dropout(0.5)
        )
        self.L2 = nn.Sequential(
            nn.Conv1d(in_channels=25, out_channels = 50, kernel_size = (9,1), padding='same' ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = (3,1), padding=(1,0)),
            nn.BatchNorm2d(50),
            nn.Dropout(0.5)
        )
        self.L3 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels = 100, kernel_size = (9,1), padding='same' ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = (3,1), padding=(1,0)),
            nn.BatchNorm2d(100),
            nn.Dropout(0.5)
        )
        self.L4 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels = 200, kernel_size = (9,1), padding='same' ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = (3,1), padding=(1,0)),
            nn.BatchNorm2d(200),
            nn.Dropout(0.5),

            nn.Flatten(2)# Nx800  
            #nn.Linear(800,self.num_classes)
        )
        self.LSTM = nn.Sequential(
            nn.LSTM(input_size=4, hidden_size=self.hidden_size,
                    num_layers=num_layers, batch_first=True)
        )
        self.L5 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(1),
            nn.Linear(200*self.hidden_size, 1024),
            nn.ReLU(),
            #nn.Batchnorm(1024),
            nn.Dropout(0.2)
        )
        self.L6 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            #nn.Batchnorm(256),
            nn.Dropout(0.2)
        )
        self.L7 = nn.Sequential(
            nn.Linear(256, self.num_classes),
        )
    
    def forward(self,x):

        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        out,_ = self.LSTM(out)
        out = self.L5(out)
        out = self.L6(out)
        out = self.L7(out)
        return out
