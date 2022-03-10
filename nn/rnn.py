'''
  File: rnn.py
 
  Author: Thomas Kost, Mark Schelbe, Zichao Xian, Trishala Chari
  
  Date: 02 March 2022
  
  @brief This file will define our RNN object
'''
import torch, torchaudio, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 

class RNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(RNN, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_len =seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.bn_lstm = nn.BatchNorm1d(hidden_size*seq_length)
        self.dropout1 = nn.Dropout(0.2)
        self.fc_1 =  nn.Linear(hidden_size*seq_length, hidden_size) #fully connected 1
        self.bn_fc_1 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = torch.flatten(output,start_dim=1)#output.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.dropout1(out)
        out = self.bn_lstm(out)
        # out = self.relu(out)
        out = self.fc_1(out) #first Dense
        # out = self.bn_fc_1(out)
        # out = self.dropout2(out)
        # out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
