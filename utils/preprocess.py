'''
  File: preprocess.py
 
  Author: Thomas Kost
  
  Date: 11 March 2022
  
  @brief file for preprocessing data

'''
import numpy as np
import torch, torchaudio, torchvision
from torch.autograd import Variable 

def std_preprocess_EEG(X_test,y_test,person_train_valid,X_train_valid,y_train_valid,person_test, val_size = 5000):
    y_train_valid -= 769
    y_test -= 769
    X_train_valid_prep,y_train_valid_prep = data_prep(X_train_valid,y_train_valid,2,2,True, 0, 500)
    x_test, y_test = data_prep(X_test,y_test,2,2,True, 0, 500)
    person_train_valid = np.tile(person_train_valid,4)
    person_test = np.tile(person_test,4)
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(8460, val_size, replace=False)
    ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))
    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid] 
    (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
    x_train = Variable(torch.Tensor(x_train))
    x_valid = Variable(torch.Tensor(x_valid))
    x_test = Variable(torch.Tensor(x_test))
    
    y_train = Variable(torch.Tensor(y_train))
    y_train = torch.reshape(y_train,  (y_train.shape[0], 1)) 
    
    y_valid = Variable(torch.Tensor(y_valid))
    y_valid = torch.reshape(y_valid,  (y_valid.shape[0], 1)) 
    
    y_test = Variable(torch.Tensor(y_test))
    y_test = torch.reshape(y_test, (y_test.shape[0], 1))

    return (X_test,y_test,person_train_valid,X_train_valid,y_train_valid,person_test)
    
def data_prep(X,y,sub_sample,average,noise, trim_begin, trim_end):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,trim_begin:trim_end]
    print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    print('Shape of X after subsampling and concatenating:',total_X.shape)
    return total_X,total_y
