# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 16:10:25 2018

@author: h.oberoi
"""

import torch 
import torch.nn as pytorch
import torch.nn.functional as F
import numpy as np



batch_size = 200
learning_rate = 0.001
segment_size = 128


class Model(pytorch.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer_1 = pytorch.Conv2d(in_channels = 3 ,out_channels = 196 , kernel_size = (16,6) , stride = 1 ,padding = (7,2))
        torch.nn.init.normal(self.conv_layer_1.weight,std = 0.01)
        self.max_pool_1 = pytorch.MaxPool2d((1,4),stride = 1,padding = (0,1))
        self.ReLU = pytorch.ReLU()
        
        self.fc1 = pytorch.Linear( 196*127*4 + 40 ,1024)
        torch.nn.init.normal(self.fc1.weight,std = 0.01)
        self.fc2 = pytorch.Linear(1024,6)
        torch.nn.init.normal(self.fc2.weight,std = 0.01)
        
        
    def forward(self,X,features_batch):
        X = self.conv_layer_1( X )
       # print("conv "+str(X.shape))
        X = self.max_pool_1( X )
        #print('max_pool '+str(X.shape))
        X = X.view(-1,196*127*4)
        
        X = self.ReLU( X )
        X = torch.cat((X,features_batch),dim=1)
        X = self.ReLU(F.dropout( self.fc1( X ) , p=0.05 ))
        X = self.fc2( X )
        return X
        
    
def centralize(X):
    X = X.T - np.mean(X.T,axis = 0)
    return X.T

def standardize(X):
    X = (X.T - np.mean(X.T,axis = 0))/np.std(X.T,axis=0)
    return X.T
        
def data_loading():
        #Training Data Loading
        
        training_data = open("D:/uci_data/all_data.csv")
        training_data = np.loadtxt(fname = training_data,delimiter =  ',')
        
        acc_X_1 = np.array(training_data[:,0:128]).astype(np.double)
        acc_X_2 = centralize(acc_X_1)
        acc_X_3 = standardize(acc_X_1)
        
        acc_Y_1 = np.array(training_data[:,128:2*128]).astype(np.double)
        acc_Y_2 = centralize(acc_Y_1)
        acc_Y_3 = standardize(acc_Y_1)
        
        acc_Z_1 = np.array(training_data[:,2*128:3*128]).astype(np.double)
        acc_Z_2 = centralize(acc_Z_1)
        acc_Z_3 = standardize(acc_Z_1)
        
        gyro_X_1 = np.array(training_data[:,3*128:4*128]).astype(np.double)
        gyro_X_2 = centralize(gyro_X_1)
        gyro_X_3 = standardize(gyro_X_1)
        
        gyro_Y_1 = np.array(training_data[:,4*128:5*128]).astype(np.double)
        gyro_Y_2 = centralize(gyro_Y_1)
        gyro_Y_3 = standardize(gyro_Y_1)
        
        gyro_Z_1 = np.array(training_data[:,5*128:6*128]).astype(np.double)
        gyro_Z_2 = centralize(gyro_Z_1)
        gyro_Z_3 = standardize(gyro_Z_1)
        
        X_train_1 = []
        X_train_1.append(acc_X_1)
        X_train_1.append(acc_Y_1)
        X_train_1.append(acc_Z_1)
        X_train_1.append(gyro_X_1)
        X_train_1.append(gyro_Y_1)
        X_train_1.append(gyro_Z_1)
        
        X_train_2 = []
        X_train_2.append(acc_X_2)
        X_train_2.append(acc_Y_2)
        X_train_2.append(acc_Z_2)
        X_train_2.append(gyro_X_2)
        X_train_2.append(gyro_Y_2)
        X_train_2.append(gyro_Z_2)
        
        X_train_3 = []
        X_train_3.append(acc_X_3)
        X_train_3.append(acc_Y_3)
        X_train_3.append(acc_Z_3)
        X_train_3.append(gyro_X_3)
        X_train_3.append(gyro_Y_3)
        X_train_3.append(gyro_Z_3)
        
        X_train = []
        X_train.append(X_train_1)
        X_train.append(X_train_2)
        X_train.append(X_train_3)
        
        
        X_train = np.array(X_train)
        
        X_train = np.transpose(X_train, (2, 0, 3, 1))
        print(X_train.shape)
        
        
        training_result = open("D:/uci_data/answers.csv")
        training_result = np.loadtxt(fname = training_result,delimiter =  ',')
        Y_train = []
        [Y_train.append(list(line).index(1)) for line in training_result]
        Y_train = np.array(Y_train).astype(np.long)
        print(Y_train.shape)
        
        features_train = open("D:/uci_data/features.csv")
        features_train = np.loadtxt(fname = features_train,delimiter =  ',')
        features_train = features_train - np.mean(features_train,axis = 0)
        features_train = features_train/np.std(features_train,axis=0)
        print(features_train.shape)
        
        
        
        
        #####TESTING#######
        
        training_data = open("D:/uci_data/all_data_test.csv")
        training_data = np.loadtxt(fname = training_data,delimiter =  ',')
        
        acc_X_1 = np.array(training_data[:,0:128]).astype(np.double)
        acc_X_2 = centralize(acc_X_1)
        acc_X_3 = standardize(acc_X_1)
        
        acc_Y_1 = np.array(training_data[:,128:2*128]).astype(np.double)
        acc_Y_2 = centralize(acc_Y_1)
        acc_Y_3 = standardize(acc_Y_1)
        
        acc_Z_1 = np.array(training_data[:,2*128:3*128]).astype(np.double)
        acc_Z_2 = centralize(acc_Z_1)
        acc_Z_3 = standardize(acc_Z_1)
        
        gyro_X_1 = np.array(training_data[:,3*128:4*128]).astype(np.double)
        gyro_X_2 = centralize(gyro_X_1)
        gyro_X_3 = standardize(gyro_X_1)
        
        gyro_Y_1 = np.array(training_data[:,4*128:5*128]).astype(np.double)
        gyro_Y_2 = centralize(gyro_Y_1)
        gyro_Y_3 = standardize(gyro_Y_1)
        
        gyro_Z_1 = np.array(training_data[:,5*128:6*128]).astype(np.double)
        gyro_Z_2 = centralize(gyro_Z_1)
        gyro_Z_3 = standardize(gyro_Z_1)
        
        X_test_1 = []
        X_test_1.append(acc_X_1)
        X_test_1.append(acc_Y_1)
        X_test_1.append(acc_Z_1)
        X_test_1.append(gyro_X_1)
        X_test_1.append(gyro_Y_1)
        X_test_1.append(gyro_Z_1)
        
        X_test_2 = []
        X_test_2.append(acc_X_2)
        X_test_2.append(acc_Y_2)
        X_test_2.append(acc_Z_2)
        X_test_2.append(gyro_X_2)
        X_test_2.append(gyro_Y_2)
        X_test_2.append(gyro_Z_2)
        
        X_test_3 = []
        X_test_3.append(acc_X_3)
        X_test_3.append(acc_Y_3)
        X_test_3.append(acc_Z_3)
        X_test_3.append(gyro_X_3)
        X_test_3.append(gyro_Y_3)
        X_test_3.append(gyro_Z_3)
        
        X_test = []
        X_test.append(X_test_1)
        X_test.append(X_test_2)
        X_test.append(X_test_3)
        
        X_test = np.array(X_test)
        X_test = np.transpose(X_test, (2, 0, 3, 1))
        print(X_test.shape)
        
        
        training_result = open("D:/uci_data/answers_test.csv")
        training_result = np.loadtxt(fname = training_result,delimiter =  ',')
        Y_test = []
        [Y_test.append(list(line).index(1)) for line in training_result]
        Y_test = np.array(Y_test).astype(np.long)
        print(Y_test.shape)
        
        features_test = open("D:/uci_data/test_features.csv")
        features_test = np.loadtxt(fname = features_test,delimiter =  ',')
        features_test = features_test - np.mean(features_test,axis = 0)
        features_test = features_test/np.std(features_test,axis = 0)
        
        
        return (X_train,Y_train,X_test,Y_test,features_train,features_test)
        
        
        
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Data Loading, Loss_function and optimizer
print("Data reading started")
X_train,Y_train,X_test,Y_test,Features_train,Features_test = data_loading()
print("Data reading completed")

#model = torch.load("model1_test.ckpt")
model = Model()
model.load_state_dict(torch.load("model20_test.ckpt"))
with torch.no_grad():
    correct = 0
    total = 0
    j=0
    i=0
    k=0
    while i < X_train.shape[0]:
        if i+batch_size < X_train.shape[0]:
            minibatch_X = X_train[i:i+batch_size]
            i+=batch_size
        else:
            minibatch_X = X_train[i:]
            i = X_train.shape[0]+1
        
        if j+batch_size < Y_train.shape[0]:
            minibatch_Y = Y_train[j:j+batch_size]
            j+=batch_size
        else:
            minibatch_Y = Y_train[j:]
            j = Y_train.shape[0]+1
        
        if k+batch_size < Features_train.shape[0]:
            minibatch_features_train = Features_train[k:k+batch_size]
            k+=batch_size
        else:
            minibatch_features_train = Features_train[k:]
            k = Features_train.shape[0]+1
    
        
        
        minibatch_X = torch.from_numpy(minibatch_X).float()
        minibatch_Y = torch.from_numpy(minibatch_Y).long()
        minibatch_features_train = torch.tensor(minibatch_features_train,requires_grad = False).float()

        
        forward_pass = model(minibatch_X,minibatch_features_train)
        _, predicted = torch.max(forward_pass.data, 1)
        total+= minibatch_Y.shape[0]
        correct+=(predicted==minibatch_Y).sum().item()
acc = (correct/total)*100
print('Training Accuracy {}'.format(acc))
with torch.no_grad():
    correct = 0
    total = 0
    j=0
    i=0
    k=0
    while i < X_test.shape[0]:
        if i+batch_size < X_test.shape[0]:
            minibatch_X = X_test[i:i+batch_size]
            i+=batch_size
        else:
            minibatch_X = X_test[i:]
            i = X_test.shape[0]+1
        
        if j+batch_size < Y_test.shape[0]:
            minibatch_Y = Y_test[j:j+batch_size]
            j+=batch_size
        else:
            minibatch_Y = Y_test[j:]
            j = Y_test.shape[0]+1
        
        if k+batch_size < Features_test.shape[0]:
            minibatch_features_test = Features_test[k:k+batch_size]
            k+=batch_size
        else:
            minibatch_features_test = Features_test[k:]
            k = Features_test.shape[0]+1
        
        minibatch_X = torch.from_numpy(minibatch_X).float()
        minibatch_Y = torch.from_numpy(minibatch_Y).long()
        minibatch_features_test = torch.tensor(minibatch_features_test,requires_grad = False).float()

        
        forward_pass = model(minibatch_X,minibatch_features_test)
        _, predicted = torch.max(forward_pass.data, 1)
        total+= minibatch_Y.shape[0]
        correct+=(predicted==minibatch_Y).sum().item()
acc = (correct/total)*100
print('Testing Accuracy {}'.format(acc))   
print("")
print("")
    


 