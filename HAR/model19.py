



import torch 
import torch.nn as pytorch
import torch.nn.functional as F
import numpy as np


number_of_epochs = 1000     # no. of epochs to train the model for
batch_size = 200            # number of input samples trained at a time.
learning_rate = 0.0005
segment_size = 128      # 128 data entries correspongding to 2.56 sec data at 50hz freq



results = open("model19.txt","w")       # Loss, training and testing accuracy of every epoch is saved in this text file.

class Model(pytorch.Module):        # extending Module class
    def __init__(self):
        super(Model, self).__init__()       # calling the upper class
        self.conv_layer_1 = pytorch.Conv2d(in_channels = 1 ,out_channels = 196 , kernel_size = (16,6) , stride = 1 ,padding = (7,2))
        self.bn1 = nn.BatchNorm2D(in_channels = 1)
        torch.nn.init.normal(self.conv_layer_1.weight,std = 0.01)       #initializing weights with normal distribution
        self.max_pool_1 = pytorch.MaxPool2d((1,4),stride = 1,padding = (0,1))
        self.ReLU = pytorch.ReLU()
        
        
        self.fc1 = pytorch.Linear( 196*127*4 + 40 ,1024)        # first linear layer after concatenatng 40 features
        self.bn2 = nn.BatchNorm1D()
        torch.nn.init.normal(self.fc1.weight,std = 0.01)
        self.fc2 = pytorch.Linear(1024,6)
        torch.nn.init.normal(self.fc2.weight,std = 0.01)
        
    def forward(self,X,features_batch,boolean):
        X = self.conv_layer_1( X )
       
        X = self.max_pool_1( X )
        
        X = X.view(-1,196*127*4)         # stretching into a linear layer
        
        X = self.ReLU( X )
        X = torch.cat((X,features_batch),dim=1)         # concatenating the statistical features
        X = self.ReLU(F.dropout( self.fc1( X ) , p=0.05,training = boolean ))
        X = self.fc2( X )
        return X
        
        
        
def centralize(X):
    X = X.T - np.mean(X.T,axis = 0)         # applying centralization by subtracting mean from the data
    return X.T

def data_loading():
        #Training Data Loading
        
        '''
        'all data' file contains the training data separated by comma ','
        One training row consistes of 128*6 values corresponging to acceleration X,Y,Z and gyroscope X,Y,Z
        '''
        
        training_data = open("D:/uci_data/all_data.csv")        
        training_data = np.loadtxt(fname = training_data,delimiter =  ',')
        
        acc_X = np.array(training_data[:,0:128]).astype(np.double)          # extracting data from 0 to 128 
        acc_X = centralize(acc_X)
        acc_Y = np.array(training_data[:,128:2*128]).astype(np.double)      # extracting data from 128 to 2*128 
        acc_Y = centralize(acc_Y)
        acc_Z = np.array(training_data[:,2*128:3*128]).astype(np.double)    # extracting data from 2*128 to 3*128 
        acc_Z = centralize(acc_Z)
        gyro_X = np.array(training_data[:,3*128:4*128]).astype(np.double)   # extracting data from 3*128 to 4*128 
        gyro_X = centralize(gyro_X)
        gyro_Y = np.array(training_data[:,4*128:5*128]).astype(np.double)   # extracting data from 4*128 to 5*128 
        gyro_Y = centralize(gyro_Y)
        gyro_Z = np.array(training_data[:,5*128:6*128]).astype(np.double)   # extracting data from 5*128 to 6*128 
        gyro_Z = centralize(gyro_Z)

        
        '''
        Stacking all the numpy arrays into a list
        '''
        X_train = []        
        X_train.append(acc_X)
        X_train.append(acc_Y)
        X_train.append(acc_Z)
        X_train.append(gyro_X)
        X_train.append(gyro_Y)
        X_train.append(gyro_Z)
        
        X_train = np.array(X_train) # Changing the list into numpy array
        X_train = np.transpose(X_train, (1, 0, 2))      
        print(X_train.shape)
        
        '''
        'answers' file contains the answer labels for the training data in one hot embedding
        '''
        training_result = open("D:/uci_data/answers.csv")
        training_result = np.loadtxt(fname = training_result,delimiter =  ',')
        Y_train = []
        [Y_train.append(list(line).index(1)) for line in training_result]   # Extracting the index where 1 is present in one hot embedding
        Y_train = np.array(Y_train).astype(np.long)
        print(Y_train.shape)
        
        features_train = open("D:/uci_data/features.csv")       # Contains the features row wise for every training example
        features_train = np.loadtxt(fname = features_train,delimiter =  ',')
        
        #normalizing features values
        features_train = features_train - np.mean(features_train,axis = 0)
        features_train = features_train/np.std(features_train,axis=0)
        print(features_train.shape)
        
        
        
        
        #####TESTING#######
        
        training_data = open("D:/uci_data/all_data_test.csv")       # Contains the testing data separated by comma ','
        training_data = np.loadtxt(fname = training_data,delimiter =  ',')
        
        '''
        Parsing is done the way as for training data.
        '''
        acc_X = np.array(training_data[:,0:128]).astype(np.double)          # extracting data from 0 to 128 
        acc_X = centralize(acc_X)
        acc_Y = np.array(training_data[:,128:2*128]).astype(np.double)      # extracting data from 128 to 2*128
        acc_Y = centralize(acc_Y)
        acc_Z = np.array(training_data[:,2*128:3*128]).astype(np.double)    # extracting data from 2*128 to 3*128 
        acc_Z = centralize(acc_Z)
        gyro_X = np.array(training_data[:,3*128:4*128]).astype(np.double)   # extracting data from 3*128 to 4*128
        gyro_X = centralize(gyro_X)
        gyro_Y = np.array(training_data[:,4*128:5*128]).astype(np.double)   # extracting data from 4*128 to 5*128 
        gyro_Y = centralize(gyro_Y)
        gyro_Z = np.array(training_data[:,5*128:6*128]).astype(np.double)   # extracting data from 5*128 to 6*128
        gyro_Z = centralize(gyro_Z)

        
        '''
        Stacking all the numpy arrays into a list
        '''
        X_test = []
        X_test.append(acc_X)
        X_test.append(acc_Y)
        X_test.append(acc_Z)
        X_test.append(gyro_X)
        X_test.append(gyro_Y)
        X_test.append(gyro_Z)
        
        X_test = np.array(X_test)
        X_test = np.transpose(X_test, (1, 0, 2))
        print(X_test.shape)
        
        '''
        'answers_test' file contains the answer labels for the testing data in one hot embedding
        '''
        training_result = open("D:/uci_data/answers_test.csv")
        training_result = np.loadtxt(fname = training_result,delimiter =  ',')
        Y_test = []
        [Y_test.append(list(line).index(1)) for line in training_result]         # Extracting the index where 1 is present in one hot embedding
        Y_test = np.array(Y_test).astype(np.long)
        print(Y_test.shape)
        
        features_test = open("D:/uci_data/test_features.csv")       # Contains the features row wise for every testing example
        features_test = np.loadtxt(fname = features_test,delimiter =  ',')
        
        #normalizing features values
        features_test = features_test - np.mean(features_test,axis = 0)
        features_test = features_test/np.std(features_test,axis = 0)
        
        X_train = np.reshape(X_train,[-1,1,128,6])      # Reshaping the training numpy array to make it compatible for input in 2D Conv network
        X_test = np.reshape(X_test , [-1,1,128,6]) 
        print(X_train.shape)
        
        return (X_train,Y_train,X_test,Y_test,features_train,features_test)
        
        
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data Loading, Loss_function and optimizer
print("Data reading started")
X_train,Y_train,X_test,Y_test,Features_train,Features_test = data_loading()
print("Data reading completed")


model = Model().to(device)

loss_function = pytorch.CrossEntropyLoss()      # Loss Function used for the model is Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate , weight_decay = 0.00005)    # ADAM optimizer used as optimization function


#Training the model

max_accuracy_train = 0
max_accuracy_test = 0

for epoch in range(number_of_epochs):
    model.train()
    j=0
    i=0
    k=0
    total_loss = 0
    while i< X_train.shape[0]:      # mini-batch training done with 'batch_size' batches
        if i+batch_size < X_train.shape[0]: 
            minibatch_X = X_train[i:i+batch_size]       # minibatch of size 'batch_size' made for training
            i+=batch_size
        else:
            minibatch_X = X_train[i:]       # minibatch made for the end examples left
            i=X_train.shape[0]+1
        
        
        if j+batch_size < Y_train.shape[0]:
            minibatch_Y = Y_train[j:j+batch_size]       # same size batch extracted containing the answer labels
            j+=batch_size
        else:
            minibatch_Y = Y_train[j:]       # minibatch made for the end examples left
            j=Y_train.shape[0]+1
        
        if k+batch_size < Features_train.shape[0]:
            minibatch_features_train = Features_train[k:k+batch_size]       # same size features batch extracting for training purpose
            k+=batch_size
        else:
            minibatch_features_train = Features_train[k:]       # minibatch made for the end examples left
            k = Features_train.shape[0]+1
        
        
        minibatch_X = torch.from_numpy(minibatch_X).float() 
        minibatch_Y = torch.from_numpy(minibatch_Y).long()
        minibatch_features_train = torch.tensor(minibatch_features_train,requires_grad = False).float()     # Setting the requires_grad as false to avoid 
                                                                                                            # training of features in linear layer
                                                                                        
        
        
        forward_pass = model(minibatch_X,minibatch_features_train,True)     #making a forward pass on the network with minibatches
        loss = loss_function(forward_pass , minibatch_Y)                    #calculating the loss from the minibatch
        total_loss+=loss.item()                                             #adding to the total loss
        optimizer.zero_grad()                                               #setting all parameters to be zero
        loss.backward()                                                     #using backpropagation 
        optimizer.step()                                                    #taking one step of optimization using our optimizer
        
    print('Epoch {},Loss {}'.format(epoch,total_loss))
    print("")
    results.write('Epoch {},Loss {}'.format(epoch,total_loss))      # writing the results in the text file
    results.write("\n")
    model.eval()
    
    
    '''
    Following the same steps above but wthout training to find the training accuracy and saving the best training result in model19_train.ckpt
    '''
    with torch.no_grad():       # gradient/training is not required in this piece of code
        correct = 0
        total = 0
        j=0
        i=0
        k=0
        while i < X_train.shape[0]:     # mini-batches created with 'batch_size' sizes
            if i+batch_size < X_train.shape[0]:
                minibatch_X = X_train[i:i+batch_size]       # minibatch of size 'batch_size' made 
                i+=batch_size
            else:
                minibatch_X = X_train[i:]       # minibatch made for the end examples left
                i = X_train.shape[0]+1
            
            if j+batch_size < Y_train.shape[0]:
                minibatch_Y = Y_train[j:j+batch_size]       # same size batch extracted containing the answer labels
                j+=batch_size
            else:
                minibatch_Y = Y_train[j:]       # minibatch made for the end examples left
                j = Y_train.shape[0]+1
            
            if k+batch_size < Features_train.shape[0]:
                minibatch_features_train = Features_train[k:k+batch_size]       # same size features batch extracting for accuracy prediction
                k+=batch_size
            else:
                minibatch_features_train = Features_train[k:]       # minibatch made for the end examples left
                k = Features_train.shape[0]+1
        
            
            minibatch_X = torch.from_numpy(minibatch_X).float()
            minibatch_Y = torch.from_numpy(minibatch_Y).long()
            minibatch_features_train = torch.tensor(minibatch_features_train,requires_grad = False).float()  # Setting the requires_grad as false to avoid 
                                                                                                             # training of features in linear layer

            
            forward_pass = model(minibatch_X,minibatch_features_train,False)        # making a forward pass to get predicted labels
            _, predicted = torch.max(forward_pass.data, 1)
            total+= minibatch_Y.shape[0]
            correct+=(predicted==minibatch_Y).sum().item()      # increment the correct value if preduction matches with actual answer label
    acc = (correct/total)*100
    if acc > max_accuracy_train:
        torch.save(model.state_dict(), 'model19_train.ckpt')        # model saved for the best training accuracy achieved
        max_accuracy_train = acc
    print('Training Accuracy {}'.format(acc))
    results.write('Training Accuracy {}'.format(acc))       #writing the results in the text file
    results.write("\n")
    
    
    
    '''
    Following the same steps above but wthout training to find the testing accuracy and saving the best test result in model19_test.ckpt
    '''
    with torch.no_grad():        # gradient/training is not required in this piece of code
        correct = 0
        total = 0
        j=0
        i=0
        k=0
        while i < X_test.shape[0]:          # mini-batches created with 'batch_size' sizes
            if i+batch_size < X_test.shape[0]:
                minibatch_X = X_test[i:i+batch_size]        # minibatch of size 'batch_size' made
                i+=batch_size
            else:
                minibatch_X = X_test[i:]        # minibatch made for the end examples left
                i = X_test.shape[0]+1
            
            if j+batch_size < Y_test.shape[0]:
                minibatch_Y = Y_test[j:j+batch_size]        # same size batch extracted containing the answer labels
                j+=batch_size
            else:
                minibatch_Y = Y_test[j:]        # minibatch made for the end examples left
                j = Y_test.shape[0]+1
            
            if k+batch_size < Features_test.shape[0]:
                minibatch_features_test = Features_test[k:k+batch_size]         # same size features batch extracting for accuracy prediction
                k+=batch_size
            else:
                minibatch_features_test = Features_test[k:]         # minibatch made for the end examples left
                k = Features_test.shape[0]+1
            
            minibatch_X = torch.from_numpy(minibatch_X).float()
            minibatch_Y = torch.from_numpy(minibatch_Y).long()
            minibatch_features_test = torch.tensor(minibatch_features_test,requires_grad = False).float()   # Setting the requires_grad as false to avoid 
                                                                                                             # training of features in linear layer

            
            forward_pass = model(minibatch_X,minibatch_features_test,False)     # making a forward pass to get predicted labels
            _, predicted = torch.max(forward_pass.data, 1)
            total+= minibatch_Y.shape[0]
            correct+=(predicted==minibatch_Y).sum().item()          # increment the correct value if preduction matches with actual answer label
    acc = (correct/total)*100
    if acc > max_accuracy_test:
        torch.save(model.state_dict(), 'model19_test.ckpt')         # model saved for the best test accuracy achieved
        max_accuracy_test = acc
    print('Testing Accuracy {}'.format(acc))   
    print("")
    print("")
    results.write('Testing Accuracy {}'.format(acc))        #writing the results in the text file
    results.write("\n")
    results.write("\n")
    results.flush()

results.close()

     
        

   

























