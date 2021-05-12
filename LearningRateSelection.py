import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn
import torch.optim as optim
from random import randint

#Downloading MNIST data provided by PyTorch
#print('Starting data download')
#mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
#mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
#print('Completed data download')

print('Starting data download')
mnist_trainset=torch.load('./data/MNIST/processed/training.pt')
mnist_testset=torch.load('./data/MNIST/processed/test.pt')
print('Completed data download')

#This is to check whether cuda is available.
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device=torch.device('cuda')
    print("The device is" +str(device))
torch.cuda.empty_cache()

#Defining the model. Here I am using the model structure of LeNet-5 model introduced in paper 'Gradient-based learning Applied to document recognition'
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        #Convolutional layer which take in 1 channel input. Outputs 6 activation maps by convolution of 5x5 filters.
        #A 2 layer padding is added to make the input from 28x28 to 32x32 as to avoid sudden reduction of input.
        #default stride is 1
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        
        #Convolutional layer with ihnput of 6 activation map of previous convolutional layer. Out put is 16 acvtivation maps 
        #by convolution of 5x5 filters.
        #output dimension is ((28-5)/1)+1 = 24 i.e 24x24
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        #Fully connected layer which takes the flattened values as input
        self.fc1   = nn.Linear(16*5*5, 120)
        #Fully Connected layer
        self.fc2   = nn.Linear(120, 84)
        #Fully Connected layer with outputs 10 values corresponding to the 10 digits.
        self.fc3   = nn.Linear(84, 10)
        
        #Initializing ReLu
        self.ReLu = nn.ReLU()
        
        #Initializing maxpool
        self.MaxPool2d = nn.MaxPool2d(2)
        
    def forward(self, x):
        
        #1st layer convolution. Output dimension is ((32-5)/1)+1 = 28 i.e 28x28
        x=self.conv1(x)
        
        #Relu Activation Function
        x=self.ReLu(x)
        
        #Max pooling the output. Filter size of 2 and stride =2.
        #Output dimension is ((28-2)/2)+1 = 14 i.e 14x14
        x=self.MaxPool2d(x)

        #2nd layer convolution. Output dimension is ((14-5)/1)+1 = 10 i.e 10x10
        x=self.conv2(x)
        
        #Relu Activation Function
        x=self.ReLu(x)
        
        #Max pooling the output. Filter size of 2 and stride =2.
        #Output dimension is ((10-2)/2)+1 = 5 i.e 5x5
        x=self.MaxPool2d(x)
        
        #Here the 2D output is flattened to 1D for inputting it to Fully connected layers.
        #output of previous layers is 16 5x5 activations maps. Hence the flattened version will have 16*5*5 = 400 elments
        x = x.view(-1,16*5*5)
        
        #pass the flattened version to fully connected layer.
        x=self.fc1(x)
        
        #relu activation function.
        x=self.ReLu(x)
        
        #2nd fully connected layer.
        x=self.fc2(x)
        
        #relu activation function.
        x=self.ReLu(x)
        
        #Last fully connected layer. Outputs 10 values corresponding to the 10 digits.
        x =self.fc3(x)
        
        return x



##Getting the Training data.
#Converting the byte tensor to float tensor for calculation.
#Normalizing the input by dividing the value with 255 which is tha maximum value.
train_images = (mnist_trainset[0].float())/255
#Getting the training data labels
train_labels = mnist_trainset[1]

##Getting the Testing data.
#Converting the byte tensor to float tensor for calculation.
#Normalizing the input by dividing the value with 255 which is tha maximum value.
test_images = (mnist_testset[0].float())/255
#Getting the testing data labels
test_labels = mnist_testset[1]

#Initializing the batch size to 100. So each mini batch will have 100 images for training and testing.
bs=100


#Function to calculate the Error 
def getError( scores , labels ):

    #Get the indices of the maximum value in each row. That is values from 0 to 9.
    predicted_labels = scores.argmax(dim=1)
    
    #Check whether the maximum value in scores is equal to the label given. So if the label value is 5 and 5th row index
    #of scores has the largrest value then the prediction is correct else not.
    #Value will be 1 if both are same else 0
    compareResult = (predicted_labels == labels)
    
    #Sum all the values to find number of matches.
    totalMatchedValues=compareResult.sum()
    
    #Calculateb the error.
    return 1-(totalMatchedValues.float()/bs)


#Function to find the accuracy of the current model in test data.
def findAccuracyOnTestSet(learningRate):

    
    running_error=0
    num_batches=0

    for i in range(0,10000,bs):

        #Get the minibatch 
        minibatch_data =  test_images[i:i+bs].unsqueeze(dim=1)
        minibatch_label= test_labels[i:i+bs]
    
        #Sent it to GPU
        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)  
        
        #Do the prediction with trained model
        scores=model( minibatch_data ) 

        #Get the error test set
        error = getError( scores , minibatch_label)

        #Add the error of current minibatch to total batch
        running_error += error.item()

        #append the batch count
        num_batches+=1


    total_error = running_error/num_batches
    print('learning rate =',learningRate,' error rate on test set =', total_error*100)

#Initilaizing the Loss. Here I chose CrossEntropy loss as its a classification problem.    
criterion = nn.CrossEntropyLoss()
learningRates = [10, 1 , 0.1, 0.01, 0.001, 0.0001,0.00001]

for learningRate in learningRates:

    #Assigning the model    
    model = LeNet5()

    #Senting model to GPU.
    model = model.to(device)
    
    #Initilaizing the optimizer. Here I selected the SGD optimizer to show the effect of learning rate.        
    optimizer=optim.Adam(model.parameters(),lr=learningRate)

    #Total of 200 epochs through the training data.
    for epoch in range(1,151): 

        #To calculate the loass and error values
        running_loss=0
        running_error=0
        num_batches=0
        
        #For shuffling the train data
        shuffled_indices=torch.randperm(60000)    
        for imageIndex in range(0,60000,bs):
            
            #making the previous calculated gradient to zero for the new mini batch.        
            optimizer.zero_grad()
            
            #Calculating the indices of the current mini batch.
            indices=shuffled_indices[imageIndex:imageIndex+bs]
            #Unsqueezing in Dimension 1 to add the Channel length - [bs,28,28] to [bs,1,28,28] where 1 is the number of channels.
            minibatch_data =  train_images[indices].unsqueeze(dim=1)
            minibatch_label=  train_labels[indices]
            
            #Senting the minibatch to GPU
            minibatch_data=minibatch_data.to(device)
            minibatch_label=minibatch_label.to(device)
            
            #Indicating that the gradient of the minibatch need to be calculated.
            minibatch_data.requires_grad_()

            #Predicting using the model.
            scores=model( minibatch_data ) 

            #Calculating the loss between predicted and label values.
            loss =  criterion( scores , minibatch_label) 

            #Calculating the gradient associated with each parameter in the model with the loss
            loss.backward()
            
            #Updating the model parameters with the gradient found.
            optimizer.step()
            
            #Adding the loss of the current minibatch to total loss of epoch
            running_loss += loss.detach().item()
            
            #getting the error of current minibatch and adding it to total error.
            error = getError( scores.detach() , minibatch_label)
            running_error += error.item()
            
            #Appending number of batches
            num_batches+=1        
        
            #deleting the tesnors from GPU to save memmory.
            del minibatch_data
            del minibatch_label
            
        #getting the stats of the epoch by averaging it.
        total_loss = running_loss/num_batches
        total_error = running_error/num_batches    
        
        #Printing the average loss and error.
        print('epoch=',epoch)
        print('learning rate =',learningRate,' loss=', total_loss)
        print('learning rate =',learningRate,' total error=',total_error*100)
        
        #Find the average error 
        findAccuracyOnTestSet(learningRate)

    del model
