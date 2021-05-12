import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn
import torch.optim as optim
from random import randint
import torchvision.models as models

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

   
model = models.resnet50(pretrained=False)
model.conv1=nn.Conv2d(1,64,kernel_size=7,stride=(2, 2),padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, 10)

#Senting model to GPU.
model = model.to(device)

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
def findAccuracyOnTestSet():

    
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
    print( 'error rate on test set =', total_error*100)

#Initilaizing the Loss. Here I chose CrossEntropy loss as its a classification problem.    
criterion = nn.CrossEntropyLoss()

#Initilaizing the optimizer. Here I selected the Adam optimizer which has adaptive learning rate property.
#Adam also uses momentum for speeding up the learning process.
# lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False are the default parameters
optimizer=optim.Adam(model.parameters(), lr=0.001)

#Total of 200 epochs through the training data.
for epoch in range(1,201):          
   
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
    print('loss=', total_loss)
    print('total error=',total_error*100)
    
    #Find the average error 
    findAccuracyOnTestSet()

