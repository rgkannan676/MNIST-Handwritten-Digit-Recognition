import torch
import numpy as np

from random import randint

#Downloading MNIST data provided by PyTorch
#print('Starting data download')
#mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
#mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
#print('Completed data download')

print('Starting data loading')
mnist_trainset=torch.load('./data/MNIST/processed/training.pt')
#mnist_testset=torch.load('./data/MNIST/processed/test.pt')
print('Completed data loading')

#This is to check whether cuda is available.
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device=torch.device('cuda')
    print("The device is" +str(device))
torch.cuda.empty_cache()

#Normalizing the image data value between 0 and 1
train_images = (mnist_trainset[0].float())/255
train_labels = mnist_trainset[1]

#Function to get the number of wrong predictions in a minibatch
def getError( scores , labels ):
    
    #Get the indices of the maximum value in each row. That is values from 0 to 9.
    predicted_labels = scores.argmax(axis=1)

    #Check whether the maximum value in scores is equal to the label given. So if the label value is 5 and 5th row index
    #of scores has the largrest value then the prediction is correct else not.
    #Value will be 1 if both are same else 0
    compareResult = (predicted_labels == labels)
    compareResult.astype(float)
    #Sum all the values to find number of matches.
    totalMatchedValues=compareResult.sum()
    #Calculateb the error percentage.
    return (1-(totalMatchedValues/bs))*100

#Function to implement ReLu. Here it makes all negative values to 0
def implement_ReLU(Y):
    return np.array(Y).clip(min=0)

#Hot encode the labels   
def get_hot_encodedLabels(labels):
    T=np.zeros((bs,10))
    for i in range(0,bs):
        T[i,labels[i]]=1
    return T                

#Calculate the total Loss of a batch
def calculate_loss_bs(sftmax,T):
    temp=np.multiply(sftmax,T).sum(axis=1)
    individualLoss= -np.log(temp)
    totalLoss= individualLoss.sum()
    return totalLoss
    

bs = 100

##Find Initialization range of 1st layer i.e [-y,y) where y = 1/sqrt(n) , where n is the number of input connections to layer
firstLayerIntLimit = 1/np.sqrt(784)

#Initilaizing First layer weight and bias matrix
Uw = np.random.rand(784,1000) * (2 * firstLayerIntLimit) + (- firstLayerIntLimit)
Ub = np.random.rand(1,1000) * (2 * firstLayerIntLimit) + (- firstLayerIntLimit)

##Find Initialization range of 1st layer i.e [-y,y) where y = 1/sqrt(n) , where n is the number of input connections to layer
secondLayerIntLimit = 1/np.sqrt(1000)

#Initilaizing Second layer weight and bias matrix
Vw = np.random.rand(1000,10) * (2 * secondLayerIntLimit) + (- secondLayerIntLimit)
Vb = np.random.rand(1,10) * (2 * secondLayerIntLimit) + (- secondLayerIntLimit)



for epoch in range(1,10):          
   
    #To calculate the loass and error values
    running_loss=0
    running_error=0
    num_batches=0
    
    #For shuffling the train data
    shuffled_indices=torch.randperm(60000)
 
    for batchNumber in range(0,60000,bs):
    
        #input Matrix
        X = np.zeros((bs,784))
        
        #First layer output matrix
        Y = np.zeros((bs,1000))

        #relU output Matrix
        Yr = np.zeros((bs,1000))
        
        #Predicted matrix
        Z = np.zeros((bs,10))
    
        ## Started the Forward Propagation.
        
        #Calculating the indices of the current mini batch.
        indices=shuffled_indices[batchNumber:batchNumber+bs]
        
        #Getting image data and flattening it to pass it to MLP.
        minibatch_data =  train_images[indices]
        
        #initializing input by flattening the images.
        X = np.array(minibatch_data.view(bs,784))
        
        #getting image label
        minibatch_label=  np.array(train_labels[indices])
        
        #Output of the first Layer
        Y = np.matmul(X,Uw) + Ub
        
        #Applying the ReLu in 1st Layer output
        Yr=implement_ReLU(Y)
        
        #Output of the second Layer
        Z = np.matmul(Yr,Vw) + Vb
        
        #get the percentage of error in training data
        averageBatchError = getError(Z,minibatch_label)
        print('Average Error of batch ',num_batches,' is ',averageBatchError )
       
        running_error+=averageBatchError
        
        #Calculate the sum of exponents of the prediction of data instance.
        softmaxDemominator = np.resize(np.exp(Z).sum(axis=1) ,(bs,1))
        
        #Calaculate 
        softmax = np.exp(Z)/softmaxDemominator
        
        #get the hot encoded labels
        T = get_hot_encodedLabels(minibatch_label)
        
        #Getting the total loss of the batch.
        averageBatchLoss= calculate_loss_bs(softmax,T)/bs
        
        print('Average Loss of batch ',num_batches,' is ',averageBatchLoss )
        
        #adding to runnig error
        running_loss+=averageBatchLoss
        
        
        ## Completed the Forward Propagation.
        
        ## Started calculating derivatives.
        
        #Calculating dL/dZ.        
        dL_dZ = softmax - T
        
        #Calculating dL/dVw = dL/dZ * YR
        dL_dVw = np.matmul(Yr.transpose(),dL_dZ)
        #Calculating dL/dVb = dL/dZ *1
        dL_dVb = dL_dZ
        #Calculating dL/dYr = dL/dZ * Vw
        dL_dYr = np.matmul( dL_dZ, Vw.transpose())
        
        #Calculating dL_dY = dL/dYR * 1 if Y >0  
        #and = 0 if Y<=0
        
        #Yr has all the positive components in Y and 0 fro negetive
        Yrelu = Yr
        #making all positive values to 1 and others are 0
        Yrelu[Yrelu > 0]=1
                
        #Elementwise multiplication
        dL_dY = np.multiply(Yrelu,dL_dYr)
        
        #Calaculate dL_dUw = dL_dY * X 
        dL_dUw = np.matmul(X.transpose(),dL_dY)
        
        #Calculate dL_dUb = dL/dY * 1
        dL_dUb = dL_dY
        
        ## Completed calculating derivatives.
        
        ## Started updating weights and bias.
        
        #initializing learning rate
        lr=0.001
        
        #updating Vw weight
        Vw= Vw - lr * dL_dVw
        
        #updating Vb weight
        Vb= Vb - lr * dL_dVb
        
        #updating Uw weight
        Uw= Uw - lr * dL_dUw
        
        #updating Ub weight
        Ub= Ub - lr * dL_dUb
        
        num_batches = num_batches +1
        
    print('>>Loss of epoch ',epoch,' is ',running_loss/num_batches)
    print('--Error of epoch ',epoch,' is ',running_error/num_batches)