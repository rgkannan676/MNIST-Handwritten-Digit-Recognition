# MNIST Handwritten Digit Recognition

In this project I am trying to classify the handwritten digits in the MNIST data set. The MNIST dataset contains 60,000 training images of handwritten digits from 0 to 9 and 10,000 images for testing. It thus has 10 different classes, where each grayscale digit image is represented by a 28×28 matrix. 
# Data Set
The MNIST data set provided by PyTorch framework is used in this project. The data set consists of 60000 training and 10000 test images. 

![image](https://user-images.githubusercontent.com/29349268/118025629-a3d25600-b392-11eb-9659-6c704e493c60.png)

Note. From “An analysis of image storage systems for scalable training of deep neural networks” by Lim, S.H., Bottou, L., Young, S.R., & Patton, R.M., 2016. (https://www.researchgate.net/publication/306056875_An_analysis_of_image_storage_systems_for_scalable_training_of_deep_neural_networks).

# Network Design
In this project I am using a convolutional network LeNet-5 introduced in the research paper ‘Gradient-based learning Applied to document recognition’. The Model consists of 2 convolutional layers with a Max Pool layer and ReLu activation. The output of this is flattened and passed through 3 fully connected layers.

![image](https://user-images.githubusercontent.com/29349268/118025708-b8aee980-b392-11eb-95c6-a5c21c15cc63.png)

Note. From “Gradient-Based learning applied to Document recognition” by LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P., 2018, Proceedings of the IEEE, 86(11) (http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). Copyright 2020 by IEEE).
Below is the network architecture with parameters which was built using the PyTorch Framework given in the order of input to output. The parameters passed to each layer can be observed from this.

**LeNet5 composition**
```
	First Convolution layer 
	Conv2d(in_channels =1, out_channels =6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
	ReLu: ReLU()
	First Max Pool Layer
	MaxPool2d (kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	Second Convolution Layer
	Conv2d (in_channels =6, out_channels =16, kernel_size=(5, 5), stride=(1, 1))
	ReLu : ReLU()
	Second Max Pool Layer
	MaxPool2d (kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	First Fully Connected Layer
	Linear (in_features=400, out_features=120, bias=True)
	ReLu : ReLU()
	Second Fully Connected Layer
	Linear (in_features=120, out_features=84, bias=True)
	ReLu : ReLU()
	Third Fully Connected Layer
	Linear (in_features=84, out_features=10, bias=True
  ```
  
**Detailed analysis of layers in LeNet-5**

**First Convolutional layer**

The input of the first convolutional layer is the 1 channel 28x 28 gray scale handwritten digit image. In this network, a two-layer padding is done to prevent the quick dimension shrinking of the activation map to next layers. 
Six filters of dimension 5x5 is used for the convolution in this layer. The default stride of 1 is used here.  The output activation map dimension from the equation: 
 ((Input width + 2 * padding – Filter Size) /Stride) +1 = ((28 + 2* 2 – 5)/1) +1 = 28 
Therefore, the dimension of the 6 activation maps will be 28x28. So here you can observe that due to padding, the dimension of input and output is same.

**First Max Pool layer**

The output activation maps of the first convolutional layer are passed to a max pooling layer of filter size 2 and Stride of 2. Max pool layer is used to reduce the dimension of the input to its following layers. Max pooling extracts the sharpest features of an image. Pooling also helps to bring translation invariance in the images. The output activation map dimension can be obtained from the below equation: 
((Input width – Filter Size) /Stride) +1 = ((28 – 2)/2) +1 = 14
Therefore, the dimension of the output of max pool layer is 14x14.

![image](https://user-images.githubusercontent.com/29349268/118026032-f875d100-b392-11eb-86e8-c32af1e2f22f.png)

Note. In this figure, each colour corresponds to 2x2 filter position after each stride with stride=2. The output contains the maximum value from each filter position. From “Applied Deep Learning - Part 4: Convolutional Neural Networks” by Medium, 2017 (https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2). 

**ReLu activation Layer**

The ReLu (Rectified Linear Unit) activation layer is used for bringing nonlinearity in the network function. The equation of ReLu function is ReLU(x) = (x)+ = max(0,x). If the input to ReLu is less than zero, then output will be zero. For positive values output is same as input.

![image](https://user-images.githubusercontent.com/29349268/118026124-104d5500-b393-11eb-81cb-fd481bdae1f4.png)

Note. From “RELU” by PyTorch, 2018 (https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html). Copyright 2019 by Torch Contributors
The reason why ReLu was used as activation function instead of Sigmoid or Tanh is because ReLu does not have the vanishing gradient issue. For Sigmoid or Tanh when the input is hight, corresponding gradient value can be low hence slowing down the learning. But for ReLu the gradient will be always 1 for the positive inputs hence will not have vanishing gradient issue.

**Second Convolutional Layer**

The input of the second convolutional layer is the 6 channels with dimension of 14x 14 
16 filters of dimension 5x5 is used for the convolution in this layer. The default stride of 1 is used here.  The output activation map dimension from the equation: 
((Input width – Filter Size) /Stride) +1 = ((14 – 5)/1) +1 = 10. 
Therefore, the dimension of the 16 activation maps will be 10x10.

**Second Max Pool layer**

The output activation maps of the second convolutional layer are passed to a max pooling layer of filter size 2 and Stride of 2. The output activation map dimension can be obtained from the below equation: 
((Input width – Filter Size) /Stride) +1 = ((10 – 2)/2) +1 = 5. 
Therefore, the dimension of the output of max pool layer is 5x5.

**Fully Connected Layers**

The output of the max pool layer which is 16 5x5 activation maps is flattened into 16x5x5 = 400 elements and is passed as the input of the First Fully Connected layer.
Here the features extracted using the convolutional layers is passed as input to the fully connected layers to perform the classification task. There is a ReLu activation layer present after each fully connected layer to induce nonlinearity.
The output of the First Fully Connected layer is passed to the Second Fully Connected Layer and its output is passed to the Third Fully Connected Layer. The output dimension of the third fully connected layer is 10, which corresponds to the 10 digits that need to be classified.

# Loss Function
Loss function is the measure of how good the model can predict compared to the ground truth. To quantify how a given model performs, the loss function evaluates to what extent the actual outputs are correctly predicted by the model outputs. When the difference between prediction and ground truth is large then the loss will also be large and if the prediction and ground truth are similar then loss will be small. The aim of network training to reduce this loss. In neural network the loss will be computed using a loss function. Then for performing the gradient descent, the gradient of each parameter will be computed. This calculated gradient will be used by the optimizer to update the parameter weight.
The task of this project is to classify the images of the input digits. In this project I will be using the Cross-Entropy Loss function as the task is to classify between C classes.

**Cross Entropy Loss**

The cross-entropy loss of a multiclass classification can be given as 
LCross-Entropy = − P(Yi truth) log( P(Yi predicted) ) 
P(Yi truth) =1 for the Class associated to that data instance and 0 for all the other classes.
The P(Yi predicted) is found using the SoftMax function given below. 
P(Yi predicted) = exp(Yi) / (∑For all Classes output exp(Y))  
SoftMax is the ratio of exponent of the value predicted for the ground truth class by the model with the sum of exponent of the value predicted for all the classes. If the value predicted for a class is high, then the probability will also be high. 
Therefore, the loss function LCross-Entropy = − P(Yi truth) log( P(Yi predicted) )  will be low when the probability of P(Yi predicted) is high as the model is doing a good job. But if P(Yi predicted) is low then the loss will be high, and the model will be heavily penalized for this wrong prediction. 
After the loss is calculated  the gradient of loss function with respect to all the weight parameters and bias along with the input and output of each layers will be calculated using the backward propagation  to use it in the gradient descent step.

![image](https://user-images.githubusercontent.com/29349268/118026586-89e54300-b393-11eb-90eb-28949b779954.png)

Note. We can observe that the loss will be very high if the predicted probability is low and vice versa. From “Loss Functions” by Machine Learning Glossary, (https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#:~:text=Cross%2Dentropy%20loss%2C%20or%20log,diverges%20from%20the%20actual%20label). Copyright 2017 Revision 91f7bc03.

**Cross Entropy Loss Function of PyTorch Framework**

In this project I am using the cross-entropy loss function provided by PyTorch - torch.nn.CrossEntropyLoss(). The predicted values, i.e. the 10 output values of the last fully connected layer and the true class label will be given as the input to the loss function. Once the loss is calculated, PyTorch framework provides a simple way to calculate all the gradients required for the backpropagation. By calling loss.backward() the network will calculate all the gradients associated with each parameter. 
Below is a code snippet explaining the steps in loss function calculation and gradient generation.
```
#Initilaizing the Loss. Here I chose Cross Entropy loss as it is a classification problem.
criterion = nn.CrossEntropyLoss()
#Calculating the loss between predicted and label values.
 loss = criterion( predictedValues , trueLabel)
#Calculating the gradient associated with each parameter in the model with the loss
loss.backward()
```

# Optimizer Selection
The optimizer decides how the weights will be updated once the gradients for back propagation is found. In this project I am using the optimizers provided by the PyTorch framework. In the following section, I will be comparing multiple optimizer performance with a fixed initial learning rate.

**Comparison between multiple optimizers**
For selecting the best optimizer, I compared multiple optimizers provided by PyTorch keeping the learning rate =0.001.  Details of the optimizers which I checked is given below.

_**SGD (Stochastic Gradient Descent)**_ – This is the ordinary Stochastic Gradient descent. 

_**Adadelta**_ - This optimization is a stochastic gradient descent method that is based on adaptive learning rate.

_**AdaGrad**_ - This algorithm individually adapts the learning rates of all model parameters by scaling them inversely proportional to the square root of the sum of all historical squared values of the gradient. This improves the learning rates, especially in the convex regions of error function

_**RMSProp**_ (Root Mean Square Propagation) - RMSprop improves upon AdaGrad algorithms uses an exponentially decaying average to discard the history from extreme past so that it can converge rapidly after finding a convex region.

_**Adam**_ - This optimizer combines RMSprop and momentum methods. Adam is generally regarded as robust to hyperparameters and works well on many applications.

_**Adamax**_ - It is a variant of Adam based on the infinity norm. Adamax is sometimes superior to adam, especially in models with embeddings.


![image](https://user-images.githubusercontent.com/29349268/118026747-bef19580-b393-11eb-8a2b-aebcf728a4a4.png)

Note. Graph illustrates how fast the optimizer can reduce the loss after multiple epochs.

From the comparison between different optimizers Adam and RMSProp showed similar performance with large decrease in loss in fewer epochs. 
SGD and Adadelta has the slowest loss reduction and were not able to attain the loss reduction obtained by other optimizers even after multiple epochs.
From above observations I was able to get into conclusion that Adam was the best choice as the loss decreased much faster than the rest of the optimizers.
**Comparison between multiple Learning Rate of the Optimizer**

The next hyperparameter that need to be decided is the initial learning rate of the optimizer. To find this parameter I tested the decrease in loss after multiple epochs by initializing the Adams optimizer with multiple learning rates. 
The below graph shows the result. 

![image](https://user-images.githubusercontent.com/29349268/118026865-e8122600-b393-11eb-9270-7ac2e1bed85f.png)

Let us analyse each learning rate with respect to their graph.

_10_ - An oscillating graph is observed i.e. loss keeps on increasing and decreasing. Seems like the weight parameters are diverging away from the global minima as the learning rate is high.

_1_ - The loss decreases at the beginning but remains constant after few epochs. Since the learning rate is high it is not able to converge to local minimum. 

_0.1_	– The graph is similar to the learning rate 1. 

_0.01_ – We can observe that the loss decreases during the initially epochs but while reaching the final epochs the loss starts increasing. Seems like the learning rate is high at the final stages and hence diverging away from local minimum.

_0.001_ – We can observe that the loss decreases during the initial epochs and keeps on decreasing till it reaches near to zero. This is the behaviour of a good learning rate and is selected as the initial learning rate for training.

_0.0001_ – The graph is similar to 0.001, but the initial decrease in loss is less compared to 0.001.

_0.00001_ – The initial decrease in loss is less compared to 0.001 and 0.0001. This is because the rate of learning is slow due to small Learning rate parameter. We can observe that loss is decreasing after each epoch but was not able to attain the minimum loss even after 150 epochs due to slow learning.

From the above observations, value of the initial learning rate of Adam optimizer is selected as 0.001

**Adam Optimizer of PyTorch Framework**

In this project I am using the Adam Optimizer function provided by PyTorch - torch.optim.Adam() with initial learning rate = 0.001 and all the default parameters betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False. 
Before each mini batch you will have to remove the gradients calculated in the previous batch by calling optimizer.zero_grad(). Once the gradients a re calculated by the loss function we can update the weights of the parameters using the gradient by simply calling optimizer.step(). This will update all the weights in the neural network.
Below is a code snippet explaining the steps in Optimizer initialization and parameter weights updating.
```
#Initilaizing the optimizer with default parameters.
#lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False are the default parameters
optimizer=optim.Adam(model.parameters(), lr=0.001)
#making the previous calculated gradient to zero for the new mini batch.        
optimizer.zero_grad()
#Updating the model parameters with the gradient found.
optimizer.step()
```

# Training and Testing the Model
Once the Optimizer, Learning rate and Loss function is decided. We can start the training of the model.  For training I am using the PyTorch frame work. A total of 200 epochs tarining on 60000 MNIST data set was done. The model from each epoch is tested with the testing set of 10000 data sets. The loss of training , the error percentage in both traing and testing were noted.

![image](https://user-images.githubusercontent.com/29349268/118026951-fe1fe680-b393-11eb-9df5-87743fbc5839.png)


The lowest testing error 0.72% was reported for the model in the 128 th epoch.  So as per procedure the model with the lowest error rate in test data set will be choosen for future classifications.


![image](https://user-images.githubusercontent.com/29349268/118027013-0aa43f00-b394-11eb-807d-a299a5ad2c97.png)

![image](https://user-images.githubusercontent.com/29349268/118027087-1db70f00-b394-11eb-8cf3-1fd3737acda6.png)

Note. The lowest testing error is found in the 128th epoch model i.e. 0.72%. 

# Improvements in the Suggested Network
The network which I have used is LeNet5 architecture with 61,706 parameters is a small network compared to the state-of-the-art architectures like ResNet. Below are few suggestions to improve the performance of the task.

**Use Deeper Models with more parameters**

One of the ways to improve the performance of a specific task is to use a deeper network with more parameters like ResNet which has 23,522,250 which is much bigger than the LeNet5. 
To verify this, I trained the same data set with a ResNet for the same number of epochs. 
 For this experiment, I used the ResNet model provided by the PyTorch without any pretraining. The model was trained with optimizer and learning rate. Below Graph Shows the comparison between ResNet and LeNet5 error rate in test set.

![image](https://user-images.githubusercontent.com/29349268/118027159-332c3900-b394-11eb-9b7a-610e7a478e81.png)

From the graph, it is evident that the error percentage of ResNet is better and it shows better performance than the LeNet5.

![image](https://user-images.githubusercontent.com/29349268/118027227-45a67280-b394-11eb-8b3d-018597f917c4.png)

Using deeper networks will make the network to be able to express more compared to small networks. But sometimes using deeper networks can be an overkill for small tasks. 

**Use Ensemble Learning**

Ensemble learning is another method to improve the accuracy of the tasks. The multiple classifiers of the same type will be trained. For testing the output with more votes will the output of the total network. This will improve the accuracy compared to using one classifier.
This will also reduce the chances of the model become overfitted to the training data as the data will be distributed among the different classifiers. The most used ensemble techniques like bagging or boosting can be implemented.

![image](https://user-images.githubusercontent.com/29349268/118027287-57881580-b394-11eb-9eba-cfe6ea21c451.png) 

Note. The output of the 3 networks will be compared and the one with more votes will the output of the total network. From “Review: LeNet-1, LeNet-4, LeNet-5, Boosted LeNet-4 (Image Classification)” by Medium (https://sh-tsang.medium.com/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17).


# References

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998, November). Gradient-Based learning applied to Document recognition. Yann LeCun's. http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
LeCun, Y., Cortes, C., & Burges, C.J.C. (n.d.). The MNIST Database of handwritten digits. Yann LeCun's. https://yann.lecun.com/exdb/mnist/
Lim, S. H., Young, S. R., & Patton, R. (2016). Example images from the MNIST dataset [Image]. Researchgate. https://www.researchgate.net/publication/306056875_An_analysis_of_image_storage_systems_for_scalable_training_of_deep_neural_networks
Log Loss when true label = 1 [Graph]. (n.d.). Machine Learning Glossary. https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#:~:text=Cross%2Dentropy%20loss%2C%20or%20log,diverges%20from%20the%20actual%20label
[Online Image]. (2017). Medium - towards data science. https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
PyTorch. (n.d.). CrossEntropyLoss. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
PyTorch. (n.d.). Linear. https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
PyTorch. (n.d.). MaxPool2d. https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
PyTorch. (n.d.). ReLU. https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
PyTorch. (n.d.). Torch.optim. https://pytorch.org/docs/stable/optim.html
Tsang, S. (2018, August 8). Review: Lenet-1, lenet-4, lenet-5, boosted lenet-4 (Image classification). Medium. https://sh-tsang.medium.com/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17

