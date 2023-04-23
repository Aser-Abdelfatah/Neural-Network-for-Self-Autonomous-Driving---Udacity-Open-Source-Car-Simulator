# NeuroDrive: Neural Networks for Autonomous Driving
This project leverages the power of deep neural networks to design a model capable of simulating human driving in a virtual environment with the help of Udacity open-source self-driving car simulator.

# Introduction 
The development of deep neural network models has revolutionized the field of autonomous driving, providing a powerful tool for training autonomous cars to navigate and make decisions on the road. This project will leverage the power of deep neural networks to design a model capable of simulating human driving in a virtual environment with the help of open-source self-driving car simulators. Developing a self-driving car is a complex process that requires automating an array of human functions, such as perception of surroundings, following traffic laws, and decision-making. A typical self-driving car would incorporate many machine learning models with each model performing different functions; for instance, the self-driving car needs a computer-vision model to identify traffic lights and road signs as well as a reinforcement-learning model to make decisions, such as whether the car will take a turn. Also, the development of self-driving cars extends beyond the machine learning algorithms to entail development of sensors, radars, and hardware components that can provide accurate inputs to the machine learning models. I decided to use an open-source car simulator to focus only on the machine learning portion of the self-driving car development. Previous research has examined the development of self-driving cars using simulators. There are many open-source self-driving car simulators with each simulator having an edge over the others in particular areas and lagging behind in other areas. Some of the important factors to consider while picking a simulator are perception, localization, vehicle control, and creation of dynamic 3D virtual environments (Kaur et al.). The simulator I'm using is UDACITY’s autonomous car simulator. Previous research that analyzed this simulator pointed out an important disadvantage, which is the absence of noise in the simulator environment making the simulator unrealistic in the real world (Duong). Nonetheless, the car simulator allows me to source the data by using the training mode, which is a game-like mode wherein I drive the car in a track, and I take decisions to move the car depending on the surrounding environment; the model provides us with a complete dataset of the car surroundings in form of three images from the front and sides of the car and the decisions I took in the form of the steering angle, speed, and acceleration. Using this data, I will design a deep-learning-based regression model that takes the images of the surrounding as an input and predicts the correct steering angle as an output.

# Methods 
## Data Augmentation:
### Balancing the number of Turns
The data was sourced from Udacity's open-source car simulator using two different routes. The first route didn’t have any right turns, so this bias needs to be accounted for or otherwise, the model will fail to predict any right turns. The data was augmented by flipping a random set of the images and negating the steering angle. Also, the distribution of the steering angle was unbalanced because most steering angles were just pointing straight. This unbalance was tackled by deleting a randomized set of the data that pointed straight.

![image](https://user-images.githubusercontent.com/47282229/233096879-8d650169-3d87-4855-b2ef-2c2a7648c858.png)

This is a histogram of the steering angles of the first route before augmenting data.

![image](https://user-images.githubusercontent.com/47282229/233097025-7e1825b7-9eb6-4d48-a6fa-f582de42d167.png)

This is the distribution after data augmentation

![image](https://user-images.githubusercontent.com/47282229/233097291-6fb66353-efec-443b-b9e3-d543d4dc2e23.png)

The number of right and left turns is more balanced compared to the original data, and the distribution of the data is more uniform after removing some of the data corresponding to going straight.

### Noise
To make the model more robust and deployable, I augmented the data by adding noise in the form of randomized rotations, shifts, and blurs. Furthermore, I decided to change the brightness of some of the images randomly to ensure the model is capable of working in both day and night and in shade. Also, the added noise will help the model escape local minimum and avoid over-fitting.
This image shows an image after and before adding noise.
![image](https://user-images.githubusercontent.com/47282229/233096656-ca066551-9f71-456a-a3eb-f4a2e67118cb.png)

## Data Pre-Processing
### Normalization
I normalized the pixel values to lie between 0 and 1 to reduce the effect of variations in lighting, contrast, and color.
### YUV
I decided to use YUV color space over RGB to separate color and brightness information, allowing for more efficient analysis of the image data.
### Standardization
I standardized the pixel values to have consistent mean and standard deviation to standardize the brightness and contrast of the images, making them more comparable and easier to process by the neural network.
### Resize
I cropped the image to disregard irrelevant features and decreased the image size for lighter, faster processing
### Gaussian Blur
I used Gaussian Blur in image pre-processing to reduce image noise and smooth out details. 

This image show the original image vs the pre-processed image.

![image](https://user-images.githubusercontent.com/47282229/233095819-8d84fbda-7a07-42b8-8c5e-af4118b2395a.png)

## Model Design
### Model 1
The model used 7 convolution layers with each one having 1.5 more filters than the previous one. The first convolution layer had 18 filters, and the last one had 128 filters. To lighten the model, tackle over-fitting, and down-sample the output feature map, I used four max_pooling layers, halving the output shape after every two convolutional layers and after the last layer. After that, I used a flatten layer to flatten the output of convolutional layers and four dense layers as the backbone of the fully connected model that outputs the steering angle. To reduce overfitting, I used three dropout layers, dropping 50% of the inputs after each of the first three dense layers. In modile compiling, ADAM optimizer was used with a learning rate of 0.0001.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________

conv2d_23 (Conv2D)           (None, 64, 198, 18)       504       
_________________________________________________________________
conv2d_24 (Conv2D)           (None, 62, 196, 24)       3912      
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 31, 98, 24)        0         
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 29, 96, 36)        7812      
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 27, 94, 48)        15600     
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 13, 47, 48)        0         
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 11, 45, 64)        27712     
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 9, 43, 96)         55392     
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 4, 21, 96)         0         
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 2, 19, 128)        110720    
_________________________________________________________________
max_pooling2d_15 (MaxPooling (None, 1, 9, 128)         0         
_________________________________________________________________
dropout_12 (Dropout)         (None, 1, 9, 128)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_9 (Dense)              (None, 100)               115300    
_________________________________________________________________
dropout_13 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 50)                5050      
_________________________________________________________________
dropout_14 (Dropout)         (None, 50)                0         
_________________________________________________________________
dense_11 (Dense)             (None, 10)                510       
_________________________________________________________________
dropout_15 (Dropout)         (None, 10)                0         
_________________________________________________________________
output (Dense)               (None, 1)                 11        
_________________________________________________________________

Total params: 342,523
_________________________________________________________________

Trainable params: 342,523
_________________________________________________________________

Non-trainable params: 0
_________________________________________________________________
None


### Model 2
Model 2 is a slightly lighter version of model 1, using 5 convolutional layers instead of 7 while keeping the same number of layers of other types. The focus was on decreasing the number of convolutional layers because they are by far more computationally intensive compared to fully-connected, drop_out, and max_pooling layers. This model has 222,395 and consists of a total of 18 layers. 


_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 64, 198, 24)       672       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 32, 99, 24)        0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 30, 97, 36)        7812      
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 28, 95, 48)        15600     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 14, 47, 48)        0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 12, 45, 64)        27712     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 6, 22, 64)         0         
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 4, 20, 64)         36928     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 2, 10, 64)         0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 2, 10, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1280)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 100)               128100    
_________________________________________________________________
dropout_5 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_6 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_7 (Dropout)          (None, 10)                0         
_________________________________________________________________
output (Dense)               (None, 1)                 11        
_________________________________________________________________
Total params: 222,395
_________________________________________________________________

Trainable params: 222,395
_________________________________________________________________

Non-trainable params: 0



# Discussion 
## Evaluation Metric
The main metric used to evaluate model performance is the loss calculated as the mean squared error between the actual and the predicted steering angle. MSE was used over MAE to penalize extreme errors because in real-life, a considerably poor prediction will cause a car accident and may lead to loss of life and property.

## Choice of Activation Function
I tested the usage of SoftMax, ReLU and eLU. While eLU and ReLU shared quite similar performance in terms of loss value, the models running with eLU were over-fitting while those running with ReLU were not. SoftMax was considerably worse with loss values ranging around 0.3 while that of eLU and ReLU ranging around 0.09 and 0.08 respectively. ReLU was picked as the main activation function to counter over-fitting. 

![image](https://user-images.githubusercontent.com/47282229/233093506-94cd5cf1-9434-441b-a71c-4c8f63923912.png)
![image](https://user-images.githubusercontent.com/47282229/233093616-fcfea842-4574-49a5-9eaf-69ab432a9bc4.png)


## Lighter Model vs Original
While the original model had 342,523 parameters and 20 layers compared to 222,395 parameters and 18 layers for the lighter model, the running time -– using mini-batch of size 100  -– of the former was 30 minutes and 50 seconds while that of the latter was 27 minutes and 55 seconds. So, there was no significant difference when it came to running time. The validation loss of the original model --  while using ReLU activation function and mini-batch of size 100 -- was 0.0847 while it was 0.0955 for the lighter model. While using the eLU activation function, the validation loss after the last epoch for the original model was 0.0978 compared to 0.0927 for the lighter model. With one model having lower loss with ReLU and the other having lower loss with eLU and given that the difference in losses and running time are almost negligible, I can say there is no significant difference between the two models. The lighter model, however, was less prone to over-fitting as shown in the graphs.

![image](https://user-images.githubusercontent.com/47282229/233093796-e28124ad-fac9-48de-8440-f6eb9ab38949.png)
![image](https://user-images.githubusercontent.com/47282229/233093971-7196b037-d641-489b-83c2-9856f69cc27d.png)

![image](https://user-images.githubusercontent.com/47282229/233094290-6e37a173-748c-49d3-b3a4-053d1e6cbe55.png)
![image](https://user-images.githubusercontent.com/47282229/233094117-f2e511e3-2ac7-4af0-938e-3ab6e1389a5c.png)


## Stochastic gradient descent vs Mini-batch stochastic gradient descent vs Batch gradient descent
In terms of running time, the stochastic gradient descent took 53 minutes and 40 seconds compared to 30 minutes and 50 seconds for the mini-batch stochastic gradient with the mini-batch being set to 100. The limited computational capabilities of my local machine made it impossible to run the batch gradient descent as the tensor generated didn’t fit into memory. The largest batch I could use was 1/16 the size of the data. The running time of the model with mini-batch of size 1/16 of the data was 27 minutes and 38 seconds. 1/16 of the data size is 424 images. The GPU I'm using is RTX 2060 Ti laptop version for reference.

In terms of over-fitting, both the mini-batch stochastic gradient with mini-batch of size 100 and mini-batch of size 1/16 of the data were not over-fitting after 25 epochs while the stochastic gradient descent started over-fitting in the fourth epoch. ReLU activation function was used while comparing the models

In terms of the loss values, the loss value of stochastic gradient descent in the last epoch was 0.0959 while it was 0.0847 for the mini-batch stochastic gradient descent and 0.0862 for the 1/16 model. From the results, the mini-batch stochastic gradient descent with batch of size 100 achieved lowest validation loss, had a good running time, and avoided over-fitting, so it's preferred over the alternatives.
 
 ![image](https://user-images.githubusercontent.com/47282229/233094456-2c409345-fb6f-4880-8df6-b6380985901e.png)
![image](https://user-images.githubusercontent.com/47282229/233094607-22a68b8d-d757-4e22-bfee-061df1a996d8.png)
![image](https://user-images.githubusercontent.com/47282229/233095554-dd35aec8-5c16-42f4-89cb-5cca1a28c549.png)

## ResNet50, ResNet101, Xception, and MobileNetV2
To evaluate my model performance, I decided to deploy various pre-designed models along with pre-trained weights from Keras Applications. I chose Xception, ResNet50, and ResNet101 as the heavy models and MobileNetV2 as the light model. The validation loss for the four models ranged from 0.3 to 0.32 in the 25 epochs, which implies that either they were stuck in a local minimum or that the models reached their maximum performance. Also, the heavy models took significantly longer compared to my lighter model, which was expected given the depth and the number of parameters. For instance, the Xception model took 168 minutes to complete the 25 epochs and the ResNet 50 took 268 minutes. I was planning for further testing with those models to determine whether the model was actually stuck in a local minimum, but the computational power of my local device was an obstacle.

![image](https://user-images.githubusercontent.com/47282229/233204129-0b5e1a7d-1721-4d40-9467-cf1cf44493b8.png)
![image](https://user-images.githubusercontent.com/47282229/233204157-cc4327f6-46af-4368-82db-fee372195a50.png)
![image](https://user-images.githubusercontent.com/47282229/233204206-e5d2bb41-4341-498a-bfa9-5547fd908b4f.png)

## Conclusion
In conclusion, the project successfully utilized a convolutional neural network model to accurately predict the steering angle of a self-driving car, surpassing some of the best pre-trained models, such as ResNet50, ResNet101, and Xception. The model was designed with multiple convolutional and fully connected layers, along with appropriate ReLU activation function to enable effective feature extraction and decision-making. The data augmentation techniques, including adding noise and balancing the data, the choice of mini-batch gradient descent, and the use of dropout layers added to the robustness of the model evident by the fact that it was not over-fitting even after 25 epochs -- while using ReLU activation function. A signifcant pitfall worth noting in model testing and training is the computational limitations of the machine, which prevented further testing of the pre-trained models and the use of batch gradient descent. So, further testing is essential as explained in the next section. Over all, the project highlights the potential of using deep learning techniques, such as convolutional neural networks, to take decisions based on image data and being deployed in developing advanced self-driving car systems. 

# Reflection 
## Future Development and Recommendations
Another way to test the model's robustness is using the data from route 1 as the training data and the data from route 2 as the testing data and vice versa to see how the model works in completely new environments. Contingent on available computational resources, the model shold be tested with varying batch sizes larger than 1/16 the size of the data. Furthermore, to tune more hyper-parameters, such as the learning rate of the Adam optimizer, the choice of activation function, and the depth and the width of layers, further testing has to be executed while varying those parameters until the best results are achieved in terms of validation loss and over-fitting. Also, the effect of noise and pre-processing functions should be quantified through testing with and without the noise and pre-processing function. In addition, even though the model is considerably faster than the pre-trained models, such as ResNet50 and Xception, it's considerably slow, taking 30 minutes for training and testing. One way to optimize the model is by using quantization to reduce the number of parameters and increase running speed so that the neural networks can be deployed on embedded systems. The efficiency of the quantization will be measured by comparing the running speed of the original model and the quantized model while keeping a negligible drop in accuracy.

# References:
* Duong, M.T., Do, T.D., Le, M.H. (2018). Navigating Self-Driving Vehicles Using Convolutional Neural Network. 2018 4th International Conference on Green Technology and Sustainable Development (GTSD), 607-610, https://ieeexplore.ieee.org/abstract/document/8595533.
* Kaur, P., Taghavi, S., Tian, Z., Shi, W. (2021). A Survey on Simulators for Testing Self-Driving Cars. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9499331&tag=1

