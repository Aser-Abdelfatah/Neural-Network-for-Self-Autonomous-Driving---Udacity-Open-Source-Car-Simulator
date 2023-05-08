## By Aser Atawya and Michele Tang
This is a two-part project on neural networks and self-driving cars. The first part describes a technical project where we trained two models to simulate a self-driving car and the second part is an ethics investigation.

# Building a Model
## Abstract 
This project leveraged the power of deep neural networks to design a model capable of simulating human driving in a virtual environment with the help of Udacity open-source self-driving car simulator. Before training the model, we accounted for biases, such as the lack of right turns and a large number of straight steering angles and pre-processed the data. We also added noise to the data by adding randomized rotations, shifts, and blurs as well adjusting the brightness of some images. We trained a model with 7 convolutional layers, using ADAM optimizer, ReLU activation function, and learning rate of 0.0001. Ultimately, the model had a total of 20 layers and 345,523 parameters. Additionally, the model used mini-batch stochastic gradient descent with a batch size of 100. To evaluate our model, we ran the model on the simulator and calculated loss value using the mean squared error and compared the results with some of the pre-trained models, such as ResNet50, ResNet101, and Xception. Our model outperformed the pre-trained models in terms of loss value and running time. We also wanted to quantify the effect of adding noise and found that adding noise significantly reduced validation and training loss. However, the model with noise did start to overfit in the 24th and 25th epochs. We also compared the efficacy of using stochastic gradient descent over mini-batch stochastic gradient descent. We found that mini-batch gradient descent did not overfit after 25 epochs whereas stochastic gradient descent did. Mini-batch gradient descent also had lower validation loss. 

## Introduction
The development of deep neural network models has revolutionized the field of autonomous driving, providing a powerful tool for training autonomous cars to navigate and make decisions on the road. This project will leverage the power of deep neural networks to design a model capable of simulating human driving in a virtual environment with the help of open-source self-driving car simulators.

Developing a self-driving car is a complex process that requires automating an array of human functions, such as perception of surroundings, following traffic laws, and decision-making. A typical self-driving car would incorporate many machine learning models with each model performing different functions; for instance, the self-driving car needs a computer-vision model to identify traffic lights and road signs as well as a reinforcement-learning model to make decisions, such as whether the car will take a turn. Also, the development of self-driving cars extends beyond the machine learning algorithms to entail development of sensors, radars, and hardware components that can provide accurate inputs to the machine learning models.

We decided to use an open-source car simulator to focus only on the machine learning portion of the self-driving car development. And the most common form of machine learning algorithms used in autonomous cars is neural networks. Four of the most common deep learning methods used in the development of self-driving cars are convolutional neural networks, recurrent neural networks, auto-encoders, and deep reinforcement learning[^1]. The problems the models need to solve include obstacle detection, scene classification and understanding, lane recognition, path planning, motion control, and traffic signs and lights recognition[^1]. Previous research has put emphasis on safety by developing models capable of dealing with bad drivers of other cars, right of way laws, unstructured roads, pedestrians, and responsibility for actions[^2].

Previous research has also examined the development of self-driving cars using simulators. There are many open-source self-driving car simulators, such as CARLA and LGSVL, with each simulator having an edge over the others in particular areas and lagging behind in other areas. Some of the important factors to consider while picking a simulator are perception, localization, vehicle control, and creation of dynamic 3D virtual environments[^3]. The simulator we are using is Udacity’s autonomous car simulator[^4]. Previous research that analyzed this simulator pointed out an important disadvantage, which is the absence of noise in the simulator environment making the simulator unrealistic in the real world[^5]. Nonetheless, the car simulator allows us to source the data by using the training mode, which is a game-like mode wherein we drive the car in a track, and we take decisions to move the car depending on the surrounding environment; the model provides us with a complete dataset of the car surroundings in form of three images from the front and sides of the car and the decisions we took in the form of the steering angle, speed, and acceleration. Using this data, we will design a deep-learning-based regression model that takes the images of the surrounding, which can come in the form of a front-facing camera photo or a side-facing camera photo as an input and predicts the correct steering angle as an output. This project sets itself apart from Udacity tutorials and other projects on the internet by deploying pre-designed models, such as ResNet50, Resnet101, and Xception, to compare and contrast their performance with our model.

<img src="https://user-images.githubusercontent.com/47282229/234185589-9713bb93-a7db-47df-8003-e164b73da702.png"  width="1200" height="300">
 

## Methods
### Data Sourcing
The self-driving car simulator that we are currently using to collect the dataset and test the models is Udacity’s autonomous car simulator. One of the important drawbacks of the simulator is that it doesn’t have much functionality; for instance, there are no traffic signs, pedestrians, etc. It only provides training and testing for taking turns, speeding, and slowing down. The simulator has two tracks: the first one is fairly simple while the second is very complex with hard turns, bridges, shade, and inclines.

### Data Augmentation:
#### Balancing the number of Turns
The data was sourced from Udacity's open-source car simulator using the two different routes. The following figure displays an image from each route.

<img src=https://user-images.githubusercontent.com/47282229/233096879-8d650169-3d87-4855-b2ef-2c2a7648c858.png width="1200" height="250">

The first route didn’t have any right turns, so this bias needs to be accounted for or otherwise, the model will fail to predict any right turns. The data was augmented by flipping a random set of the images and negating the steering angle.
```
# this functions flips the image and reverses the steering angle to account for the flip
def balance(image, angle):
    image = cv2.flip(image, 1)
    angle = -angle
    return image, angle


# balance right and left turns
pivot = len(images_route_1_balanced)
for i in range(3000):
    # Gets a random index of route_1 data since the imbalance is concentrated in route_1 data
    index = random.randint(0, pivot - 1)
    images_balanced[index], angles_balanced[index] = balance(images_balanced[index], angles_balanced[index])
    images_route_1_balanced[index], angles_route_1_balanced[index] = balance(images_route_1_balanced[index], angles_route_1_balanced[index])

```
This figure displays the effect of the balance function.

<img src=https://user-images.githubusercontent.com/47282229/234185181-7770ecc5-6ecc-4b07-9d57-ebfbf2cef95a.png width="1200" height="250">


Also, the distribution of the steering angle was imbalanced because most steering angles were just pointing straight. This imbalance was tackled by deleting a randomized set of the data that pointed straight. Deleting a randomized set of that data was the easiest solution given that the collected data is fairly large with 19401 images. Other methods to tackle this imbalance include editing the data loader to load only balanced batches of the data, which would stop the need to delete any parts of the data but would be harder in implementation. This histogram shows the distribution of the steering angles of the first route before augmenting data.
<p align="center">
<img src=https://user-images.githubusercontent.com/47282229/233097025-7e1825b7-9eb6-4d48-a6fa-f582de42d167.png width="500" height="325">
</p>

And this histogram shows the distribution after data augmentation
<p align="center">
<img src=https://user-images.githubusercontent.com/47282229/233097291-6fb66353-efec-443b-b9e3-d543d4dc2e23.png  width="500" height="325">
</p>
The number of right and left turns is more balanced compared to the original data, and the distribution of the data is more uniform after removing some of the data corresponding to going straight.

#### Noise
To make the model more robust and deployable, we augmented the data by adding noise in the form of randomized rotations, shifts, and blurs. Furthermore, we decided to change the brightness of some of the images randomly to ensure the model is capable of working in both day and night and in shade. Also, the added noise will help the model escape local minimum and avoid overfitting.
This image shows an image after and before adding noise.

<img src=https://user-images.githubusercontent.com/47282229/233096656-ca066551-9f71-456a-a3eb-f4a2e67118cb.png width="1200" height="250">

### Data Pre-Processing
#### Normalization
We normalized the pixel values to lie between 0 and 1 to reduce the effect of variations in lighting, contrast, and color.
#### YUV
We decided to use YUV color space over RGB to separate color and brightness information, allowing for more efficient analysis of the image data[^19].
#### Standardization
We standardized the pixel values to have consistent mean and standard deviation to standardize the brightness and contrast of the images, making them more comparable and easier to process by the neural network.
#### Resize
We cropped the image to disregard irrelevant features, such as the sky and the verge of the road, and resized the image for lighter, faster processing
#### Gaussian Blur
We used Gaussian Blur in image pre-processing to reduce image noise and smooth out details.

This image shows the original image vs the pre-processed image.

<img src= https://user-images.githubusercontent.com/47282229/233095819-8d84fbda-7a07-42b8-8c5e-af4118b2395a.png width="1200" height="250">

### Model Design
#### Model 1
The model used 7 convolution layers with each one having 1.5 more filters than the previous one to capture more abstractions and complex patterns. The first convolution layer had 18 filters, and the last one had 128 filters. To lighten the model, tackle overfitting, and down-sample the output feature map, we used four max_pooling layers, halving the output shape after every two convolutional layers and after the last layer. After that, we used a flatten layer to flatten the output of convolutional layers and four dense layers as the backbone of the fully connected model that outputs the steering angle. To reduce overfitting, we used three dropout layers, dropping 50% of the inputs after each of the first three dense layers. In model compiling, ADAM optimizer was used with a learning rate of 0.0001. The model had a total of 20 layers and 342,523 parameters.

<table>
  <tr>
    <td>Layer Number</td>
    <td>Layer Type</td>
    <td>Layer Input</td>
    <td>Layer Output</td>
    <td>Number of Parameters</td>
    <td>Layer Hyperparameter</td>
    <td>Hyperparameter Values</td>
  </tr>
  <tr>
    <td rowspan="3">1</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 66, 200, 3)</td>
    <td rowspan="3">(None, 64, 198, 18)</td>
    <td rowspan="3">504</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">18</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="3">2</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 64, 198, 18)</td>
    <td rowspan="3">(None, 62, 196, 24)</td>
    <td rowspan="3">3912</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">24</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="2">3</td>
    <td rowspan="2">MaxPooling2D</td>
    <td rowspan="2">(None, 62, 196, 24)</td>
    <td rowspan="2">(None, 31, 98, 24)</td>
    <td rowspan="2">0</td>
    <td rowspan="1">kernel size</td>
    <td rowspan="1">(2,2)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> None </td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="3">4</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 31, 98, 24)</td>
    <td rowspan="3">(None, 29, 96, 36)</td>
    <td rowspan="3">7812</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">36</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
    </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="3">5</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 29, 96, 36)</td>
    <td rowspan="3">(None, 27, 94, 48)</td>
    <td rowspan="3">15600</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">48</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="2">6</td>
    <td rowspan="2">MaxPooling2D</td>
    <td rowspan="2">(None, 27, 94, 48)</td>
    <td rowspan="2">(None, 13, 47, 48)</td>
    <td rowspan="2">0</td>
    <td rowspan="1">kernel size</td>
    <td rowspan="1">(2,2)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> None </td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  </tr>
  <tr>
    <td rowspan="3">7</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 13, 47, 48)</td>
    <td rowspan="3">(None, 11, 45, 64)</td>
    <td rowspan="3">27712</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">64</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
    </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="3">8</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 11, 45, 64)</td>
    <td rowspan="3">(None, 9, 43, 96)</td>
    <td rowspan="3">55392</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">96</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="2">9</td>
    <td rowspan="2">MaxPooling2D</td>
    <td rowspan="2">(None, 9, 43, 96)</td>
    <td rowspan="2">(None, 4, 21, 96)</td>
    <td rowspan="2">0</td>
    <td rowspan="1">kernel size</td>
    <td rowspan="1">(2,2)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> None </td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="3">10</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 4, 21, 96)</td>
    <td rowspan="3">(None, 2, 19, 128)</td>
    <td rowspan="3">110720</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">128</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="2">11</td>
    <td rowspan="2">MaxPooling2D</td>
    <td rowspan="2">(None, 2, 19, 128)</td>
    <td rowspan="2">(None, 1, 9, 128)</td>
    <td rowspan="2">0</td>
    <td rowspan="1">kernel size</td>
    <td rowspan="1">(2,2)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> None </td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">12</td>
    <td rowspan="1">Dropout</td>
    <td rowspan="1">(None, 1, 9, 128)</td>
    <td rowspan="1">(None, 1, 9, 128)</td>
    <td rowspan="1">0</td>
    <td rowspan="1">Rate</td>
    <td rowspan="1">0.5</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">13</td>
    <td rowspan="1">Flatten</td>
    <td rowspan="1">(None, 1, 9, 128)</td>
    <td rowspan="1">(None, 1152)</td>
    <td rowspan="1">0</td>
    <td rowspan="1"></td>
    <td rowspan="1"></td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">14</td>
    <td rowspan="1">Dense</td>
    <td rowspan="1">(None, 1152)</td>
    <td rowspan="1">(None, 100)</td>
    <td rowspan="1">115300</td>
    <td rowspan="1">Units</td>
    <td rowspan="1">100</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">15</td>
    <td rowspan="1">Dropout</td>
    <td rowspan="1">(None, 100)</td>
    <td rowspan="1">(None, 100)</td>
    <td rowspan="1">0</td>
    <td rowspan="1">Rate</td>
    <td rowspan="1">0.5</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">16</td>
    <td rowspan="1">Dense</td>
    <td rowspan="1">(None, 100)</td>
    <td rowspan="1">(None, 50)</td>
    <td rowspan="1">5050</td>
    <td rowspan="1">Units</td>
    <td rowspan="1">50</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">17</td>
    <td rowspan="1">Dropout</td>
    <td rowspan="1">(None, 50)</td>
    <td rowspan="1">(None, 50)</td>
    <td rowspan="1">0</td>
    <td rowspan="1">Rate</td>
    <td rowspan="1">0.5</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">18</td>
    <td rowspan="1">Dense</td>
    <td rowspan="1">(None, 50)</td>
    <td rowspan="1">(None, 10)</td>
    <td rowspan="1">510</td>
    <td rowspan="1">Units</td>
    <td rowspan="1">10</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">19</td>
    <td rowspan="1">Dropout</td>
    <td rowspan="1">(None, 10)</td>
    <td rowspan="1">(None, 10)</td>
    <td rowspan="1">0</td>
    <td rowspan="1">Rate</td>
    <td rowspan="1">0.5</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
   <tr>
    <td rowspan="1">20</td>
    <td rowspan="1">Dense</td>
    <td rowspan="1">(None, 10)</td>
    <td rowspan="1">(None, 1)</td>
    <td rowspan="1">11</td>
    <td rowspan="1">Units</td>
    <td rowspan="1">1</td>
  </tr>
</table>


#### Model 2
Model 2 is a slightly lighter version of model 1, using 5 convolutional layers instead of 7 while keeping the same number of layers of other types. The focus was on decreasing the number of convolutional layers because they are by far more computationally intensive compared to fully-connected, dropout, and max_pooling layers. For instance, fully connected layers and pooling layers take only 5 to 10% of the computational time[^20]. This model has 222,395 parameters and consists of a total of 18 layers.

<table>
  <tr>
    <td>Layer Number</td>
    <td>Layer Type</td>
    <td>Layer Input</td>
    <td>Layer Output</td>
    <td>Number of Parameters</td>
    <td>Layer Hyperparameter</td>
    <td>Hyperparameter Values</td>
  </tr>
  <tr>
    <td rowspan="3">1</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 66, 200, 3)</td>
    <td rowspan="3">(None, 64, 198, 24)</td>
    <td rowspan="3">672</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">24</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="2">2</td>
    <td rowspan="2">MaxPooling2D</td>
    <td rowspan="2">(None, 64, 198, 24)</td>
    <td rowspan="2">(None, 32, 99, 24)</td>
    <td rowspan="2">0</td>
    <td rowspan="1">kernel size</td>
    <td rowspan="1">(2,2)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> None </td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="3">3</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 32, 99, 24)</td>
    <td rowspan="3">(None, 30, 97, 36)</td>
    <td rowspan="3">7812</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">36</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
    </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="3">4</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 30, 97, 36)</td>
    <td rowspan="3">(None, 28, 95, 48)</td>
    <td rowspan="3">15600</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">48</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="2">5</td>
    <td rowspan="2">MaxPooling2D</td>
    <td rowspan="2">(None, 28, 95, 48)</td>
    <td rowspan="2">(None, 14, 47, 48)</td>
    <td rowspan="2">0</td>
    <td rowspan="1">kernel size</td>
    <td rowspan="1">(2,2)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> None </td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  </tr>
  <tr>
    <td rowspan="3">6</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 14, 47, 48)</td>
    <td rowspan="3">(None, 12, 45, 64)</td>
    <td rowspan="3">27712</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">64</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
    </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="2">7</td>
    <td rowspan="2">MaxPooling2D</td>
    <td rowspan="2">(None, 12, 45, 64)</td>
    <td rowspan="2">(None, 6, 22, 64) </td>
    <td rowspan="2">0</td>
    <td rowspan="1">kernel size</td>
    <td rowspan="1">(2,2)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> None </td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="3">8</td>
    <td rowspan="3">Convolution2D</td>
    <td rowspan="3">(None, 6, 22, 64)</td>
    <td rowspan="3"> (None, 4, 20, 64)</td>
    <td rowspan="3">36928</td>
    <td rowspan="1">Number of Filters</td>
    <td rowspan="1">64</td>
  </tr>
  <tr>
   <td> kernel size</td>
   <td> (3,3)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> (1,1)</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="2">9</td>
    <td rowspan="2">MaxPooling2D</td>
    <td rowspan="2">(None, 4, 20, 64)</td>
    <td rowspan="2">(None, 2, 10, 64)</td>
    <td rowspan="2">0</td>
    <td rowspan="1">kernel size</td>
    <td rowspan="1">(2,2)</td>
  </tr>
  <tr>
   <td> Strides</td>
   <td> None </td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">10</td>
    <td rowspan="1">Dropout</td>
    <td rowspan="1">(None, 2, 10, 64)</td>
    <td rowspan="1"> (None, 2, 10, 64)  </td>
    <td rowspan="1">0</td>
    <td rowspan="1">Rate</td>
    <td rowspan="1">0.5</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">11</td>
    <td rowspan="1">Flatten</td>
    <td rowspan="1">(None, 2, 10, 64)</td>
    <td rowspan="1">(None, 1280)</td>
    <td rowspan="1">0</td>
    <td rowspan="1"></td>
    <td rowspan="1"></td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">12</td>
    <td rowspan="1">Dense</td>
    <td rowspan="1">(None, 1280)</td>
    <td rowspan="1">(None, 100)</td>
    <td rowspan="1">128100</td>
    <td rowspan="1">Units</td>
    <td rowspan="1">100</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">13</td>
    <td rowspan="1">Dropout</td>
    <td rowspan="1">(None, 100)</td>
    <td rowspan="1">(None, 100)</td>
    <td rowspan="1">0</td>
    <td rowspan="1">Rate</td>
    <td rowspan="1">0.5</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">14</td>
    <td rowspan="1">Dense</td>
    <td rowspan="1">(None, 100)</td>
    <td rowspan="1">(None, 50)</td>
    <td rowspan="1">5050</td>
    <td rowspan="1">Units</td>
    <td rowspan="1">50</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">15</td>
    <td rowspan="1">Dropout</td>
    <td rowspan="1">(None, 50)</td>
    <td rowspan="1">(None, 50)</td>
    <td rowspan="1">0</td>
    <td rowspan="1">Rate</td>
    <td rowspan="1">0.5</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">16</td>
    <td rowspan="1">Dense</td>
    <td rowspan="1">(None, 50)</td>
    <td rowspan="1">(None, 10)</td>
    <td rowspan="1">510</td>
    <td rowspan="1">Units</td>
    <td rowspan="1">10</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
  <tr>
    <td rowspan="1">17</td>
    <td rowspan="1">Dropout</td>
    <td rowspan="1">(None, 10)</td>
    <td rowspan="1">(None, 10)</td>
    <td rowspan="1">0</td>
    <td rowspan="1">Rate</td>
    <td rowspan="1">0.5</td>
  </tr>
  <tr>
   <td rowspan="1"> </td>
  </tr>
   <tr>
    <td rowspan="1">18</td>
    <td rowspan="1">Dense</td>
    <td rowspan="1">(None, 10)</td>
    <td rowspan="1">(None, 1)</td>
    <td rowspan="1">11</td>
    <td rowspan="1">Units</td>
    <td rowspan="1">1</td>
  </tr>
</table>

For the dropout rate of both models, 0.5 was picked after testing the model with different values. When we used rate = 0.7, both the validation loss and the training loss were quite higher compared to the model with rate = 0.5. The results of both models are visualized as follow.


<p align="center">
    <img src=https://user-images.githubusercontent.com/47282229/236675084-76cc67df-489c-4913-bfab-eea995ecc84d.png width="500" height="325">
    <img src=https://user-images.githubusercontent.com/47282229/236675171-6db4c403-ae25-4e33-ac53-9d1f78431906.png width="500" height="325">
 </p>
 


## Discussion - Model
### Evaluation Metric
The main metric used to evaluate the model performance is whether the car crashes while it is driving autonomously. Another metric is the loss calculated as the mean squared error between the actual and the predicted steering angle. MSE was used over MAE to penalize extreme errors because in real-life, a considerably poor prediction will cause a car accident and may lead to loss of life and property.

### Choice of Activation Function
We tested the usage of SoftMax, ReLU and eLU. While eLU and ReLU shared quite similar performance in terms of loss value, the models running with eLU were overfitting while those running with ReLU were not. SoftMax was considerably worse with loss values ranging around 0.3 while that of eLU and ReLU ranging around 0.09 and 0.08 respectively. ReLU was picked as the main activation function to counter overfitting. This resistance to overfitting can be explained by the fact that the ReLU function sets any negative values to zero, which reduces the complexity of the model and prevents the amplification of noise. However, the choice of activation function alone is unlikely to be the sole cause of overfitting; overfitting should be further analyzed in the greater context of the network size and complexity, the number of epochs and training iterations, the choice of hyperparameters, etc.


<p align="center">
    <img src=https://user-images.githubusercontent.com/47282229/233093506-94cd5cf1-9434-441b-a71c-4c8f63923912.png width="500" height="325">
    <img src=https://user-images.githubusercontent.com/47282229/233093616-fcfea842-4574-49a5-9eaf-69ab432a9bc4.png width="500" height="325">
 </p>


### Lighter Model vs Original
The table summarizes the differences between the original and the lighter model. The model was using mini-batch stochastic gradient with batch size = 100.

|        Point of Comparison   | Original Model      | Lighter Model     |
| ---------------------------------- | ------------------------ | ------------------    |
| Number of Parameters     | 342,523                 |  222,395                     |
| Number of Layers    		| 20		      |  18                     |
| Running Time  	| 30 minutes and 50 seconds         |  27 minutes and 55 seconds    |
| Validation Loss (ReLU)  	| 0.0847          |  0.0955    |
| Validation Loss (eLU)  	| 0.0978         |  0.0927   |

With one model having lower loss with ReLU and the other having lower loss with eLU and given that the difference in losses and running time are almost negligible, we can say there is no significant difference between the two models. The lighter model, however, was less prone to overfitting as shown in the graphs.

<p align="center">
    <img src=https://user-images.githubusercontent.com/47282229/233093796-e28124ad-fac9-48de-8440-f6eb9ab38949.png width="500" height="325">
    <img src=https://user-images.githubusercontent.com/47282229/233093971-7196b037-d641-489b-83c2-9856f69cc27d.png width="500" height="325">
 </p>
 
<p align="center">
    <img src=https://user-images.githubusercontent.com/47282229/233094290-6e37a173-748c-49d3-b3a4-053d1e6cbe55.png width="500" height="325">
    <img src=https://user-images.githubusercontent.com/47282229/233094117-f2e511e3-2ac7-4af0-938e-3ab6e1389a5c.png width="500" height="325">
 </p>
 
### Effect of Noise

To quantify the effect of adding noise on model performance, we tested the original model using ReLU activation function and mini-batch of size = 100 with and without adding noise to the training data. Results visualized below show that the augmented data considerably reduced both validation and training loss, especially in the starting epochs. The lowest validation loss of the model with augmented data was recorded in the 23rd epoch at 0.0839, which is lower than the 0.0847 recorded in the 25th epoch as the minimum validation loss of the model without data. This improvement in performance can be explained by the effect of adding noise in smoothing out the data and reducing the impact of outliers. And since the noise is completely randomized, we avoid introducing bias that can negatively impact the performance of the model. However, we notice the model with noise started overfitting and recorded higher validation loss in the 24th and the 25th epochs, which may seem counterintuitive since adding noise primarily aims to reduce overfitting. However, this can be explained by the fact that we added a very high level of noise, so the model started memorizing the noise in the training data, leading to overfitting. This issue will be tackled by reducing noise level.

<p align="center">
    <img src=https://user-images.githubusercontent.com/47282229/233093506-94cd5cf1-9434-441b-a71c-4c8f63923912.png width="500" height="325">
    <img src=https://user-images.githubusercontent.com/47282229/236673310-ffd3a7c4-7ab6-4ae4-ad34-6df7e5421ecb.png width="500" height="320">

 </p>
 
### Stochastic gradient descent vs Mini-batch stochastic gradient descent vs Batch gradient descent
The following table displays the running time. The GPU we are using is the RTX 2060 Ti laptop version with 6 GB vram for reference.

| stochastic gradient descent        | mini-batch ( batch size = 100) stochastic gradient     |
| ------------------------------------------ | --------------------------------------------------------------------    |
| 53 minutes and 40 seconds       |  30 minutes and 50 seconds                                      |

The limited computational capabilities of Aser's local machine made it impossible to run the batch gradient descent as the tensor generated didn’t fit into memory. The largest batch we could use was 1/16 the size of the data, which is 424 images, and the running time of that model was 27 minutes and 38 seconds. 

In terms of overfitting, both the mini-batch stochastic gradient models were not overfitting after 25 epochs while the stochastic gradient descent started overfitting in the fourth epoch. ReLU activation function was used while comparing the models

The following table displays the validation loss values for each of the models. 
| stochastic gradient descent       |  mini-batch ( batch size = 100) stochastic gradient    |  mini-batch ( batch size = 424) stochastic gradient  |
| ------------------------------------------ | ---------------------------------------------------------------    | ---------------------------------------------------------------    |
| 0.0959                                        |  0.0847                                                                    |             0.0862                                                        |

The mini-batch stochastic gradient descent with batch size = 100 achieved the lowest validation loss, had a good running time, and avoided overfitting, so it's preferred over the alternatives.

<p align="center">
    <img src=https://user-images.githubusercontent.com/47282229/233094456-2c409345-fb6f-4880-8df6-b6380985901e.png width="500" height="325">
    <img src=https://user-images.githubusercontent.com/47282229/233094607-22a68b8d-d757-4e22-bfee-061df1a996d8.png width="500" height="325">
 </p>
 
<p align="center">
 <img src=https://user-images.githubusercontent.com/47282229/233095554-dd35aec8-5c16-42f4-89cb-5cca1a28c549.png width="500" height="325">
</p>

### ResNet50, ResNet101, Xception, and MobileNetV2
To evaluate the model performance, we decided to deploy various pre-designed models along with pre-trained weights from Keras Applications. We chose Xception, ResNet50, and ResNet101 as the heavy models and MobileNetV2 as the light model. The validation loss for the four models ranged from 0.3 to 0.32 in the 25 epochs, which implies that either they were stuck in a local minimum or that the models reached their maximum performance. Also, the heavy models took significantly longer compared to the lighter model, which was expected given the depth and the number of parameters. For instance, the Xception model took 168 minutes to complete the 25 epochs and the ResNet 50 took 268 minutes. We were planning for further testing with those models while varying the learning rate to determine whether the model was actually stuck in a local minimum, but the computational power of Aser's local device was an obstacle.

<p align="center">
    <img src=https://user-images.githubusercontent.com/47282229/233204129-0b5e1a7d-1721-4d40-9467-cf1cf44493b8.png width="500" height="325">
    <img src=https://user-images.githubusercontent.com/47282229/233204157-cc4327f6-46af-4368-82db-fee372195a50.png width="500" height="325">
 </p>
 
<p align="center">
 <img src=https://user-images.githubusercontent.com/47282229/233204206-e5d2bb41-4341-498a-bfa9-5547fd908b4f.png width="500" height="325">
</p>


### Model Testing using Simulator
This video displays the car driving autonomously using the original model trained with mini-batch gradient descent and ReLU activation function after augmenting dataset by adding noise.
<div align="center">
  <video src=https://user-images.githubusercontent.com/47282229/236750866-4e0ed38d-1ef4-418d-aef8-e07ea5695667.mp4 width="600" height="400" />
</div> 


## Conclusion - Model
In conclusion, the project successfully utilized a convolutional neural network model to accurately predict the steering angle of a self-driving car, surpassing some of the best pre-trained models, such as ResNet50, ResNet101, and Xception in our initial testing. It's important to point out to the fact that the pre-trained models need to be further tested with different learning rates to ensure the success of our mode. The model was designed with multiple convolutional and fully connected layers, along with appropriate ReLU activation function to enable effective feature extraction and decision-making. The data augmentation techniques, including adding noise and balancing the data, the choice of mini-batch gradient descent, and the use of dropout layers added to the robustness of the model evident by the fact that it was not overfitting even after 25 epochs -- while using ReLU activation function. A significant pitfall worth noting in model testing and training is the computational limitations of the machine, which prevented further testing of the pre-trained models and the use of batch gradient descent. So, further testing is essential as explained in the next section. Overall, the project highlights the potential of using deep learning techniques, such as convolutional neural networks, to make decisions based on image data and being deployed in developing advanced self-driving car systems.

# Ethics Investigation
## Abstract
For the ethics portion of the project, we surveyed existing literature on the ethics of self-driving cars and wrote additional analysis by applying an argument posed by Heather Douglas in “Inductive Risk and Values in Science”. Douglas argues that scientists must include non-epistemic values in decisions throughout the scientific process because steps along the scientific process carry inductive risk, or risk of error, that have non-epistemic consequences. Non-epistemic values are all values that are not included in epistemic values, which are associated with knowledge and truth. Non-epistemic values include values such as moral values, equality, economic prosperity, and wellbeing. This paper applies Douglas’ argument to self-driving cars in scenarios of self-driving car judgment, selection of physical car parts, and elements of the machine learning pipeline. Ultimately, we argue that consideration of non-epistemic values should be included throughout the self-driving car development process because there is high potential for inductive risk that has non-epistemic consequences, such as decline in human well-being. 

## Introduction
The issue of ethics is integral to the development of self-driving cars because of the complexity of the issue and the kinds of stakeholders involved. As Goodall argues, discussing ethics for self-driving cars is relevant because self-driving cars will inevitably crash and be involved in morally ambiguous situations[^6]. However, addressing ethics in self-driving cars is complex because of many different approaches being taken to analyze ethical issues with self-driving cars. In this paper, we will provide a brief literature review on groups of arguments being addressed in self-driving car ethics. Then, we will offer an analysis of self-driving car ethics through the philosophical framework of inductive risk formulated by Carl Hempel and Heather Douglas and argue that value-based design decisions are crucial and relevant to ethics in self-driving cars.

## Literature Review
Early work in self-driving cars focused on convincing people that ethics in self-driving cars was relevant. For example, Goodall’s early work argued that considering ethics was crucial for self-driving car development and addressed criticisms against the “need for ethics research in automated vehicle decisions systems.”[^6] Additionally, Lin argued for the relevance of ethics in self-driving cars, brought up various hypothetical situations, and sought to raise awareness of ethical issues during the “early stage of the technology.”[^7]

As more research into self-driving car ethics developed, researchers took different approaches to examining ethics. Some authors, such as Goodall[^6] and Karnouskos[^8] explored the use of existing moral frameworks to guide the development of self-driving vehicles. Goodall[^6] noted that machine ethics researchers had already discussed the use of moral theories, such as deontology and utilitarianism, as a starting point for “computational moral modeling”, along with their flaws. Similarly, Karnouskos[^8] conducted research using an experimental philosophy approach to determine whether people would accept cars that followed utilitarian, deontological, relativist, absolutist, and pluralistic moral frameworks.

Other authors situated their work within hypothetical situations and thought experiments. Perhaps the most famous of these is the trolley problem. Nyholm and Smids[^9] examined the true relevance of the trolley problem to real-life self-driving cars. They argue that while the trolley problem has general relevance to self-driving cars, it is an imperfect analogy due to differences in instantaneous human decision-making versus cars’ “prospective decisions, or contigency-planning”[^9]. That is, self-driving car developers have time to make decisions for cars well in advance of an event actually occuring. Additionally, self-driving car ethics involve several decision-makers[^9] which the classical trolley problem does not account for. Lin[^7] also discussed the trolley problem along with hypothetical situations involving animals, self-sacrifice, and “no-win” scenarios.

Other authors, such as Holstein et al.[^10] expand the view of ethics in self-driving cars beyond just the direct harm a car may have on life and property to ethics throughout the entire self-driving car process. Namely, they discuss issues of privacy, trust, transparency, reliability, and quality assurance[^10]. They also address the ethics of the social challenges produced by self-driving car deployment, such as job displacement[^10]. Nyholm and Smids[^9] also expand the discussion beyond moral responsibility to legal responsibility.

## Inductive Risk, Non-Epistemic Values, and Machine Learning
In this paper, we will focus on moral responsibility instead of legal and other types of responsibility. Our focus will be on the decision-making process at various points in the design and development stage of a self-driving car. To offer a point of analysis, we will be applying the concepts of inductive risk and non-epistemic values to the issue of self-driving cars. We draw from works by Heather Douglas and Carl Hempel to do so.

In “Inductive Risk and Values in Science,” Heather Douglas[^11] argues for the inclusion of non-epistemic values in science because various decisions in the scientific process carry inductive risk and have non-epistemic effects. To explain some terminology, epistemic values are values that focus on the “attainment of…knowledge” and truth[^12]. Non-epistemic values are all values not included in epistemic values, for example, moral values and “safety, sustainability, equality, nonmaleficence, reliability, economic prosperity and wellbeing”[^12]. Inductive risk is the “chance that one will be wrong in accepting (or rejecting) a scientific hypothesis”[^13].  

Douglas’ argument draws from Hempel’s[^13] work in ‘Science and Human Values”. Hempel[^13] argues that because there is potential error in accepting or rejecting a hypothesis, scientists must weigh the consequences of each outcome. An example of this would be weighing the consequences of having false positives against false negatives in patient disease detection. To weigh these consequences, Hempel argues that values are relevant and thus criticizes a view of value-free science[^11]. Douglas[^11] further expands Hempel’s work and argues that scientists take on inductive risk at multiple points of the scientific process, including choosing “problems to pursue”, methodological choice, such as choosing levels of statistical significance, and scientific reasoning.

Douglas and Hempel’s work has been aptly applied to machine learning models. Karaca[^14] applies the concept of inductive risk to binary classification models. She argues that non-epistemic values and inductive risk are relevant to binary classification models because those models are developed “with respect to the error costs specified in a cost matrix”, the inductive risk one is willing to take has a direct impact on the model’s optimization process[^14]. Thus, developers must consider their end users and the non-epistemic consequences they might face due to errors and quantify the cost of errors[^14]. Sullivan[^15] also argues that “constructing explanations” after a model is built requires consideration of non-epistemic values.

Another issue of relevance to inductive risk and non-epistemic values is opacity in ML models. As Sullivan[^16] argues, deep neural networks, including convolutional neural networks, recurrent neural networks, and multilayer perceptrons, are becoming opaquer to their creators as they grow in complexity. Modelers might not be able to explain how their algorithms work because they constantly evolve or why they produced a certain output[^16]. If we do not know or understand something, it can carry high non-epistemic consequences[^15].

## Inductive Risk, Non-Epistemic Values, and Self-Driving Cars
In the rest of this paper, we will apply the concepts of inductive risk and non-epistemic values to self-driving car ethics. In doing so, we highlight the relevance of considering inductive risk and non-epistemic values in multiple parts of the self-driving car development process. Firstly, it is clear that self-driving cars involve non-epistemic values such as moral values, economic values, legal values, etc. Additionally, all the arguments about inductive risk in the ML pipeline also apply to the use of ML in self-driving cars. To extend these arguments further, we will examine issues of inductive risk and non-epistemic values in self-driving car judgment, selection of physical parts for a car, data collection, and neural network implementation for self-driving cars.

First, we examine inductive risk and non-epistemic values in self-driving car judgment. Goodall[^6] argues that engineers must “teach the elements of good judgment to cars”. While Goodall[^6] does not specify the way that engineers might “teach” judgment to cars, e.g. what kind of models or engineering practices, he does discuss scenarios that engineers have to consider that involve inductive risk. Goodall[^6] argues that a self-driving car cannot always be 100 percent certain that “the road is clear and that crossing the double yellow line is safe.” Thus, engineers must decide at what level of certainty should a car execute a decision[^6]. These decisions will often vary depending on value judgments about the car’s surroundings[^6]. Goodall’s argument displays the relevance of inductive risk in self-driving cars. Because there is no certainty in a self-driving car’s decisions, engineers must choose the points at which they are willing to tolerate risk. This can become more complicated when considering the opaqueness of machine learning models and the lack of control by engineers. This specific case is somewhat similar to Douglas’ argument about choosing levels of statistical significance in an experiment. In determining a numerical threshold, an engineer must also consider the risk of error, especially because errors in self-driving cars have non-epistemic consequences, such as loss in human wellbeing through death and decline in economic prosperity through property damage. Additionally, one example of a non-epistemic judgment a self-driving car might have to make is a value judgment[^6]. For example, a car would have to consider the value of its occupants and its surrounding objects[^6]. A car might also have to weigh the value of different kinds of entities, including humans versus animals[^7].  

Next, we examine issues of inductive risk and non-epistemic values in the selection of physical parts of a self-driving car. For a self-driving car to work, it requires a variety of sensors and physical parts to help detect physical surroundings and gather information[^10]. Holstein et al.[^10] argue that the selection of these parts can have ethical issues. For example, a company could choose to buy a cheaper part at the cost of lower quality, even if the lower quality reduces the car’s ability to make accurate decisions[^10]. This scenario highlights the relevance of non-epistemic values in self-driving cars. Developers must weigh non-epistemic values such as economic prosperity and human wellbeing against each other. Specifically, they are reasoning about the level of inductive risk they are willing to take when choosing a certain part and the types and levels of non-epistemic consequences they might face. Although the public would likely desire engineers to weigh human suffering over economic prosperity, from the perspective of an engineer working at a profit-maximizing company, “the economic aspects might be seen as the highest priority”[^10]. Of course, the decision is not as simple as blanket choosing human life over profit, but rather choosing the point at which large increases in profit outweigh small decreases in human survival or the other way around. Even more specifically, an engineer could be choosing a cheaper part because they know that there is a 100 percent chance of increased economic prosperity (i.e., they are guaranteed that the cheaper part reduces their costs overall and increase profits) and are willing to accept what they believe is a small risk that human wellbeing could decrease.

Finally, we examine issues of inductive risk and non-epistemic values in data collection and elements of the ML pipeline. Various parts of the ML pipeline involve safety issues due to “non-transparency, probabilistic error rate, training-based nature, and instability”[^17]. Rao and Frtunikj[^18] outline that there are three main issues in using deep neural networks for the development of self-driving cars: dataset completeness, neural network implementation, and transfer learning. The first two issues are most relevant to this paper. Firstly, for dataset completeness, Rao and Frtunikj[^18] argue that “it is impossible to ensure a 100 percent coverage of real-world scenes through a single dataset, regardless of its size”, which is especially relevant to deep neural networks for self-driving cars because they are often trained using supervised learning. Thus, the key issue for developers is determining at what point their dataset is “complete” so a model can “[achieve] its maximum possible generalization ability”[^18]. Specifically, including a substantial number of anomalies is crucial[^18]. In the data collection step of the ML pipeline, engineers consider their tolerance of inductive risk when determining the completeness of their data. If they do not consider enough anomalies, they run the risk of their model failing to perform in different scenarios that have non-epistemic consequences, such as reduction in human wellbeing or decline in economic prosperity. Rao and Frtunikj[^18] suggest engineers create synthetic datasets to address the completeness issue, but the issue of inductive risk tolerances still stands. Secondly, Rao and Frtunikj[^18] point out that one cannot isolate different elements of safety to different parts of code in neural networks. A neural network is developed by assembling layers and thus has no direct adherence to any safety goals[^18]. Thus, engineers must determine some other way to ensure that these neural networks produce outputs that minimize the inductive risk associated with extreme non-epistemic values, such as decline in human wellbeing through death. Rao and Frtunijk[^18] suggest various tools for neural network explainability, such as visual representations of convolutional neural networks or the use of “partial specifications” to “prove the plausibility of the network output and to filter out false positives”, to address this issue. The exploration of explainable artificial intelligence is beyond the scope of this paper, but will certainly become important in changing the levels of inductive risk engineers may have to take in the ML pipeline.  

## Personal Opinions
### Aser
Another important ethical consideration in the context of self-driving cars is accountability. In case of accidents, such as the death of Elaine Herzberg by a self-driving Uber car in 2018, how to assign responsibility for the actions of self-driving cars and to hold those responsible accountable for any harm caused?

In terms of inductive risk, it is important to recognize that self-driving cars will never be able to completely eliminate the risk of accidents because they operate in highly complex and uncertain environments. This makes it challenging to predict and account for all potential risks and uncertainties, leading to increased inductive risk. Therefore, there needs to be a focus on developing effective risk mitigation strategies and continuous monitoring and evaluation of the performance of self-driving cars to minimize inductive risk. Also, the inductive risk should be quantified so that manufacturers are held accountable if their cars don't meet a certain threshold for inductive risk. This could involve developing clear regulations and standards for self-driving cars, as well as creating mechanisms for independent testing and certification.
In terms of non-epistemic values, it is important that the accountability of self-driving cars be fair so that no one should be held accountable for an accident that was not their fault. For instance, if a self-driving car causes an accident because of a programming error or a sensor malfunction, it would be unfair to hold the owner of the car accountable. It is also important that the accountability of self-driving cars be just, which means that the consequences of an accident should be proportional to the degree of responsibility for the accident. For example, if a self-driving car causes an accident because the owner was not paying attention, the owner should be held more accountable than if the accident was caused by a programming error.

In order to promote accountability, self-driving cars need to be designed in a way that supports transparency, which can involve providing detailed information about the performance and decision-making of self-driving systems to relevant stakeholders, including passengers, regulatory bodies, and law enforcement agencies. This can help build trust in self-driving cars and increase accountability for their actions. Ultimately, developing new legal frameworks is a pressing demand to address the unique challenges posed by self-driving cars, such as questions of how to assign responsibility for accidents that result from programming errors or sensor malfunctions. With Waymo launching its autonomous ride-hailing services for external users in Phoenix, Los Angeles, and San Francisco this year, these ethical questions need to be addressed and incorporated into legislation sooner than later.

### Michele
It is abundantly clear to me that we need deep consideration of ethics before we deploy self-driving cars on a large scale, especially fully autonomous ones. As self-driving cars are already becoming more popular, we also need additional legal frameworks to regulate their use. However, as the analysis on inductive risk throughout various parts of the self-driving car process highlights, it is not enough to just consider regulating the completed car. There are many steps along the process that might need regulation and measures for accountability. For example, there must be documented decision making and thus accountability for choices such as physical car part selection. In a perfect world, companies could be required to hire ethicists to thoroughly consider and document crucial decisions and assumptions being made. Additionally, there would be swift legal action to promote safety in self-driving cars. The biggest challenge it seems now is that certain parts of the ML pipeline are incredibly opaque. Perhaps new research in explainable AI can help people understand where and how certain decisions in the ML pipeline are being made. 

At the end of the day, as Goodall argues[^8], no one expects self-driving cars to be perfect. However, companies need to be transparent about what assumptions they are making and what decisions they are making when developing their cars so consumers and lawmakers can make informed decisions. Hopefully we can live in a world where self-driving cars are largely safer than human drivers and cause few casualties and injuries. 

## Conclusion - Ethics Investigation
Self-driving car ethics is a complicated and complex area of research. There are different approaches to studying it, including using moral frameworks and investigating different social consequences. One novel approach of analysis is through the lens of inductive risk and non-epistemic values. Ultimately, consideration of inductive risk and non-epistemic values is crucial and relevant to the study of self-driving car ethics.


# Reflection - Final
## Future Development and Recommendations
### Model
For the model, another way to test the it's robustness is using the data from route 1 as the training data and the data from route 2 as the testing data and vice versa to see how the model works in completely new environments. Contingent on available computational resources, the model should be tested with varying batch sizes larger than 1/16 the size of the data. Furthermore, to tune more hyper-parameters, such as the learning rate of the Adam optimizer, the choice of activation function, and the depth and the width of layers, further testing has to be executed while varying those parameters until the best results are achieved in terms of validation loss and over-fitting. While the effect of data augmentation has been quantified by measuring model performance with and without noise, the analysis of data augmentation can’t be considered thorough unless further testing is executed to measure the effect of every specific augmentation and preprocessing function and using different activation functions. To address the limited computational power of Aser’s local device, the class server can be used to speed up model testing, but re-installing dependencies may be inconvenient; however, the Anaconda environment is provided on this GitHub repository. In addition, even though the model is considerably faster than the pre-trained models, such as ResNet50 and Xception, it's considerably slow, taking 30 minutes for training and testing. One way to optimize the model is by using quantization to reduce the number of parameters and increase running speed so that the neural networks can be deployed on embedded systems. The efficiency of the quantization will be measured by comparing the running speed of the original model and the quantized model while keeping a negligible drop in accuracy.

### Ethics Investigation
For the ethics investigation, one way to continue this work is to explore the field of explainable artificial intelligence and consider how that changes the landscape of inductive risk for engineers. Explainable artificial intelligence might help engineers have more control over their models and increase understanding of models, which could have a large impact on risk of error. Next time, we might consider doing more exploratory work into fields of self-driving car ethics that are not specific to moral responsibility.

# References
[^1]: J. Ni, Y. Chen, Y. Chen, J. Zhu, D. Ali, C. Weidong, “A Survey on Theories and Applications for Self-Driving Cars Based on Deep Learning Methods,” in Applied Sciences, vol. 10, no. 8, 2020. [https://doi.org/10.3390/app10082749](https://doi.org/10.3390/app10082749)

[^2]: S. Shalev-Shwartz, S. Shammah, A. Shashua, “On a formal model of safe and scalable self-driving cars,” in ARXIV, Aug. 2017. https://arxiv.org/abs/1708.06374

[^3]: P. Kaur, S. Taghavi, Z. Tian, W. Shi, “A Survey on Simulators for Testing Self-Driving Cars,” 2021. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9499331&tag=1

[^4]: https://github.com/udacity/self-driving-car-sim

[^5]: M.T. Duong, T.D. Do, M.H. Le, “Navigating Self-Driving Vehicles Using Convolutional Neural Network,” in 2018 4th International Conference on Green Technology and Sustainable Development (GTSD), pp. 607-610, 2018. https://ieeexplore.ieee.org/abstract/document/8595533

[^6]: N. J. Goodall, "Can you program ethics into a self-driving car?," in IEEE Spectrum, vol. 53, no. 6, pp. 28-58, June 2016. https://spectrum.ieee.org/can-you-program-ethics-into-a-selfdriving-car

[^7]: P. Lin, "Why Ethics Matters for Autonomous Cars," in Autonomes Fahren, M. Maurer, J. Gerdes, B. Lenz, H. Winner, Springer Vieweg, 2015, pp. 69-85, 2015. https://link.springer.com/chapter/10.1007/978-3-662-45854-9_4

[^8]: S. Karnouskos, "Self-Driving Car Acceptance and the Role of Ethics," in IEEE Transactions on Engineering Management, vol. 67, no. 2, pp. 252-265, May 2020. https://ieeexplore.ieee.org/document/8542947

[^9]: S. Nyholm, J. Smids, "The Ethics of Accident-Algorithms for Self-Driving Cars: an Applied Trolley Problem?," in Ethical Theory and Moral Practice, vol. 19, pp. 1275-1289, Jul. 2016. https://doi.org/10.1007/s10677-016-9745-2

[^10]: T. Holstein, G. Dodig-Crnkovic, P. Pelliccione, "Ethical and Social Aspects of Self-Driving Cars," ARXIV, Jan. 2018. https://arxiv.org/abs/1802.04103

[^11]: H. Douglas, "Inductive Risk and Values in Science," Philosophy of Science, vol. 67, no. 4, pp. 559-579, Dec, 2000. http://www.jstor.org/stable/188707

[^12]: S. Diekmann, M. Peterson, "The Role of Non-Epistemic Values in Engineering Models," Science and Engineering Ethics, vol. 19, no. 1, pp. 207-218, 2013. https://doi.org/10.1007/s11948-011-9300-4

[^13]: C. G. Hempel, "Science and Human Values," in Aspects of Scientific Explanation and Other Essays in the Philosophy of Science, The Free Press, 1965. pp. 81-96

[^14]: K. Karaca, "Values and inductive risk in machine learning modelling: the case of binary classification models," in European Journal for Philosophy of Science,  vol. 11, no. 102, Oct. 2021. https://doi.org/10.1007/s13194-021-00405-1

[^15]: E. Sullivan. "How Values Shape the Machine Learning Opacity Problem," in Scientific Understanding and Representation, I. Lawler, K. Khalifa, E. Shech, Routledge, 2022, pp. 306-322. https://philarchive.org/archive/SULHVS-2

[^16]: E. Sullivan. "Understanding from Machine Learning Models," in The British Journal for the Philosophy of Science, vol. 73, no. 1, pp. 109-133, 2022. https://www.journals.uchicago.edu/doi/full/10.1093/bjps/axz035

[^17]: R. Salay, R. Queiroz, K. Czarnecki, "An analysis of ISO 26262: Using machine learning safely in automotive software," in ARVIX, Sep. 2017. https://arxiv.org/abs/1709.02435

[^18]: Q. Rao and J. Frtunikj, "Deep Learning for Self-Driving Cars: Chances and Challenges," 2018 IEEE/ACM 1st International Workshop on Software Engineering for AI in Autonomous Systems (SEFAIAS), Gothenburg, Sweden, 2018, pp. 35-38. https://ieeexplore.ieee.org/document/8452728

[^19]: M. Podpora, G. P. Korbas, A.  Kawala-Janik, “YUV vs RGB – Choosing a Color Space for Human-Machine Interaction,” 2014 Federated Conference on Computer Science and Information Systems, Warsaw, Poland, 2014, pp 29-34. https://annals-csis.org/Volume_3/pliks/206.pdf

[^20]: K. He, J. Sun, “Convolutional Neural Networks at Constrained Time Cost,” in ARVIX, Dec. 2014. https://arxiv.org/abs/1412.1710
