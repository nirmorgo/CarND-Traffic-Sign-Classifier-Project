# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distribution.jpg "Visualization"
[image2]: ./examples/random_batch.jpg "Random Batch"
[image4]: ./examples/flip_aug.jpg "Flip Augmentation"
[image5]: ./examples/warp_aug.jpg "Warp Augmentation"
[image6]: ./examples/web_images/30kmh.jpg "Traffic Sign 1"
[image7]: ./examples/web_images/highway_entry.jpg "Traffic Sign 2"
[image8]: ./examples/web_images/no_entry.jpg "Traffic Sign 3"
[image9]: ./examples/web_images/no_limit.jpg "Traffic Sign 4"
[image10]: ./examples/web_images/Road_Work.jpg "Traffic Sign 5"
[image11]: ./examples/web_images/stop.jpg "Traffic Sign 6"
[image12]: ./examples/web_images_result.jpg "Web Images Result"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/nirmorgo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 X 32 X 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

We can take a look at a random batch of images from the training set:

![alt text][image2]
These are the road signs that we all know. When looking at a random sample from the training set, we can see that 32X32 image quality is not so fun to look at :) There is also a big variance in the image quality and the lighting conditions, some images are very dark and some are blurry. for example, the 50km/hr sign at the first row of image is almost unrecognizable even for a human. it would be interesting to see how well my classification algorithm deal with that data.

We can also look at the distribution of the classes:

![alt text][image1]

We can see that the classes are not evenly distributed. most speed signs have appear X8-10 times more than other classes such as "dangurous curve to the left" or "end of no passing". While this type of distribution might make our life harder when trying to classify the rare classes correctly, it seems like it is the natural distribution of traffic signs. the big classes are indeed the most common traffic sign that we see as we drive around, and the small classes a re more rare.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For image preprocessing I normalized the images between 0.1 and 0.9
It was enough to encourage faster convergence.
I was also experimenting with other types of preprocessing, but everything else that I have tried have made things worse.


#### Augmentation
In order to train big models, you need lots and lots of data. and i found that the ~35k images in my training set were not enough. in order to create additional data i used 2 techniques:
Flip augmentation - some of the classes have horizontal / vertical symmetry, some classes can change their meaning when flipped (right/left).
![alt text][image4]

Warp augmentation - choose random corners and warp the image
![alt text][image5]
With these two techniques I was able to increase the number of my training samples from ~35k to ~345k images (could have gone for more i had more computation power)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Dropout				|												|
| Max pooling1      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Dropout				|												|
| Max pooling2	      	| 2x2 stride,  outputs 8x8x64 				    |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Dropout				|												|
| Max pooling3	      	| 2x2 stride,  outputs 4x4x128 				    |
| Flatten1 (on pooling1)| outputs 8192   								|
| Dropout				|												|
| Flatten2 (on pooling2)| outputs 4096   								|
| Dropout				|												|
| Flatten3 (on pooling3)| outputs 2048   								|
| concat                | outputs 2048+4096+8192						|
| Fully connected		| outputs 1024        							|
| Fully connected		| outputs 512        							|
| Fully connected		| outputs 43        							|
| Softmax				| 

I use 3 levels of double 3X3 Convolution Relu layers followed by 2X2 max pooling. i concatenate the flattened outputs from all the pooling layers before I'm moving it forward into the fully connected layers. by doing so, I gain two benefits:
* Features from 3 different scales have the same effect on the fully connected layers
* My net got a little deep. The interconnections allow the gradients to propogate more easily into the base layers
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ADAM optimizer with batch sizes of 256, I found that training the model for 40 epochs was usualy enough.

hyper parameters:
* learning rate: 1e-4
* dropout keep ratio: 0.8 on convolutions, 0.5 on fully connected
* lambda for L2 regularizer: 0.01 (applied only on fully connected layers)

I found that aggresive regularization of the fully connected layers was needed in order to avoid over fitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 99.3% 
* test set accuracy of 98.3%

The net architecture and strategy that i chose is highly influenced by what i saw in this [post](https://navoshta.com/traffic-signs-classification/). The author used a multi scale convlutional net. I used a very similar architecture. My main modifications:
* switched all 5X5 convolutions to double consecutive 3X3 convolutions (less parameters and wider perception field)
* add additional fully connected layer and increased number of neurons.
In total, my net is a little heavy, but my GPU didnt have any problem dealing with it.

The main parameters that i had to tune were the learning rate, L2 lambda and dropout ratio. I found that i get the best validation results when i use very agrresive regularization and ~50% dropout on the fully connected layers (although it increases the needed training time).

My choice of architecture and augmentation strategy proved to be a good one when looking at the training results. the training set is classified perfectly and the performance remain stable on the validation and test set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9] 
![alt text][image10]
![alt text][image11]

The images are in different shapes and sizes. before i send them to my classifier, i resize them to 32X32

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction (in the title of every image):
![alt text][image12]


The model was able to correctly guess all of the traffic signs, which gives an accuracy of 100%. Thumbs up!

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 30 km/h  									    | 
| 1.0     				| Right of way in next intersection				|
| .99					| No Entry										|
| 1.0	      			| End of all speed limits					 	|
| 1.0				    | Road Work     							    |
| .99				    | Stop    							            |


We can say that my Classifier was pretty certain about these predictions. :)



