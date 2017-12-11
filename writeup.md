#**Traffic Sign Recognition** 


---

**Problem - Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
1. Attached the required files 
- [ ] [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb)
- [ ] [traffic-sign-data.zip] (writeup/traffic-sign-data.zip)
- [ ] [report.html] (writeup/report.html)
- [ ] [writeup.md] (writeup.md)


2. Dataset Exploration

2.1 Dataset Summary

The following code in numpy extracted some useful information about the data

````
# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_valid = len(X_validation)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes_list,n_classes_first, n_classes_count = np.unique(y_train, return_index=True, return_counts=True)
n_classes = len(n_classes_list)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_valid)
print("Image data shape =", X_test.shape)
print("Number of classes =", n_classes)
````

````
Number of training examples = 34799
Number of testing examples = 12630
Number of validation examples = 4410
Image data shape = (12630, 32, 32, 3)
Number of classes = 43
````

2.2 Exploratory Visualization

Did couple of exploratory visualization of the distribution of data in train, test and validation. This would help in validating if they follow similar distribution to avoid over-fitting and also be consistent.

![2-2-1.png] (writeup/2-2-1.png)

Also did some spot checks to see one image from each class to see what kind of data I am dealing with

![2-2-2.png] (writeup/2-2-2.png)


3. Design and Test a Model Architecture

3.1 Preprocessing

Used some of the conventional techniques like converting to grayscale and then normalizing

````
def normalize(X):
    x_min = X.min(axis=(1, 2), keepdims=True)
    x_max = X.max(axis=(1, 2), keepdims=True)
    X_ = (X - x_min)/(x_max-x_min)
    return X_

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return gray[...,np.newaxis]

def pre_process(X):
    gray = rgb2gray(X)
    return normalize(gray)

````

Also since the data wasn't enough, I improved the scale of the data by rotating and moving the pixels by a random amount. I basically did 2 operation twice on each image in different perumation - rotate(translate(image)) & translate(rotate(image))
Both operations used skimage library to manipulate images

````
def rotate(X):
    X_add = np.empty(shape = (X.shape[0], 32,32,3), dtype = 'float32')
    for i in range(X.shape[0]):
        X_add[i] = transform.rotate(X[i], np.random.randint(-10,10))
    return X_add

def translate(X):
    X_add = np.empty(shape = (X.shape[0], 32,32,3), dtype = 'float32')
    x_trans = np.random.randint(-3,3)
    y_trans = np.random.randint(-3,3)
    for i in range(X.shape[0]):
        X_add[i] = transform.warp(X[i], transform.AffineTransform(translation=(x_trans, y_trans)))
    return X_add

def more_data(X, y):
    X1 = translate(rotate(X))
    X2 = rotate(translate(X))
    X = np.concatenate([X, X1, X2])
    y = np.concatenate([y, y, y])
    return shuffle(X, y)
````

With the conventional techniques I am putting all features on the same scale which improves learning as well as simplifies it by reducing the dimesions which will not necessarily yield better results. Augmenting the data allowed me to converge faster in fewer epochs than necessary. 30 vs 50. 


3.2 Model Architecture
Used the recommended LeNet architecture with

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Convolution layer         		| strides = 1x1, Input = 32x32x1 image, Output = 28x28x6 | 
| Activation     	|  Rectified linear unit (ReLU) Activation	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution layer 	    | strides = 1x1, Input = 14x14x6, Output = 10x10x6  |
| Activation     	|  Rectified linear unit (ReLU) Activation	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten | output = 100   |
| Fully connected		|  Input = 400. Output = 120.        									|
| Activation     	|  Rectified linear unit (ReLU) Activation	|
| Fully connected		|  Input = 120. Output = 84         									|
| Activation     	|  Rectified linear unit (ReLU) Activation	|
|  Fully connected				| Input = 84. Output = 43.        									|

3.3 Model Training

Used the AdamOptimizer which although generally used more computation power than other techniques like GradientDescentOptimizer but provides faster convergence with minimal interventions to modify to learning rate. 

The initial epochs of 20 with no augmented data resulted in 91% on validation set, I changed the epochs size to 50 while retaining the same batch size, this gave me a validation of 93%. I augmented the data and changed the epochs to 40 since it was converging faster now which gave me a final result of 93.7%. 

3.4 Solution Approach
The eventual model yielded following results

````
Model saved in (minutes) =  44.440166942278545
Validation Accuracy = 0.937
Test Accuracy = 0.916
````

The model was based as mentioned earlier on LeNet architecture. It is based on the ever popular Convolutional Neural Networks which does an excellent job in image classification. LeNet is one of the simple implementation of CNN and made sense to use it first. Since CNN does the feature extraction for us already, it was no-brainer to use it for a finite set of pixel matrices. Other alternatives could have been ResNet or AlexNet implementation of CNN. 

Once the architecture was setup, I went with a default iteration of 10,15, 20 Epochs. They were converging close to 91%. Once I increased the Epochs to 50, I found 93% validation rate which would have been enough for this assignment. I tried to augment the data further to reduce the epochs, the image augmentation I did initially was actually not performing well. THat was due to a bug in the code which caused augmented image to be wiped out completely from PIL ilbrary. I used the skimage library which was performant and gave me good augmented image. This excercise allowed me to reduce epochs and have faster convergence. 

I didn't see any instance of overfitting but I still tried to add a dropout regularization before the final fully connected layer. I didn't see any major performance improvement. 

4. Test a Model on New Images
4.1 Acquiring New Images - Downloaded the following 5 images and cropped/resized them to 32x32

![4-1-1.png] (writeup/4-1-1.png)

The images were downloaded based on the background, orientation, light effects and some additional noise in the sign itself.

4.2 Performance on New Images - The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

The model correctly predicted 6 out of 7 images which resulted in an accuracy of 85.71%. This is in contrast to accuracy on validation set of 93.7%

4.3 Model Certainty - Softmax Probabilities The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

These are the softmax probablities of images and their respective classes
Softmax
````
[
       [  5.02068863e+01,   2.50825157e+01,   1.85059738e+01, 8.66750336e+00,   2.37566519e+00],
       [  6.41252060e+01,   5.09600334e+01,   1.85903034e+01, 1.69791813e+01,   9.98717499e+00],
       [  4.63154068e+01,   3.62617850e+00,   1.33984554e+00, -4.73022223e-01,  -2.24502993e+00],
       [  1.33870726e+01,   8.82190990e+00,   7.61714554e+00, 5.78275621e-02,  -5.13552046e+00],
       [  3.18689346e+01,   2.00786018e+01,  -3.91756743e-02, -2.41427922e+00,  -3.69053125e+00],
       [  1.19942913e+01,   4.44106436e+00,   2.56481075e+00, -3.46791601e+00,  -8.00393200e+00],
       [  5.44795227e+01,   1.69301682e+01,   5.03529644e+00, 1.74080563e+00,  -2.98584270e+00]
]
````

Classes


````
[
       [31, 23, 37, 19, 10],
       [17, 41, 14, 28, 33],
       [38, 25,  1, 26, 22],
       [12, 40, 41, 17, 14],
       [ 1,  5, 25,  2, 33],
       [11, 30, 28,  6, 35]
]
````


![Figure_1-0.png] (writeup/Figure_1-0.png)

![Figure_1-1.png] (writeup/Figure_1-1.png)

![Figure_1-2.png] (writeup/Figure_1-2.png)

![Figure_1-3.png] (writeup/Figure_1-3.png)

![Figure_1-4.png] (writeup/Figure_1-4.png)

![Figure_1-5.png] (writeup/Figure_1-5.png)

![Figure_1-6.png] (writeup/Figure_1-6.png)


The model correctly predicted 6 out of 7 images. The image where it failed was a sign with additional information at the bottom. 

![5_18.png] (data/online/5_18.png)
