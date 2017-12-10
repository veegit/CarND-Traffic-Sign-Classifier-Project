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
- [ ] [traffic-sign-data.zip] (data/online-data/traffic-sign-data.zip)
- [ ] [report.pdf] (report.pdf)
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

Model Architecture - The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

3.2 Model Architecture
Used the recommended LeNet architecture with
* Convolution layer 1. The output shape should be 28x28x6.
* Activation 1. - Rectified linear unit (ReLU) Activation
* Pooling layer 1. The output shape should be 14x14x6.
* Convolution layer 2. The output shape should be 10x10x16.
* Activation 2. Rectified linear unit (ReLU) Activation
* Pooling layer 2. The output shape should be 5x5x16.
* Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 
* Fully connected layer 1. This should have 120 outputs.
* Activation 3. Your choice of activation function.
* Fully connected layer 2. This should have 84 outputs.
* Activation 4. Your choice of activation function.
* Fully connected layer 3. This should have 10 outputs.


Model Training - The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

3.3 Model Training

Used the AdamOptimizer which although generally used more computation power than other techniques like GradientDescentOptimizer but provides faster convergence with minimal interventions to modify to learning rate. 

The initial epochs of 20 with no augmented data resulted in 91% on validation set, I changed the epochs size to 50 while retaining the same batch size, this gave me a validation of 93%. I augmented the data and changed the epochs to 40 since it was converging faster now which gave me a final result of 93.7%. 

Solution Approach - The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

3.4 Solution Approach


4. Test a Model on New Images

