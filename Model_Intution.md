# Model implementation and Intution Documentation

## Table of Contents
- [Model implementation and Intution Documentation](#model-implementation-and-intution-documentation)
  - [Table of Contents](#table-of-contents)
  - [Data set Generation](#data-set-generation)
    - [Data set handling:](#data-set-handling)
  - [Preprocessing](#preprocessing)
  - [Squeezenet model](#squeezenet-model)
  - [Custom Model 0](#custom-model-0)
  - [Custom Model 1](#custom-model-1)
  - [Hpyerparameters and Obsevations](#hpyerparameters-and-obsevations)

## Data set Generation
Using Opencv library, frames are captured and stored in a new directory named "captured_frames".
Upon starting the capture for the first time after executing the program, the program considers all the upcoming frames as Class 0 and on capturing the 2nd time it creates a new directory i.e. "captured_frames/class1" and so on which contains frames corresponding to the directory(Class).
The dataset generated is stored in the same directory as "dataset.npz" file.

### Data set handling:
If the data set already exists the program would throw a message box asking to replace data or to concatenate with the existing data set.

## Preprocessing 
The dataset is stored by preprocessing the images i.e. for custom model the images are converted to size 64x64 which is further flatten to a 1-D vector while for squeezenet implementation image is converted to 299x299 pixel size.

## Squeezenet model
Implemented Squeezenet model as it is optimized to embeded devices by decreasing the computation cost by introducing Fire modules. The details of the model can be found [here](https://github.com/forresti/SqueezeNet)
The model upon comparison with iceptionresnet model takes 50% less memory ussage but found the accuracy to be lower by 6-7%.

## Custom Model 0
The custom Model 0 is a naieve neural network architecture with minimum layers. The proposed model consists of 3 layers as follows:
1. Input layer with 12288 neurons. (64x64x3 = 12288).
2. 1st layer containing 64 neurons with "Relu" activation. Main intution behind the layer is to map the contribuiton of each row/column containing all channels of the input image.
3. 2nd layer containing 3 neurons with "Relu" activation which is suposed to extract sufficient information for binary classification between 2 images by considering 3 parameters which will be trained by the model.
4. Output layer containing "Softmax" activation for multiclass classificaiton.
   
Note: the 2nd layer is considered for binary classification with sigmoid output layer but upon modifying later can used for multi class classification where the number of classes are low.

## Custom Model 1
The custom Model 1 is a deep neural network model with 5 layers. We cannot make a deeper model as we have to consider computational cost and Training time as well.

1.Input layer with 12288 neurons. (64x64x3 = 12288).
2. 1st layer containing 4096 neurons with "Relu" activation. Main intution behind the layer is to find the contribution of each pixel by combining all of their channels.
3. 2nd layer containing 256 neurons. The intution was that this layer will try to find out the lines, curves or the edges in the image.
4. 3rd layer containing 64 neurons i.e. the max length in vertical/horizontal dimension. This layer is expected to find some shapes in image by using the edges or curve detected in previous layer or consider the image made of certain shapes.
5. 4th layer containing 16 neurons are supposed to look for objects appearing in the images by making use of the shapes observed earlier.
   
NOTE: Some additional layers can be added which would look to for certain objects as whole i.e. big and complete objects but would increase the complexity of the model

6. Output layer consisting of "num_classes" number of classes for multi-class classification through softmax acttivation.

## Hpyerparameters and Obsevations

The hyperparameter like epochs should not be set quite high as it is introducing the problem of variance. 
Model 1 and Squeezenet takes more ram compared to model 0 Thus the training time is not constant due to available resources at that time. 

For lite uses Custom model 0 works the best but the 2 class should have significat enough change for this small model to detect. Minor changes in environment between the 2 classes causes the problem of Bias and affects performance significantly.

A better model would be Model 1 but for high accuracy purposes, Squeezenet model can be better in some cases. 