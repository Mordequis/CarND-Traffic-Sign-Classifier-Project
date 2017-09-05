#**Traffic Sign Recognition** 
The goals / steps of this project are :
* Load, explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results here in this report

---
###1. Data Set Summary & Exploration

- ####I used Python and numpy  to output the following values based on the input data sets:


```
Number of training examples = 34799
Number of testing examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of unique classes/labels = 43
```

- ####TODO Include an exploratory visualization of the dataset.


Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

---
###2. Design and Test a Model Architecture

- ####Prior to training, the image data is normalized and undergoes no other preprocessing.

  - The code supports conversion to grayscale by changing n_color_channels to 1 (instead of 3) but in testing, grayscale conversion reduced the model accuracy.  The following is an example of how a sign would be converted to grayscale if enabled:

    ![alt text][image2]

  - Approximate normalization is done using the given quick formula of: (pixel - 128)/ 128



- ####The final model ("SignNet") consists of a six layer convolutional neural network. Details are shown in the following chart:

  |      Layer      |               Description                |
  | :-------------: | :--------------------------------------: |
  |      Input      |            32x32x3 RGB image             |
  | Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x26 |
  |      RELU       |                                          |
  |   Max pooling   |      2x2 stride,  outputs 14x14x26       |
  | Convolution 3x3 | 1x1 stride, valid padding, outputs 12x12x48 |
  | Convolution 1x1 | 1x1 stride, valid padding, outputs 12x12x69 |
  |      RELU       |                                          |
  |   Max pooling   |       2x2 stride,  outputs 6x6x69        |
  |  Dropout (40%)  |                                          |
  | Fully connected |    (6x6x69 = 2484 input) outputs 541     |
  |      RELU       |                                          |
  | Fully connected |               outputs 367                |
  |      RELU       |                                          |
  |  Dropout (40%)  |                                          |
  | Fully connected |                outputs 43                |
  |     Softmax     |                                          |



- ####The model is trained using the gradient descent Adaptive Moment Estimation (Adam) optimizer to minimize the loss (how far the calculated values are from the truth labels).

  - The optimizer (Adam), the batch size (128), number of epochs (10) and learning rate (0.001) were initially chosen using what was used in the LeNet lab.  Batch size and number of epochs were adjusted and then reverted to the original values because they seemed to work best.

- ####Design Approach:

  #####1.  LeNet With Dropout

    - I started with the LeNet model from lecture adapted to handle three color channels and 43 classes instead of 10. **TODO  Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.**

    - I evaluated the accuracy of the model on the test set and validation set for each epoch.  The test data accuracy was well above 0.93 but the validation accuracy was always below.

    - I added dropout (50%) to LeNet because it looked like the model might be over fitting. Training LeNetWithDropout with grayscale data, the validation accuracy was only sometimes above 0.93.  Doing the same thing with color data (no grayscale conversion during preprocessing) the validation accuracy was reliably a little above 0.93.

  |      Layer      |               Description                |
  | :-------------: | :--------------------------------------: |
  |      Input      |            32x32x3 RGB image             |
  | Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
  |      RELU       |                                          |
  |   Max pooling   |       2x2 stride,  outputs 14x14x6       |
  | Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
  |      RELU       |                                          |
  |   Max pooling   |       2x2 stride,  outputs 5x5x16        |
  |  Dropout (50%)  |                                          |
  | Fully connected |     (5x5x16 = 400 input) outputs 120     |
  |      RELU       |                                          |
  | Fully connected |                outputs 84                |
  |      RELU       |                                          |
  |  Dropout (50%)  |                                          |
  | Fully connected |                outputs 43                |
  |     Softmax     |                                          |

  #####2. SignNet Version 1

  - While LetNetWithDropout may have met the 0.93 minimum requirement, I thought I could do better if I experimented with different number of filters.

  - The first iteration of "SignNet" was LeNet with more parameters/filter depth and yielded slightly higher validation accuracy when trained for 15-20 epochs.  LeNet may work well for identifying simple 0-9 digits but perhaps it didn't have enough neurons to properly classify 43 different traffic signs.  I increased the depth of convolutional layers arbitrarily by 4.3 (43 classes/10 classes) to maintain the same ratio of the model to the output classes which yielded 1725 neurons before the fully connected layers.  From there I worked from the end backwards, again trying to maintain the same ratios as LeNet (10->84->120 = 43->367->541).  Note I picked the next largest prime numbers for fun because I thought it might enforce some degree of independence between the neurons.  Increases are shown below in bold:

  |      Layer      |               Description                |
  | :-------------: | :--------------------------------------: |
  |      Input      |            32x32x3 RGB image             |
  | Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x**26** |
  |      RELU       |                                          |
  |   Max pooling   |    2x2 stride,  outputs 14x14x**26**     |
  | Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x**69** |
  |      RELU       |                                          |
  |   Max pooling   |     2x2 stride,  outputs 5x5x**16**      |
  |  Dropout (50%)  |                                          |
  | Fully connected | (5x5x**69** = **1725** input) outputs **541** |
  |      RELU       |                                          |
  | Fully connected |             outputs **367**              |
  |      RELU       |                                          |
  |  Dropout (50%)  |                                          |
  | Fully connected |                outputs 43                |
  |     Softmax     |                                          |

  #####3. SignNet Version 2

  - Once again, while SignNet V1 may have met the 0.93 minimum requirement, I thought I could do even better if I experimented with more layers or convolution shape.
  - GoogLeNet (https://arxiv.org/pdf/1409.4842v1.pdf) inspired me to try 40% dropout rate (which improved validation accuracy), average pooling (no noticeable change so I went back to max pooling).
  - The lectures and GoogLeNet inspired me to try adding a 1x1 convolution in conjunction with a 3x3 or 5x5 convolution for more depth in the neural network.  The LeNet layer two 5x5 convolution was shrinking the number of parameters too quickly for my taste so I switched to a 3x3.  Putting a 1x1 convolution before or after the 3x3 seemed to have the same beneficial effect.  I opted to put the 1x1 after the 3x3 like in the lecture ignoring Google's use before a 3x3 or 5x5 for compute reduction purposes.
  - I started the 3x3 layer with a depth of 51 and the 1x1 with a depth of 95.  I lowered those number to 48 and 69 respectively (as shown in the architecture table shown in the "final model" bullet above) to reduce the number of overall parameters without any noticeable impact to the validation accuracy.

- ####Final Results

    - Training and validation accuracy were calculated after each epoch of training.  Test set accuracy was calculated once after the model achieved the desired training and validation accuracy levels.

      | Set Name   | Accuracy |
      | ---------- | -------- |
      | Training   | 0.998    |
      | Validation | 0.973    |
      | Test       | 0.965    |

    - One or more convolution layers may work well with identifying traffic signs because we want to identify the signs independent of spatial variation.  As cars drive around, the viewing angle and rotation of the signs will vary.  Generally speaking signs will be right side up but sometimes signs find themselves tilted.  On straight roads at a far distance, the sign face will approximately be perpendicular to the line of sight but as a road curves, elevation changes and a vehicle moves past traffic signs, the viewing angle changes.

    - One or more dropout layers may help create redundant activations for certain desired features and help prevent over fitting.  By having a more robust model, the prediction accuracy should improve assuming the network is sufficiently large.

    - The model appears to work well because once trained on one data set, the accuracy remains high when the model is tested against validation and test sets as well as new images.

---
###3. Test a Model on New Images

- ####The following five German traffic signs were captured and formatted from the following two videos:

  - https://www.videoblocks.com/video/timelapse-of-driving-on-the-highway-in-germany-cemcuka/

  - https://www.youtube.com/watch?v=J7AcbtDCAm0

  ​

  1. ![alt text][image4] The extra sign on the pole below the general caution sign may be abnormal and not represented in the training data.
  2. ![alt text][image5]The roof top in the background might be different from sky, trees or grass behind priority road signs in the training data.  The sign also isn't very bright so the difference between the white and yellow of the sign might pose a challenge.
  3. ![alt text][image6] This yield sign is shaded on a bright day which may be unclear to the trained model.
  4. ![alt text][image7] Two things that could make this "Right-of-way at the next intersection" sign hard to classify are the low resolution and the lower left corner being slightly obscured by a vehicle.
  5. ![alt text][image8]This 70 km/h speed limit is partially shaded which could make the 70 difficult to match.

- ####Prediction results:

  - Results vary each time the model is trained, tested and then tested on this new set of images.  Signs 1-3 were generally always recognized with high  probability (99.999%-100%). Signs 4 and 5 were hit or miss.  Sometimes they were correctly predicted with high (90+%) probability.  Sometimes the correct prediction had a low probability (30+%) and other times the predictions was wrong.
  - Overall the model had general accuracy of 1.00 or 0.80 but rarely had 0.60 when run on the new images.  The first two accuracy values compare favorable with the 0.965 test set accuracy.
  - Softmax probabilities are shown below for each sign.  The correct sign is in bold.

  1. | Probability | Prediction            |
     | ----------- | --------------------- |
     | 0.999990    | **General Caution**   |
     | 0.000003    | Speed limit (30km/h)  |
     | 0.000003    | Speed limit (20km/h)  |
     | 0.000001    | Wild animals crossing |
     | 0.000001    | Road work             |

  2. | Probability      | Prediction                               |
     | ---------------- | ---------------------------------------- |
     | 1.0              | **Priority road**                        |
     | 0.00000000000067 | No passing for vehicles over 3.5 metric tons |
     | 0.00000000000015 | Traffic signals                          |
     | 0.00000000000007 | Road work                                |
     | 0.00000000000003 | No entry                                 |

  3. | Probability | Prediction           |
     | ----------- | -------------------- |
     | 0.9999574   | **Yield**            |
     | 0.0000290   | Speed limit (60km/h) |
     | 0.0000038   | Keep right           |
     | 0.0000020   | End of no passing    |
     | 0.0000019   | No passing           |

  4. | Probability | Prediction                               |
     | ----------- | ---------------------------------------- |
     | 0.31        | **Right-of-way at the next intersection** |
     | 0.24        | Double curve                             |
     | 0.21        | Road work                                |
     | 0.13        | General caution                          |
     | 0.07        | Speed limit (30km/h)                     |
       Right-of-way at the next intersection, Double Curve, Road Work and General Caution all are triangular with red border. The model may be trained well to recognize that but still struggle to distinguish between the sign's internal symbol characteristics.

  5. | Probability | Prediction               |
     | ----------- | ------------------------ |
     | 0.90360     | **Speed limit (70km/h)** |
     | 0.07193     | Speed limit (20km/h)     |
     | 0.02415     | Speed limit (30km/h)     |
     | 0.00018     | Speed limit (120km/h)    |
     | 0.00008     | Speed limit (100km/h)    |
     All speed limit signs are round with a red border so the model correctly predicts the new image is a speed limit sign (top 5 results are all speed limit signs).  The model however, struggled to identify which siRight-of-way at the next intersection, Double Curve, Road Work and General Caution all are triangular with red border. The model may be trained well to recognize that but still struggle to distinguish between the sign's internal symbol characteristics particularly with the new image having partial shade over the numbers.

###4. Visualizing the Neural Network
- ####Since new Image 4 ![alt text][image7] Right-of-way at the Next Intersection seemed challenging to my model, I used Step 4 to visualize the model's feature map after the second convolution:

  ####![alt text][image9]

  As you can see, the model seems to clearly find the triangular shape interesting.



[//]: #	"Image References"
[image1]: ./examples/visualization.jpg	"Data Set Visualization"
[image2]: ./writeup_images/grayscale.png	"Grayscaling"
[image3]: ./examples/random_noise.jpg	"Random Noise"
[image4]: ./new_images/web1.png	"Traffic Sign 1"
[image5]: ./new_images/web2.png	"Traffic Sign 2"
[image6]: ./new_images/web3.png	"Traffic Sign 3"
[image7]: ./new_images/web4.png	"Traffic Sign 4"
[image8]: ./new_images/web5.png	"Traffic Sign 5"
[image9]: ./writeup_images/rightofway_visualization.png	"Right-of-Way Visualization"