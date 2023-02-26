# TNAP
A tri-node art predictor.

# TNAP - What Does It Do?
TNAP was designed on a fairly simple premise: Predicting whether artwork was AI-generated or manmade through 3 unique machine learning models. Data was used to train 3 separate models, each with their own strengths and weaknesses. Users can pass images into these models and they will collectively decide upon one answer: **AI or not**. We intensively studied the behaviors these models during training and testing and decided on decision weights that would most accurately represent the actual state of the test image.

# How To Use - GUI
To use the GUI, click either "open file" or "open folder." Select either file(s) or a folder of files. Once selected, the display will list each of the images, filename, and AI/manmade status as well as confidence level. When files are done processing, a CSV file of the results can be downloaded.

# Info For Nerds
The three nodes of our predictor are a random forest classifier, a convolution neural network, and a k-neighbors classifier.

Random Forest Classifier Stats:
  -Sklearn
  -90/10 data split
  -1000 trees
  -Max Depth of 32
  -Entropy criterion
  
Convolution Neural Network Stats:
  -Pytorch
  -90/10 data split
  -4 convolution layers with pooling
  -2 linear layers
  -SGD optimizer
  
 K-Neighbors Classifier Stats:
  -Sklearn
  -90/10 data split
  -3 neighbors
  -Uniform weights
  -Manhattan distance (L1)
  
# Future Plans
-Improving accuracy of models
  -Training on more data
  -Trying other model architectures
  -Trying new models
-Web app?
-Integration with further AI tools
