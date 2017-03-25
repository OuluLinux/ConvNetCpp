# ConvNetC++
ConvNetC++ is a C++ port of [ConvNetJS](https://github.com/karpathy/convnetjs) and [ConvNetSharp](https://github.com/cbovar/ConvNetSharp). [RecurrentJS](https://github.com/karpathy/recurrentjs) and 
[ReinforceJS](https://github.com/karpathy/reinforcejs) are also included in the port.

It currently supports:

- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/Softmax) and Regression (L2) **cost functions**
- Ability to specify and train **Convolutional Networks** that process images
- An **Reinforcement Learning** module, based on Deep Q Learning
- Deep **Recurrent Neural Networks** (RNN) 
- **Long Short-Term Memory networks** (LSTM) 
- In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

## Example code

For screenshots of examples, see [the gallery](https://github.com/sppp/ConvNetC-/blob/master/GALLERY.md).

ConvNetC++ includes all original examples from ConvNetJS and also examples from RecurrentJS and ReinforceJS, even tough they were separate libraries originally.

A typical usage might look something like:
```c++
// species a 2-layer neural network with one hidden layer of 20 neurons
Net net;

// input layer declares size of input. here: 2-D data
// ConvNet U++ works on 3-Dimensional volumes (width, height, depth), but if you're not dealing with images
// then the first two dimensions (width, height) will always be kept at size 1
InputLayer input(1, 1, 2);
net.AddLayer(input);

// declare 20 neurons
FullyConnLayer fullcon1(20);
fullcon1.bias_pref = 0.1;
net.AddLayer(fullcon1);

// declare a ReLU (rectified linear unit non-linearity)
ReluLayer relu1;
net.AddLayer(relu1);

// declare 20 neurons
FullyConnLayer fullcon2(20);
fullcon2.bias_pref = 0.1;
net.AddLayer(fullcon2);

// declare a ReLU (rectified linear unit non-linearity)
ReluLayer relu2;
net.AddLayer(relu2);

// declare a fully connected layer that will be used by the softmax layer
FullyConnLayer fullcon3(2);
net.AddLayer(fullcon3);

// a softmax classifier predicting probabilities for two classes: 0,1
SoftmaxLayer softmax(2);
net.AddLayer(softmax);

// forward a random data point through the network
Volume x(1, 1, 2, 0);
x.Set(0, +0.5);
x.Set(1, -1.3);
Volume& probability_volume = net.Forward(x);

// prob is a Volume. Volumes have a property Weights that stores the raw data, and WeightGradients that stores gradients
LOG("probability that x is class 0: " << probability_volume.GetWeights()[0]); // prints 0.50101

SgdTrainer trainer(net);
trainer.SetLearningRate(0.01).SetL2Decay(0.001);
trainer.Train(x, 0, 0);

Volume& probability_volume2 = net.Forward(x);
LOG("probability that x is class 0: " << probability_volume2.GetWeights()[0]);
// prints 0.50374
```

## Compiling the library and examples
ConvNetC++ requires the cross-platform library [Ultimate++](https://sourceforge.net/projects/upp/files/upp/2015.2/), which works in all platforms (Windows, Linux, OSX, FreeBSD). Even Windows XP is 
supported, because the U++ version 9251 and Windows 7 SDK are the minimum requirements. Getting this to work in OSX is probably easier with wine, than with native solution, which is incomplete.

After you have installed the Ultimate++, create a new assembly for the ConvNetC++ by looking included assemblies as examples.
You can compile examples with the included MINGW compiler, but compiling them with the Visual Studio compiler makes it a lot faster.
