ConvNetC++ is a C++ implementation of neural networks, together with nice native GUI demos. It currently supports:
- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/Softmax) and Regression (L2) **cost functions**
- Ability to specify and train **Convolutional Networks** that process images
- An **Reinforcement Learning** module, based on Deep Q Learning
- Deep **Recurrent Neural Networks** (RNN) 
- **Long Short-Term Memory networks** (LSTM) 
- In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

## [Download latest release (r46)](https://github.com/sppp/ConvNetCpp/releases/download/r46/ConvNetCpp-r46.zip)

# Gallery

Screenshot of [a MNIST digit classification example](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html) translated to ConvNetC++. 
![Classify MNIST example](https://github.com/sppp/ConvNetCpp/raw/master/docs/classifymnist.jpg)

Screenshot of [a CIFAR-10 classification example](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) translated to ConvNetC++. 
![Classify CIFAR-10 example](https://github.com/sppp/ConvNetCpp/raw/master/docs/classifycifar10.jpg)

Screenshot of [a MNIST digit autoencoder example](http://cs.stanford.edu/people/karpathy/convnetjs/demo/autoencoder.html) translated to ConvNetC++. 
![MNIST Autoencoder example](https://github.com/sppp/ConvNetCpp/raw/master/docs/autoencodemnist.jpg)

Screenshot of [a toy 1d regression example](http://cs.stanford.edu/people/karpathy/convnetjs/demo/regression.html) translated to ConvNetC++. 
![a toy 1d regression example](https://github.com/sppp/ConvNetCpp/raw/master/docs/toy1dregression.jpg)

Screenshot of [a toy 2d classification example](http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html) translated to ConvNetC++. 
![Classify2D example](https://github.com/sppp/ConvNetCpp/raw/master/docs/classify2d.jpg)

Screenshot of [Deep Q Learning Demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html) translated to ConvNetC++. 
![Deep Q Learning Demo](https://github.com/sppp/ConvNetCpp/raw/master/docs/deepqlearning.jpg)

Screenshot of [Image Painting Demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html) translated to ConvNetC++. 
![Image Painting Demo](https://github.com/sppp/ConvNetCpp/raw/master/docs/imagepainting.jpg)

Screenshot of [Automatic Prediction Demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/automatic.html) translated to ConvNetC++. 
![Automatic Prediction Demo](https://github.com/sppp/ConvNetCpp/raw/master/docs/optimization.jpg)

Screenshot of [Trainer demo on MNIST](http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html) translated to ConvNetC++. 
![Trainer demo on MNIST](https://github.com/sppp/ConvNetCpp/raw/master/docs/benchmark.jpg)

Screenshot of [GridWorld: Dynamic Programming Demo](http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html) translated to ConvNetC++.
![GridWorld: Dynamic Programming Demo](https://github.com/sppp/ConvNetCpp/raw/master/docs/gridworld.jpg)

Screenshot of [GridWorld: Temporal Difference Learning Demo](http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html) translated to ConvNetC++.
![GridWorld: Temporal Difference Learning Demo](https://github.com/sppp/ConvNetCpp/raw/master/docs/tempdiff.jpg)

Screenshot of [PuckWorld: Deep Q Learning Demo](http://cs.stanford.edu/people/karpathy/reinforcejs/puckworld.html) translated to ConvNetC++.
![PuckWorld: Deep Q Learning Demo](https://github.com/sppp/ConvNetCpp/raw/master/docs/puckworld.jpg)

Screenshot of [WaterWorld: Deep Q Learning Demo](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html) translated to ConvNetC++.
![WaterWorld: Deep Q Learning Demo](https://github.com/sppp/ConvNetCpp/raw/master/docs/waterworld.jpg)

Screenshot of [Deep Recurrent Nets character generation demo](http://cs.stanford.edu/people/karpathy/recurrentjs/) translated to ConvNetC++.
![Deep Recurrent Nets character generation demo](https://github.com/sppp/ConvNetCpp/raw/master/docs/chargen.jpg)

