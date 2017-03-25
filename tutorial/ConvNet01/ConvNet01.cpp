#include "ConvNet01.h"

/*

Original in ConvNetJS:

	var layer_defs = [];
	// input layer of size 1x1x2 (all volumes are 3D)
	layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
	// some fully connected layers
	layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
	layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
	// a softmax classifier predicting probabilities for two classes: 0,1
	layer_defs.push({type:'softmax', num_classes:2});
	 
	// create a net out of it
	var net = new convnetjs.Net();
	net.makeLayers(layer_defs);
	 
	// the network always works on Vol() elements. These are essentially
	// simple wrappers around lists, but also contain gradients and dimensions
	// line below will create a 1x1x2 volume and fill it with 0.5 and -1.3
	var x = new convnetjs.Vol([0.5, -1.3]);
	 
	var probability_volume = net.forward(x);
	console.log('probability that x is class 0: ' + probability_volume.w[0]);
	// prints 0.50101

*/

CONSOLE_APP_MAIN {
	
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
	
	
	/*
	Note from translator:
	Actually weights in layers are randomized, and values are not 0.50101 and 0.50374,
	but the second should be higher value than first.
	*/
	
}
