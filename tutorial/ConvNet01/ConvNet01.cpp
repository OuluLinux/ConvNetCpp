#include "ConvNet01.h"

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
	net.AddLayer(fullcon1);
	
	// declare a ReLU (rectified linear unit non-linearity)
	ReluLayer relu;
	net.AddLayer(relu);
	
	// declare a fully connected layer that will be used by the softmax layer
	FullyConnLayer fullcon2(10);
	net.AddLayer(fullcon2);
	
	// declare the linear classifier on top of the previous hidden layer
	SoftmaxLayer softmax(10);
	net.AddLayer(softmax);
	
	// forward a random data point through the network
	Vector<double> weights1;
	weights1.Add(+1.0);	weights1.Add(-1.0);
	Volume x1(weights1);
	
	Vector<double> weights2;
	weights2.Add(-1.0);	weights2.Add(+1.0);
	Volume x2(weights2);
	
	Volume& prob1 = net.Forward(x1);
	
	// prob is a Volume. Volumes have a property Weights that stores the raw data, and WeightGradients that stores gradients
	DUMPC(prob1.GetWeights());
	LOG("probability that x1 is class 0: " << prob1.GetWeights()[0]);
	
	SgdTrainer trainer(net);
	trainer.learning_rate = 0.01;
	trainer.l2_decay = 0.001;
	for (int i = 0; i < 100; i++) {
		trainer.Train(x1, 0); // train the network, specifying that x1 is class zero
		trainer.Train(x2, 1); // train the network, specifying that x2 is class one
	}
	
	Volume& prob2 = net.Forward(x1);
	DUMPC(prob2.GetWeights());
	LOG("probability that x1 is class 0: " << prob2.GetWeights()[0]);
	
	Volume& prob3 = net.Forward(x2);
	DUMPC(prob3.GetWeights());
	LOG("probability that x2 is class 1: " << prob3.GetWeights()[1]);
	
}
