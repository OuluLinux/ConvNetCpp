#include <ConvNet/ConvNet.h>
using namespace ConvNet;

/*

Original in ConvNetJS:

	var layer_defs = [];
	layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
	layer_defs.push({type:'fc', num_neurons:5, activation:'sigmoid'});
	layer_defs.push({type:'regression', num_neurons:1});
	var net = new convnetjs.Net();
	net.makeLayers(layer_defs);
	 
	var x = new convnetjs.Vol([0.5, -1.3]);
	 
	// train on this datapoint, saying [0.5, -1.3] should map to value 0.7:
	// note that in this case we are passing it a list, because in general
	// we may want to  regress multiple outputs and in this special case we
	// used num_neurons:1 for the regression to only regress one.
	var trainer = new convnetjs.SGDTrainer(net,
	              {learning_rate:0.01, momentum:0.0, batch_size:1, l2_decay:0.001});
	trainer.train(x, [0.7]);
	 
	// evaluate on a datapoint. We will get a 1x1x1 Vol back, so we get the
	// actual output by looking into its 'w' field:
	var predicted_values = net.forward(x);
	console.log('predicted value: ' + predicted_values.w[0]);
*/

CONSOLE_APP_MAIN {
	
	Session ses;
	Net& net = ses.GetNetwork();
	
	ses.AddInputLayer(1, 1, 2);
	
	// Fully connected layer with sigmoid activator
	ses.AddFullyConnLayer(5);
	ses.AddSigmoidLayer();
	
	// Regression layer with fully connected layer
	ses.AddFullyConnLayer(1);
	ses.AddRegressionLayer();
	
	// train on this datapoint, saying [0.5, -1.3] should map to value 0.7:
	// note that in this case we are passing it a list, because in general
	// we may want to  regress multiple outputs and in this special case we
	// used num_neurons:1 for the regression to only regress one.
	SgdTrainer trainer(net);
	trainer.learning_rate = 0.01;
	trainer.momentum = 0.0;
	trainer.batch_size = 1;
	trainer.l2_decay = 0.001;
	ses.SetTrainer(trainer);
	
	Volume vol(1, 1, 2);
	vol.Set(0, +0.5);
	vol.Set(1, -1.3);
	
	// evaluate on a datapoint. We will get a 1x1x1 Vol back, so we get the
	// actual output by looking into its 'w' field:
	trainer.Train(vol, 0, 0.7);
	
	Volume& predicted_values = net.Forward(vol);
	LOG("predicted value: " << predicted_values.Get(0));
	
}
