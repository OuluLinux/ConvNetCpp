#include "Layers.h"


namespace ConvNet {

MaxoutLayer::MaxoutLayer(int group_size) : group_size(group_size) {
	
}

void MaxoutLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	output_depth = (int)floor(input_depth / (double)group_size);
	output_width = input_width;
	output_height = input_height;
	
	switches.SetCount(output_width * output_height * output_depth, 0); // useful for backprop
}

Volume& MaxoutLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	int depth = output_depth;
	output_activation.Init(output_width, output_height, output_depth, 0.0);
	
	// optimization branch. If we're operating on 1D arrays we dont have
	// to worry about keeping track of x,y,d coordinates inside
	// input volumes. In convnets we do :(
	if (output_width == 1 && output_height == 1) {
		for (int i = 0; i < depth; i++) {
			int ix = i * group_size; // base index offset
			double a = input.Get(ix);
			int ai = 0;
			
			for (int j = 1; j < group_size; j++) {
				double a2 = input.Get(ix + j);
				if (a2 > a) {
					a = a2;
					ai = j;
				}
			}
			
			output_activation.Set(i, a);
			switches[i] = ix + ai;
		}
	}
	else {
		int n = 0; // counter for switches
		for (int x = 0; x < input.GetWidth(); x++) {
			for (int y = 0; y < input.GetHeight(); y++) {
				for (int i = 0; i < depth; i++) {
					int ix = i * group_size;
					double a = input.Get(x, y, ix);
					int ai = 0;
					
					for (int j = 1; j < group_size; j++) {
						double a2 = input.Get(x, y, ix + j);
						if (a2 > a) {
							a = a2;
							ai = j;
						}
					}
					
					output_activation.Set(x, y, i, a);
					switches[n] = ix + ai;
					n++;
				}
			}
		}
	}
	
	return output_activation;
}

void MaxoutLayer::Backward() {
	Volume& input = *input_activation; // we need to set dw of this
	Volume& output = output_activation;
	int depth = output_depth;
	
	input.ZeroGradients(); // zero out gradient wrt data
	
	// pass the gradient through the appropriate switch
	if (output_width == 1 && output_height == 1) {
		for (int i = 0; i < depth; i++) {
			double chain_gradient_ = output.GetGradient(i);
			input.SetGradient(switches[i], chain_gradient_);
		}
	}
	else {
		// bleh okay, lets do this the hard way
		int n = 0; // counter for switches
		for (int x = 0; x < output.GetWidth(); x++) {
			for (int y = 0; y < output.GetHeight(); y++) {
				for (int i = 0; i < depth; i++) {
					double chain_gradient_ = output.GetGradient(x, y, i);
					input.SetGradient(x, y, switches[n], chain_gradient_);
					n++;
				}
			}
		}
	}
}

}
