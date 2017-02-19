#include "Layers.h"


namespace ConvNet {


DropOutLayer::DropOutLayer(double drop_prob) : drop_prob(drop_prob) {
	if (!(drop_prob >= 0.0 && drop_prob <= 1.0))
		throw ArgumentException("DropOutLayer probability is not valid");
	
}

void DropOutLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	// computed
	output_width = input_width;
	output_height = input_height;
	output_depth = input_depth;
	
	dropped.Clear();
	dropped.SetCount(output_width * output_height * output_depth, false);
}

Volume& DropOutLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	Volume& output = output_activation;
	
	int length = input.GetLength();
	
	if (is_training) {
		// do dropout
		for (int i = 0; i < length; i++) {
			if (Randomf() < drop_prob) {
				output.Set(i, 0);
				dropped[i] = true;
			} // drop!
			else {
				dropped[i] = false;
			}
		}
	}
	else {
		// scale the activations during prediction
		for (int i = 0; i < length; i++) {
			// NOTE:
			//  in direct C# version translation: output->Set(i, output->Get(i) * (1 - drop_prob));
			//  but in original JS version was V2.w[i]*=this.drop_prob;
			output.Set(i, output.Get(i) * drop_prob);
		}
	}
	
	return output_activation; // dummy identity function for now
}

void DropOutLayer::Backward() {
	Volume& input = *input_activation; // we need to set dw of this
	Volume& output = output_activation;
	
	int length = input.GetLength();
	input.ZeroGradients(); // zero out gradient wrt data
	
	for (int i = 0; i < length; i++) {
		if (!dropped[i]) {
			input.SetGradient(i, output.GetGradient(i)); // copy over the gradient
		}
	}
}

}
