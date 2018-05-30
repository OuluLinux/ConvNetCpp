#include "ConvNet.h"

namespace ConvNet {


void LayerBase::InitLrn(int input_width, int input_height, int input_depth) {
	
	// computed
	output_width = input_width;
	output_height = input_height;
	output_depth = input_depth;
}

Volume& LayerBase::ForwardLrn(Volume& input, bool is_training) {
	input_activation = &input;
	
	output_activation.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0);
	S_cache.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0);
	
	int n2 = n / 2;
	
	for (int x = 0; x < input.GetWidth(); x++) {
		for (int y = 0; y < input.GetHeight(); y++) {
			for (int i = 0; i < input.GetDepth(); i++) {
				double ai = input.Get(x, y, i);
				
				// normalize in a window of size n
				double den = 0.0;
				for (int j = max(0, i - n2); j <= min(i + n2, input.GetDepth() - 1); j++) {
					double aa = input.Get(x,y,j);
					den += aa * aa;
				}
				den *= alpha / n;
				den += k;
				S_cache.Set(x, y, i, den); // will be useful for backprop
				den = pow(den, beta);
				output_activation.Set(x, y, i, ai/den);
			}
		}
	}
	
	return output_activation;
}

void LayerBase::BackwardLrn() {
	// evaluate gradient wrt data
	Volume& input = *input_activation; // we need to set dw of this
	input.SetConstGradient(0); // zero out gradient wrt data
	Volume& output = output_activation; // computed in forward pass
	
	int n2 = n / 2;
	
	for (int x = 0; x < input.GetWidth(); x++) {
		for (int y = 0; y < input.GetHeight(); y++) {
			for (int i = 0; i < input.GetDepth(); i++) {
				double chain_grad = output.GetGradient(x,y,i);
				double S = S_cache.Get(x, y, i);
				double SB = pow(S, beta);
				double SB2 = SB*SB;
				
				// normalize in a window of size n
				int begin = max(0, i - n2);
				int end = min(i + n2, input.GetDepth() - 1);
				for (int j = begin; j <= end; j++) {
					double aj = input.Get(x, y, j);
					double g = -aj * beta * pow(S, beta - 1) * alpha / n * 2 * aj;
					if (j == i)
						g += SB;
					g /= SB2;
					g *= chain_grad;
					input.AddGradient(x, y, j, g);
				}
			}
		}
	}
}

String LayerBase::ToStringLrn() const {
	return Format("LRN: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
