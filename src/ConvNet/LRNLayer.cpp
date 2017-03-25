#include "ConvNet.h"

namespace ConvNet {

// a bit experimental layer for now. I think it works but I'm not 100%
// the gradient check is a bit funky. I'll look into this a bit later.
// Local Response Normalization in window, along depths of volumes
LrnLayer::LrnLayer(double k, int n, double alpha, double beta) {
	this->k = k;
	this->n = n;
	this->alpha = alpha;
	this->beta = beta;
	
    // checks
    ASSERT_(n % 2 == 0, "ERROR: n should be odd for LRN layer");
}

void LrnLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	// computed
	output_width = input_width;
	output_height = input_height;
	output_depth = input_depth;
}

Volume& LrnLayer::Forward(Volume& input, bool is_training) {
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

void LrnLayer::Backward() {
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

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void LrnLayer::Store(ValueMap& map) const {
	STOREVAR(out_depth, output_depth);
	STOREVAR(out_sx, output_width);
	STOREVAR(out_sy, output_height);
	STOREVAR(layer_type, GetKey());
	STOREVAR(k, k);
	STOREVAR(n, n);
	STOREVAR(alpha, alpha);
	STOREVAR(beta, beta);
}

void LrnLayer::Load(const ValueMap& map) {
	LOADVAR(output_depth, out_depth);
	LOADVAR(output_width, out_sx);
	LOADVAR(output_height, out_sy);
	LOADVAR(k, k);
	LOADVAR(n, n);
	LOADVAR(alpha, alpha);
	LOADVAR(beta, beta);
}

String LrnLayer::ToString() const {
	return Format("LRN: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
