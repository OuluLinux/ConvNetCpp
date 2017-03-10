#include "Layers.h"


namespace ConvNet {


PoolLayer::PoolLayer(int width, int height) {
	stride = 2;
	pad = 0;
	
	this->width = width;
	this->height = height;
}

void PoolLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	UpdateOutputSize();
}

void PoolLayer::UpdateOutputSize() {
	// computed
	output_depth = input_depth;
	output_width = (int)floor((input_width + pad * 2 - width) / (double)stride + 1);
	output_height = (int)floor((input_height + pad * 2 - height) / (double)stride + 1);
	
	// store switches for x,y coordinates for where the max comes from, for each output neuron
	switchx.SetCount(output_width * output_height * output_depth, 0);
	switchy.SetCount(output_width * output_height * output_depth, 0);
}

Volume& PoolLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	
	output_activation.Init(output_width, output_height, output_depth, 0.0);
	
	int n = 0; // a counter for switches
	for (int depth = 0; depth < output_depth; depth++)
	{
		// WTF C# version int n = depth * output_width * output_height;
		
		double x = -1.0 * pad;
		for (int ax = 0; ax < output_width; x += stride, ax++) {
			double y = -1.0 * pad;
			for (int ay = 0; ay < output_height; y += stride, ay++) {
				// convolve centered at this particular location
				double a = -DBL_MAX;
				int winx = -1, winy = -1;
				
				for (int fx = 0; fx < width; fx++) {
					for (int fy = 0; fy < height; fy++) {
						double oy = y + fy;
						double ox = x + fx;
						if (oy >= 0 && oy < input.GetHeight() && ox >= 0 && ox < input.GetWidth()) {
							double v = input.Get(ox, oy, depth);
							// perform max pooling and store pointers to where
							// the max came from. This will speed up backprop
							// and can help make nice visualizations in future
							if (v > a) {
								a = v;
								winx = ox;
								winy = oy;
							}
						}
					}
				}
				
				switchx[n] = winx;
				switchy[n] = winy;
				n++;
				output_activation.Set(ax, ay, depth, a);
			}
		}
	}
	
	return output_activation;
}

void PoolLayer::Backward() {
	// pooling layers have no parameters, so simply compute
	// gradient wrt data here
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out gradient wrt data
	
	int n = 0;
	for (int depth = 0; depth < output_depth; depth++)
	{
		// WTF C# version int n = depth * output_width * output_height;
		
		for (int ax = 0; ax < output_width; ax++) {
			for (int ay = 0; ay < output_height; ay++) {
				double chain_gradient_ = output_activation.GetGradient(ax, ay, depth);
				input.AddGradient(switchx[n], switchy[n], depth, chain_gradient_);
				n++;
			}
		}
	}
}

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void PoolLayer::Store(ValueMap& map) const {
	STOREVAR(out_depth, output_depth);
	STOREVAR(out_sx, output_width);
	STOREVAR(out_sy, output_height);
	STOREVAR(layer_type, GetKey());
	STOREVAR(sx, width);
	STOREVAR(sy, height);
	STOREVAR(stride, stride);
	STOREVAR(pad, pad);
}

void PoolLayer::Load(const ValueMap& map) {
	LOADVAR(output_depth, out_depth);
	LOADVAR(output_width, out_sx);
	LOADVAR(output_height, out_sy);
	LOADVAR(width, sx);
	LOADVAR(height, sy);
	LOADVAR(stride, stride);
	LOADVARDEF(pad, pad, 0); // backwards compatibility
	int length = output_depth * output_width * output_height;
	switchx.SetCount(0);
	switchx.SetCount(length, 0);
	switchy.SetCount(0);
	switchy.SetCount(length, 0);
}

String PoolLayer::ToString() const {
	return Format("Pool: w:%d, h:%d, d:%d stride:%d pad:%d",
		output_width, output_height, output_depth, stride, pad);
}

}
