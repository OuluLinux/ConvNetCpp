#include "LayerBase.h"


namespace ConvNet {

void LayerBase::InitUnpool(int input_width, int input_height, int input_depth) {
	
	// Update Output Size
	
	// computed
	output_depth = input_depth;
	output_width  = (input_width - 1)  * stride - pad * 2 + width;
	output_height = (input_height - 1) * stride - pad * 2 + height;
	
	// store switches for x,y coordinates for where the max comes from, for each output neuron
	switchx.SetCount(output_width * output_height * output_depth, 0);
	switchy.SetCount(output_width * output_height * output_depth, 0);
}

Volume& LayerBase::ForwardUnpool(Volume& input, bool is_training) {
	input_activation = &input;
	
	output_activation.Init(output_width, output_height, output_depth, 0.0);
	
	int input_width = input.GetWidth();
	int input_height = input.GetHeight();
	
	int n = 0; // a counter for switches
	for (int depth = 0; depth < output_depth; depth++)
	{
		// WTF C# version int n = depth * output_width * output_height;
		
		int x = +1.0 * pad - floor(width - 1);
		for (int ax = 0; ax < output_width; ax++) {
			int y = +1.0 * pad - floor(height - 1);
			for (int ay = 0; ay < output_height; ay++) {
				// convolve centered at this particular location
				double a = -DBL_MAX;
				int winx = -1, winy = -1;
				
				for (int fx = 0; fx < width; fx++) {
					int ox = x + fx;
					if (ox < 0) continue;
					else if (ox >= input_width) continue;
					
					for (int fy = 0; fy < height; fy++) {
						int oy = y + fy;
						if (oy < 0) continue;
						else if (oy >= input_height) continue;
						
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
				
				switchx[n] = winx;
				switchy[n] = winy;
				n++;
				output_activation.Set(ax, ay, depth, a);
				
				if ((ay % stride) == (stride-1)) y++;
			}
			
			if ((ax % stride) == (stride-1)) x++;
		}
	}
	
	return output_activation;
}

void LayerBase::BackwardUnpool() {
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
				int x = switchx[n];
				int y = switchy[n];
				if (x != -1 && y != -1)
					input.AddGradient(x, y, depth, chain_gradient_);
				n++;
			}
		}
	}
}

String LayerBase::ToStringUnpool() const {
	return Format("Pool: w:%d, h:%d, d:%d stride:%d pad:%d",
		output_width, output_height, output_depth, stride, pad);
}

}
