#include "LayerBase.h"

namespace ConvNet {

void LayerBase::InitConv(int input_width, int input_height, int input_depth) {
	
	// Update Output Size
	
	// required
	output_depth = filter_count;
	
	// computed
	// note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
	// volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
	// final application.
	output_width = (int)floor((input_width + GetPad() * 2 - width) / (double)GetStride() + 1);
	output_height = (int)floor((input_height + GetPad() * 2 - height) / (double)GetStride() + 1);
	//if (output_width % 2 != 0) output_width++;
	//if (output_height % 2 != 0) output_height++;
	
	// initializations
	double bias = bias_pref;
	
	filters.SetCount(0);
	
	for (int i = 0; i < output_depth; i++) {
		filters.Add().Init(width, height, input_depth);
	}
	
	biases.Init(1, 1, output_depth, bias);
}


Volume& LayerBase::ForwardConv(Volume& input, bool is_training) {
	// optimized code by @mdda that achieves 2x speedup over previous version
	
	input_activation = &input;
	output_activation.Init(output_width, output_height, output_depth, 0.0);
	
	int volume_width = input.GetWidth();
	int volume_height = input.GetHeight();
	int xy_stride = GetStride();
	
	for (int depth = 0; depth < output_depth; depth++)
	{
		const Volume& filter = filters[depth];
		int y = -1 * GetPad();
		
		for (int ay = 0; ay < output_height; y += xy_stride, ay++) {
			int x = -1 * GetPad();
			for (int ax = 0; ax < output_width; x += xy_stride, ax++) {
				
				// convolve centered at this particular location
				double a = 0.0;
				for (int fy = 0; fy < filter.GetHeight(); fy++) {
					int oy = y + fy; // coordinates in the original input array coordinates
					if (oy < 0) oy = 0;
					else if (oy >= volume_height) oy = volume_height - 1;
					
					for (int fx = 0; fx < filter.GetWidth(); fx++) {
						int ox = x + fx;
						if (ox < 0) ox = 0;
						else if (ox >= volume_width) ox = volume_width -1;
						
						for (int fd = 0; fd < filter.GetDepth(); fd++) {
							// avoid function call overhead (x2) for efficiency, compromise modularity :(
							a += filter.Get(fx, fy, fd) * input.Get(ox, oy, fd);
						}
					}
				}
				
				a += biases.Get(depth);
				output_activation.Set(ax, ay, depth, a);
			}
		}
	}
	
	return output_activation;
}

void LayerBase::BackwardConv() {
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out gradient wrt bottom data, we're about to fill it
	
	int volume_width = input.GetWidth();
	int volume_height = input.GetHeight();
	int volumeDepth = input.GetDepth();
	int xy_stride = stride;
	
	
	for (int depth = 0; depth < output_depth; depth++)
	{
		Volume& filter = filters[depth];
		
		int y = -1 * pad;
		for (int ay = 0; ay < output_height; y += xy_stride, ay++) {
			
			int x = -1 * pad;
			for (int ax = 0; ax < output_width; x += xy_stride, ax++) {
				
				// convolve centered at this particular location
				double chain_gradient_ = output_activation.GetGradient(ax, ay, depth);
				ASSERT(IsFin(chain_gradient_));
				
				// gradient from above, from chain rule
				for (int fy = 0; fy < filter.GetHeight(); fy++) {
					int oy = y + fy; // coordinates in the original input array coordinates
					if (oy < 0) oy = 0;
					else if (oy >= volume_height) oy = volume_height - 1;
					
					for (int fx = 0; fx < filter.GetWidth(); fx++) {
						int ox = x + fx;
						if (ox < 0) ox = 0;
						else if (ox >= volume_width) ox = volume_width -1;
						
						for (int fd = 0; fd < filter.GetDepth(); fd++) {
							filter.AddGradient(fx, fy, fd, input.Get(ox, oy, fd) * chain_gradient_);
							input.AddGradient(ox, oy, fd, filter.Get(fx, fy, fd) * chain_gradient_);
						}
					}
				}
				
				biases.AddGradient(depth, chain_gradient_);
			}
		}
	}
}

String LayerBase::ToStringConv() const {
	return Format("Conv: w:%d, h:%d, d:%d, bias-pref:%2!,n, filters:%d l1-decay:%2!,n l2-decay:%2!,n stride:%d pad:%d",
		width, height, input_depth, bias_pref, filter_count, l1_decay_mul, l2_decay_mul, stride, pad);
}

}
