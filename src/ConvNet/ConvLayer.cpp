#include "Layers.h"

namespace ConvNet {

ConvLayer::ConvLayer(int width, int height, int filter_count) {
	stride = 1;
	pad = 0;
	
	l1_decay_mul = 0.0;
	l2_decay_mul = 1.0;
	
	this->filter_count = filter_count;
	this->width = width;
	this->height = height;
}

void ConvLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	UpdateOutputSize();
}

void ConvLayer::UpdateOutputSize() {
	// required
	output_depth = filter_count;
	
	// computed
	// note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
	// volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
	// final application.
	output_width = (int)floor((input_width + GetPad() * 2 - width) / (double)GetStride() + 1);
	output_height = (int)floor((input_height + GetPad() * 2 - height) / (double)GetStride() + 1);
	
	// initializations
	double bias = bias_pref;
	
	filters.Clear();
	
	for (int i = 0; i < output_depth; i++) {
		filters.Add().Init(width, height, input_depth);
	}
	
	biases.Init(1, 1, output_depth, bias);
}

Volume& ConvLayer::Forward(Volume& input, bool is_training) {
	// optimized code by @mdda that achieves 2x speedup over previous version
	
	input_activation = &input;
	output_activation.Init(output_width, output_height, output_depth, 0.0);
	
	int volume_width = input.GetWidth();
	int volume_height = input.GetHeight();
	int xy_stride = GetStride();
	
	for (int depth = 0; depth < output_depth; depth++)
	{
		const Volume& filter = filters[depth];
		double y = -1.0 * GetPad();
		
		for (int ay = 0; ay < output_height; y += xy_stride, ay++) {
			double x = -1.0 * GetPad();
			for (int ax = 0; ax < output_width; x += xy_stride, ax++) {
				
				// convolve centered at this particular location
				double a = 0.0;
				for (int fy = 0; fy < filter.GetHeight(); fy++) {
					double oy = y + fy; // coordinates in the original input array coordinates
					for (int fx = 0; fx < filter.GetWidth(); fx++) {
						double ox = x + fx;
						if (oy >= 0 && oy < volume_height && ox >= 0 && ox < volume_width) {
							for (double fd = 0; fd < filter.GetDepth(); fd++) {
								// avoid function call overhead (x2) for efficiency, compromise modularity :(
								a += filter.Get((filter.GetWidth() * fy + fx) * filter.GetDepth() + fd) *
									 input.Get((volume_width * oy + ox) * input.GetDepth() + fd);
							}
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

void ConvLayer::Backward() {
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out gradient wrt bottom data, we're about to fill it
	
	int volume_width = input.GetWidth();
	int volume_height = input.GetHeight();
	int volumeDepth = input.GetDepth();
	int xy_stride = GetStride();
	
	
	for (int depth = 0; depth < output_depth; depth++)
	{
		Volume& filter = filters[depth];
		
		double y = -1.0 * GetPad();
		for (int ay = 0; ay < output_height; y += xy_stride, ay++) {
			
			double x = -1.0 * GetPad();
			for (int ax = 0; ax < output_width; x += xy_stride, ax++) {
				
				// convolve centered at this particular location
				double chain_gradient_ = output_activation.GetGradient(ax, ay, depth);
				
				// gradient from above, from chain rule
				for (int fy = 0; fy < filter.GetHeight(); fy++) {
					double oy = y + fy; // coordinates in the original input array coordinates
					for (int fx = 0; fx < filter.GetWidth(); fx++) {
						double ox = x + fx;
						if (oy >= 0 && oy < volume_height && ox >= 0 && ox < volume_width) {
							for (int fd = 0; fd < filter.GetDepth(); fd++) {
								filter.AddGradient(fx, fy, fd, input.Get(ox, oy, fd) * chain_gradient_);
								input.AddGradient( ox, oy, fd, filter.Get(fx, fy, fd) * chain_gradient_);
							}
						}
					}
				}
				
				biases.AddGradient(depth, chain_gradient_);
			}
		}
	}
}

Vector<ParametersAndGradients>& ConvLayer::GetParametersAndGradients() {
	
	response.SetCount(output_depth + 1);
	
	for (int i = 0; i < output_depth; i++) {
		ParametersAndGradients& pag = response[i];
		pag.volume = &this->filters[i];
		pag.l2_decay_mul = &this->l2_decay_mul;
		pag.l1_decay_mul = &this->l1_decay_mul;
	}
	
	ParametersAndGradients& pag = response[output_depth];
	pag.volume = &this->biases;
	pag.l1_decay_mul = 0;
	pag.l2_decay_mul = 0;
	
	return response;
}

}
