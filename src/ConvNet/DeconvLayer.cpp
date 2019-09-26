#include "LayerBase.h"

namespace ConvNet {

int GetOutDimTrans(int in_dim, int filter, int padding, int strides) {
	ASSERT(strides > 0);
	ASSERT(padding <= filter-1);
	
	int ghost_dim = in_dim + (in_dim-1)*(strides-1) + (filter-1) * 2 - padding * 2;
	int out_dim = ghost_dim - (filter-1);
	//if (out_dim % 2 != 0) out_dim++;
	return out_dim;
}

void LayerBase::InitDeconv(int input_width, int input_height, int input_depth) {
	
	// Update Output Size
	
	// required
	output_depth = filter_count;
	
	// interpreted from https://github.com/vdumoulin/conv_arithmetic
	output_width  = GetOutDimTrans(input_width, width, pad, stride);
	output_height = GetOutDimTrans(input_height, height, pad, stride);
	
	// initializations
	double bias = bias_pref;
	
	filters.SetCount(0);
	
	for (int i = 0; i < output_depth; i++) {
		filters.Add().Init(width, height, input_depth);
	}
	
	biases.Init(1, 1, output_depth, bias);
}


Volume& LayerBase::ForwardDeconv(Volume& input, bool is_training) {
	// optimized code by @mdda that achieves 2x speedup over previous version
	
	input_activation = &input;
	output_activation.Init(output_width, output_height, output_depth, 0.0);
	
	int volume_width = input.GetWidth();
	int volume_height = input.GetHeight();
	int volume_depth = input.GetDepth();
	int filter_w = filters[0].GetWidth();
	int filter_h = filters[0].GetHeight();
	
	int g_width = volume_width + (volume_width-1)*(stride-1);
	int g_height = volume_height + (volume_height-1)*(stride-1);
	int gpad_w = filter_w-1 - pad;
	int gpad_h = filter_h-1 - pad;
	int ghost_w = g_width + gpad_w * 2;
	int ghost_h = g_height + gpad_h * 2;
	ghost_image.SetSize(ghost_w, ghost_h, volume_depth);
	ghost_gradients.SetSize(ghost_w, ghost_h, volume_depth);
	
	double div_tmp[36];
	for(int i = 0; i <= stride; i++)
		div_tmp[i] = (double)i / (double)stride;
	
	for (int d = 0; d < volume_depth; d++) {
		
		// Write horizontal lines
		for (int y = 0; y < volume_height; y++) {
			int gx = gpad_w;
			int gy = gpad_h + y * stride;
			
			// Write horizontal lines
			double a, b;
			b = input.Get(0, y, d);
			for (int x = 0; x < volume_width-1; x++) {
				a = b;
				b = input.Get(x+1, y, d);
				
				for (int i = 0; i < stride; i++) {
					double f = div_tmp[i];
					double v = a * (1.0 - f) + b * f;
					ghost_image.Set(gx, gy, d, v);
					gx++;
				}
			}
			ghost_image.Set(gx, gy, d, b); // last position
		}
		
		// Write area between horizontal lines
		if (stride > 1) {
			for (int y = 0; y < volume_height - 1; y++) {
				int gx = gpad_w;
				int gy = gpad_h + y * stride;
			
				for (int x = 0; x < g_width; x++) {
					double a, b;
					a = ghost_image.Get(gx, gy, d);
					b = ghost_image.Get(gx, gy + stride, d);
					
					for (int i = 1; i < stride; i++) {
						double f = div_tmp[i];
						double v = a * (1.0 - f) + b * f;
						ghost_image.Set(gx, gy + i, d, v);
					}
					
					gx++;
				}
				
			}
		}
		
		// Write left-upper corner
		{
			double v = ghost_image.Get(gpad_w, gpad_h, d);
			for (int x = 0; x < gpad_w; x++)
				for (int y = 0; y < gpad_w; y++)
					ghost_image.Set(x, y, d, v);
		}
		
		// Write right-upper corner
		{
			int x_shift = gpad_w + g_width;
			double v = ghost_image.Get(x_shift-1, gpad_h, d);
			for (int x = 0; x < gpad_w; x++)
				for (int y = 0; y < gpad_h; y++)
					ghost_image.Set(x_shift + x, y, d, v);
		}
		
		// Write bottom-left corner
		{
			int y_shift = gpad_h + g_height;
			double v = ghost_image.Get(gpad_w, y_shift-1, d);
			for (int x = 0; x < gpad_w; x++)
				for (int y = 0; y < gpad_h; y++)
					ghost_image.Set(x, y_shift + y, d, v);
		}
		
		// Write bottom-right corner
		{
			int x_shift = gpad_w + g_width;
			int y_shift = gpad_h + g_height;
			double v = ghost_image.Get(x_shift-1, y_shift-1, d);
			for (int x = 0; x < gpad_w; x++)
				for (int y = 0; y < gpad_h; y++)
					ghost_image.Set(x_shift + x, y_shift + y, d, v);
		}
		
		// Write top border
		{
			for (int x = 0; x < g_width; x++) {
				int gx = gpad_w + x;
				double v = ghost_image.Get(gx, gpad_h, d);
				for (int i = 0; i < gpad_h; i++)
					ghost_image.Set(gx, i, d, v);
			}
		}
		// Write bottom border
		{
			int y_shift = gpad_h + g_height;
			for (int x = 0; x < g_width; x++) {
				int gx = gpad_w + x;
				double v = ghost_image.Get(gx, y_shift-1, d);
				for (int i = 0; i < gpad_h; i++)
					ghost_image.Set(gx, y_shift + i, d, v);
			}
		}
		// Write left border
		{
			for (int y = 0; y < g_height; y++) {
				int gy = gpad_h + y;
				double v = ghost_image.Get(gpad_w, gy, d);
				for (int i = 0; i < gpad_w; i++)
					ghost_image.Set(i, gy, d, v);
			}
		}
		// Write right border
		{
			int x_shift = gpad_w + g_width;
			for (int y = 0; y < g_height; y++) {
				int gy = gpad_h + y;
				double v = ghost_image.Get(x_shift-1, gy, d);
				for (int i = 0; i < gpad_w; i++)
					ghost_image.Set(x_shift + i, gy, d, v);
			}
		}
	}
	
	for (int depth = 0; depth < output_depth; depth++)
	{
		const Volume& filter = filters[depth];
		
		for (int ay = 0; ay < output_height; ay++) {
			
			for (int ax = 0; ax < output_width; ax++) {
				
				// convolve centered at this particular location
				double a = 0.0;
				for (int fy = 0; fy < filter.GetHeight(); fy++) {
					for (int fx = 0; fx < filter.GetWidth(); fx++) {
						for (int fd = 0; fd < filter.GetDepth(); fd++) {
							int i = ghost_image.GetPos(ax + fx, ay + fy, fd);
							if (i < ghost_image.GetCount())
								a += filter.Get(fx, fy, fd) * ghost_image.Get(i);
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

double LayerBase::BackwardDeconv() {
	Volume& input = *input_activation;
	
	int volume_width = input.GetWidth();
	int volume_height = input.GetHeight();
	int volume_depth = input.GetDepth();
	int filter_w = filters[0].GetWidth();
	int filter_h = filters[0].GetHeight();
	
	int g_width = volume_width + (volume_width-1)*(stride-1);
	int g_height = volume_height + (volume_height-1)*(stride-1);
	int gpad_w = filter_w-1 - pad;
	int gpad_h = filter_h-1 - pad;
	int ghost_w = g_width + gpad_w * 2;
	int ghost_h = g_height + gpad_h * 2;
	
	input.ZeroGradients(); // zero out gradient wrt bottom data, we're about to fill it
	ghost_gradients.Zero();
	
	for (int depth = 0; depth < output_depth; depth++)
	{
		Volume& filter = filters[depth];
		
		for (int ay = 0; ay < output_height; ay++) {
			
			for (int ax = 0; ax < output_width; ax++) {
				
				// convolve centered at this particular location
				double chain_gradient_ = output_activation.GetGradient(ax, ay, depth);
				ASSERT(IsFin(chain_gradient_));
				
				// gradient from above, from chain rule
				for (int fy = 0; fy < filter.GetHeight(); fy++) {
					for (int fx = 0; fx < filter.GetWidth(); fx++) {
						for (int fd = 0; fd < filter.GetDepth(); fd++) {
							int i = ghost_image.GetPos(ax + fx, ay + fy, fd);
							if (i < ghost_image.GetCount()) {
								double v = ghost_image.Get(i);
								double filter_gradient = v * chain_gradient_;
								filter.AddGradient(fx, fy, fd, filter_gradient);
								
								double input_gradient = filter.Get(fx, fy, fd) * chain_gradient_;
								ghost_gradients.Add(ax + fx, ay + fy, fd, input_gradient);
							}
						}
					}
				}
				
				biases.AddGradient(depth, chain_gradient_);
			}
		}
		
	}
	
	
	double div_tmp[36];
	for(int i = 0; i <= stride; i++)
		div_tmp[i] = (double)i / (double)stride;
	
	for (int d = 0; d < volume_depth; d++) {
		
		// Right border gradient
		{
			int x_shift = gpad_w + g_width;
			for (int y = 0; y < g_height; y++) {
				int gy = gpad_h + y;
				for (int i = 0; i < gpad_w; i++) {
					double v = ghost_gradients.Get(x_shift + i, gy, d);
					ghost_gradients.Add(x_shift-1, gy, d, v);
				}
			}
		}
		// Left border gradient
		{
			for (int y = 0; y < g_height; y++) {
				int gy = gpad_h + y;
				for (int i = 0; i < gpad_w; i++) {
					double v = ghost_gradients.Get(i, gy, d);
					ghost_gradients.Add(gpad_w, gy, d, v);
				}
			}
		}
		// Bottom border gradient
		{
			int y_shift = gpad_h + g_height;
			for (int x = 0; x < g_width; x++) {
				int gx = gpad_w + x;
				for (int i = 0; i < gpad_h; i++) {
					double v = ghost_gradients.Get(gx, y_shift + i, d);
					ghost_gradients.Add(gx, y_shift-1, d, v);
				}
			}
		}
		// Top border gradient
		{
			for (int x = 0; x < g_width; x++) {
				int gx = gpad_w + x;
				for (int i = 0; i < gpad_h; i++) {
					double v = ghost_gradients.Get(gx, i, d);
					ghost_gradients.Add(gx, gpad_h, d, v);
				}
			}
		}
		// Bottom-right corner
		{
			int x_shift = gpad_w + g_width;
			int y_shift = gpad_h + g_height;
			for (int x = 0; x < gpad_w; x++) {
				for (int y = 0; y < gpad_h; y++) {
					double v = ghost_gradients.Get(x_shift + x, y_shift + y, d);
					ghost_gradients.Add(x_shift-1, y_shift-1, d, v);
				}
			}
		}
		// Bottom-left corner
		{
			int y_shift = gpad_h + g_height;
			for (int x = 0; x < gpad_w; x++) {
				for (int y = 0; y < gpad_h; y++) {
					double v = ghost_gradients.Get(x, y_shift + y, d);
					ghost_gradients.Add(gpad_w, y_shift-1, d, v);
				}
			}
		}
		// Right-upper corner
		{
			int x_shift = gpad_w + g_width;
			for (int x = 0; x < gpad_w; x++) {
				for (int y = 0; y < gpad_h; y++) {
					double v = ghost_gradients.Get(x_shift + x, y, d);
					ghost_gradients.Add(x_shift-1, gpad_h, d, v);
				}
			}
		}
		// Left-upper corner
		{
			for (int x = 0; x < gpad_w; x++) {
				for (int y = 0; y < gpad_w; y++) {
					double v = ghost_gradients.Get(x, y, d);
					ghost_gradients.Add(gpad_w, gpad_h, d, v);
				}
			}
		}
		// Area between horizontal lines
		if (stride > 1) {
			for (int y = 0; y < volume_height - 1; y++) {
				int gx = gpad_w;
				int gy = gpad_h + y * stride;
			
				for (int x = 0; x < g_width; x++) {
					for (int i = 1; i < stride; i++) {
						double f = div_tmp[i];
						double v = ghost_gradients.Get(gx, gy + i, d);
						double a = v * (1.0 - f);
						double b = v * f;
						ghost_gradients.Add(gx, gy, d, a);
						ghost_gradients.Add(gx, gy + stride, d, b);
					}
					
					gx++;
				}
				
			}
		}
		// Horizontal lines
		for (int y = 0; y < volume_height; y++) {
			int gx = gpad_w;
			int gy = gpad_h + y * stride;
			
			// Write horizontal lines
			for (int x = 0; x < volume_width-1; x++) {
				for (int i = 0; i < stride; i++) {
					double f = div_tmp[i];
					double v = ghost_gradients.Get(gx, gy, d);
					double a = v * (1.0 - f);
					double b = v * f;
					input.AddGradient(x, y, d, a);
					input.AddGradient(x+1, y, d, b);
					gx++;
				}
			}
			double v = ghost_gradients.Get(gx, gy, d);
			input.AddGradient(volume_width-1, y, d, v);
		}
		
	}
	
	return 0.0;
}

double LayerBase::BackwardDeconv(const Vector<double>& vec) {
	int count = min(vec.GetCount(), output_activation.GetCount());
	
	for(int i = 0; i < count; i++)
		output_activation.SetGradient(i, vec[i]);
	
	return BackwardDeconv();
}

String LayerBase::ToStringDeconv() const {
	return Format("Conv: w:%d, h:%d, d:%d, bias-pref:%2!,n, filters:%d l1-decay:%2!,n l2-decay:%2!,n stride:%d pad:%d",
		width, height, input_depth, bias_pref, filter_count, l1_decay_mul, l2_decay_mul, stride, pad);
}

}
