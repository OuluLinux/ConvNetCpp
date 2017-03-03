#include "ConvNet.h"

namespace ConvNet {

Volume::Volume() {
	width = 0;
	height = 0;
	depth = 0;
	length = 0;
}

Volume& Volume::operator=(const Volume& src) {
	width = src.width;
	height = src.height;
	depth = src.depth;
	length = src.length;
	weights.SetCount(src.weights.GetCount());
	for(int i = 0; i < weights.GetCount(); i++)
		weights[i] = src.weights[i];
	weight_gradients.SetCount(src.weight_gradients.GetCount());
	for(int i = 0; i < weight_gradients.GetCount(); i++)
		weight_gradients[i] = src.weight_gradients[i];
	return *this;
}

Volume::Volume(int width, int height, int depth) {
	Init(width, height, depth);
}

Volume::Volume(int width, int height, int depth, double c) {
	Init(width, height, depth, c);
}

Volume::Volume(const Vector<double>& weights) {
	// we were given a list in weights, assume 1D volume and fill it up
	width = 1;
	height = 1;
	depth = weights.GetCount();
	length = depth;
	
	this->weights <<= weights;
	weight_gradients.SetCount(depth, 0.0);
}

Volume& Volume::Init(int width, int height, int depth) {
	// we were given dimensions of the vol
	this->width = width;
	this->height = height;
	this->depth = depth;
	
	int n = width * height * depth;
	
	length = n;
	weights.SetCount(n, 0.0);
	weight_gradients.SetCount(n, 0.0);
	
	RandomGaussian& rand = GetRandomGaussian(length);

	for (int i = 0; i < n; i++) {
		weights[i] = rand;
	}
	
	return *this;
}

Volume& Volume::Init(int width, int height, int depth, double default_value) {
	// we were given dimensions of the vol
	this->width = width;
	this->height = height;
	this->depth = depth;
	
	int n = width * height * depth;
	
	length = n;
	weights.SetCount(n);
	weight_gradients.SetCount(n);
	
	for (int i = 0; i < n; i++) {
		weights[i] = default_value;
		weight_gradients[i] = 0.0;
	}
	
	return *this;
}

double Volume::Get(int x, int y, int d) const {
	int ix = ((width * y) + x) * depth + d;
	return weights[ix];
}

void Volume::Set(int x, int y, int d, double v) {
	int ix = ((width * y) + x) * depth + d;
	weights[ix] = v;
}

void Volume::Add(int x, int y, int d, double v) {
	int ix = ((width * y) + x) * depth + d;
	weights[ix] += v;
}

double Volume::GetGradient(int x, int y, int d) const {
	int ix = ((width * y) + x) * depth + d;
	return weight_gradients[ix];
}

void Volume::SetGradient(int x, int y, int d, double v) {
	int ix = ((width * y) + x) * depth + d;
	weight_gradients[ix] = v;
}

void Volume::AddGradient(int x, int y, int d, double v) {
	int ix = ((width * y) + x) * depth + d;
	weight_gradients[ix] += v;
}

void Volume::ZeroGradients() {
	for(int i = 0; i < weight_gradients.GetCount(); i++)
		weight_gradients[i] = 0.0;
}

void Volume::AddFrom(const Volume& volume) {
	for (int i = 0; i < weights.GetCount(); i++) {
		weights[i] += volume.Get(i);
	}
}

void Volume::AddGradientFrom(const Volume& volume) {
	for (int i = 0; i < weight_gradients.GetCount(); i++) {
		weight_gradients[i] += volume.GetGradient(i);
	}
}

void Volume::AddFromScaled(const Volume& volume, double a) {
	for (int i = 0; i < weights.GetCount(); i++) {
		weights[i] += a * volume.Get(i);
	}
}

void Volume::SetConst(double c) {
	for (int i = 0; i < weights.GetCount(); i++) {
		weights[i] += c;
	}
}

double Volume::Get(int i) const {
	return weights[i];
}

double Volume::GetGradient(int i) const {
	return weight_gradients[i];
}

void Volume::SetGradient(int i, double v) {
	weight_gradients[i] = v;
}

void Volume::AddGradient(int i, double v) {
	weight_gradients[i] += v;
}

void Volume::Set(int i, double v) {
	weights[i] = v;
}

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void Volume::Store(ValueMap& map) const {
	STOREVAR(sx, width);
	STOREVAR(sy, height);
	STOREVAR(depth, depth);
	
	Value w;
	for(int i = 0; i < weights.GetCount(); i++) {
		double value = weights[i];
		w.Add(value);
	}
	map.GetAdd("w") = w;
	
	Value dw;
	for(int i = 0; i < weight_gradients.GetCount(); i++) {
		double value = weight_gradients[i];
		dw.Add(value);
	}
	map.GetAdd("dw") = dw;
}

void Volume::Load(const ValueMap& map) {
	LOADVAR(width, sx);
	LOADVAR(height, sy);
	LOADVAR(depth, depth);
	
	length = width * height * depth;
	
	weights.SetCount(0);
	weights.SetCount(length, 0);
	weight_gradients.SetCount(0);
	weight_gradients.SetCount(length, 0);
	
	// copy over the elements.
	Value w = map.GetValue(map.Find("w"));
	
	for (int i = 0; i < length; i++) {
		double value = w[i];
		weights[i] = value;
	}
	
	int i = map.Find("dw");
	if (i != -1) {
		Value dw = map.GetValue(i);
		for (int i = 0; i < length; i++) {
			double value = dw[i];
			weight_gradients[i] = value;
		}
	}
}

void Volume::Augment(int crop, int dx, int dy, bool fliplr) {
	
	// note assumes square outputs of size crop x crop
	if (dx == -1) dx = Random(width - crop);
	if (dy == -1) dy = Random(height - crop);
	
	// randomly sample a crop in the input volume
	Volume W;
	if (crop != width || dx != 0 || dy != 0) {
		W.Init(crop, crop, depth, 0.0);
		for (int x = 0; x < crop; x++) {
			for (int y = 0; y < crop; y++) {
				if (x+dx < 0 || x+dx >= width || y+dy < 0 || y+dy >= height)
					continue; // oob
				for (int d = 0; d < depth; d++) {
					W.Set(x, y, d, Get(x+dx, y+dy, d)); // copy data over
				}
			}
		}
		*this = W;
	}/* else {
		W = *this;
	}*/
	
	if (fliplr) {
		// flip volume horziontally
		Volume vol;
		vol.Init(width, height, depth, 0.0);
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int d = 0; d < depth; d++) {
					vol.Set(x, y, d, Get(width - x - 1, y, d)); // copy data over
				}
			}
		}
		*this = vol; //swap
	}
}

}
