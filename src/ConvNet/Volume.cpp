#include "ConvNet.h"

namespace ConvNet {

Volume::Volume() {
	width = 0;
	height = 0;
	depth = 0;
	length = 0;
}

Volume::Volume(int width, int height, int depth) {
	ASSERT(width > 0 && height > 0 && depth > 0);
	Init(width, height, depth);
}

Volume::Volume(int width, int height, int depth, double c) {
	ASSERT(width > 0 && height > 0 && depth > 0);
	Init(width, height, depth, c);
}

Volume::Volume(const Vector<double>& weights) {
	// we were given a list in weights, assume 1D volume and fill it up
	width = 1;
	height = 1;
	depth = weights.GetCount();
	length = depth;
	
	HeaplessCopy(this->weights, weights);
	
	weight_gradients.SetCount(depth, 0.0);
}

Volume::Volume(int width, int height, int depth, const Vector<double>& weights) {
	ASSERT(width > 0 && height > 0 && depth > 0);
	this->width = width;
	this->height = height;
	this->depth = depth;
	length = width * height * depth;
	
	ASSERT(length == weights.GetCount());
	
	HeaplessCopy(this->weights, weights);
	
	weight_gradients.SetCount(length, 0.0);
}

Volume::Volume(int width, int height, int depth, Volume& vol) {
	ASSERT(width > 0 && height > 0 && depth > 0);
	this->width = width;
	this->height = height;
	this->depth = depth;
	length = width * height * depth;
	
	HeaplessCopy(this->weights, vol.weights);
	
	ASSERT(this->weights.GetCount() == length);
	
	weight_gradients.SetCount(length, 0.0);
}

Volume::~Volume() {
	
}

void Volume::Serialize(Stream& s) {
	s % weight_gradients % weights % width % height % depth % length;
}

int Volume::GetMaxColumn() const {
	double max = -DBL_MAX;
	int pos = -1;
	for(int i = 0; i < weights.GetCount(); i++) {
		double d = weights[i];
		if (i == 0 || d > max) {
			max = d;
			pos = i;
		}
	}
	return pos;
}

int Volume::GetSampledColumn() const {
	// sample argmax from w, assuming w are
	// probabilities that sum to one
	double r = Randomf();
	double x = 0.0;
	for(int i = 0; i < weights.GetCount(); i++) {
		x += weights[i];
		if (x > r) {
			return i;
		}
	}
	return weights.GetCount() - 1; // pretty sure we should never get here?
}

void Volume::SetData(Vector<double>& data) {
	weights.SetCount(data.GetCount());
	for(int i = 0; i < weights.GetCount(); i++)
		weights[i] = data[i];
	weight_gradients.SetCount(data.GetCount(), 0);
}

Volume& Volume::operator=(const Volume& src) {
	width = src.width;
	height = src.height;
	depth = src.depth;
	length = src.length;
	HeaplessCopy(weights, src.weights);
	HeaplessCopy(weight_gradients, src.weight_gradients);
	return *this;
}

Volume& Volume::Set(const Volume& src) {
	width = src.width;
	height = src.height;
	depth = src.depth;
	length = src.length;
	HeaplessCopy(weights, src.weights);
	HeaplessCopy(weight_gradients, src.weight_gradients);
	return *this;
}

Volume& Volume::Set(const Vector<double>& src) {
	width = 1;
	height = 1;
	depth = src.GetCount();
	length = depth;
	HeaplessCopy(weights, src);
	weight_gradients.SetCount(length);
	for(int i = 0; i < length; i++)
		weight_gradients[i] = 0;
	return *this;
}

Volume& Volume::Init(int width, int height, int depth) {
	ASSERT(width > 0 && height > 0 && depth > 0);
	
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
	ASSERT(width > 0 && height > 0 && depth > 0);
	
	// we were given dimensions of the vol
	this->width = width;
	this->height = height;
	this->depth = depth;
	
	int n = width * height * depth;
	int prev_length = length;
	
	length = n;
	weights.SetCount(n);
	weight_gradients.SetCount(n);
	
	for (int i = 0; i < n; i++) {
		weights[i] = default_value;
		weight_gradients[i] = 0.0;
	}
	
	return *this;
}

Volume& Volume::Init(int width, int height, int depth, const Vector<double>& w) {
	ASSERT(width > 0 && height > 0 && depth > 0);
	
	// we were given dimensions of the vol
	this->width = width;
	this->height = height;
	this->depth = depth;
	
	int n = width * height * depth;
	ASSERT(n == w.GetCount());
	
	length = n;
	weights.SetCount(n);
	weight_gradients.SetCount(n);
	
	for (int i = 0; i < n; i++) {
		weights[i] = w[i];
		weight_gradients[i] = 0.0;
	}
	
	return *this;
}

int Volume::GetPos(int x, int y, int d) const {
	ASSERT(x >= 0 && y >= 0 && d >= 0 && x < width && y < height && d < depth);
	return ((width * y) + x) * depth + d;
}

double Volume::Get(int x, int y, int d) const {
	int ix = GetPos(x,y,d);
	return weights[ix];
}

void Volume::Set(int x, int y, int d, double v) {
	int ix = GetPos(x,y,d);
	weights[ix] = v;
}

void Volume::Add(int x, int y, int d, double v) {
	int ix = GetPos(x,y,d);
	weights[ix] = weights[ix] + v;
}

void Volume::Add(int i, double v) {
	weights[i] = weights[i] + v;
}

double Volume::GetGradient(int x, int y, int d) const {
	int ix = GetPos(x,y,d);
	return weight_gradients[ix];
}

void Volume::SetGradient(int x, int y, int d, double v) {
	int ix = GetPos(x,y,d);
	weight_gradients[ix] = v;
}

void Volume::AddGradient(int x, int y, int d, double v) {
	int ix = GetPos(x,y,d);
	weight_gradients[ix] += v;
}

void Volume::ZeroGradients() {
	for(int i = 0; i < weight_gradients.GetCount(); i++)
		weight_gradients[i] = 0.0;
}

void Volume::AddFrom(const Volume& volume) {
	for (int i = 0; i < weights.GetCount(); i++) {
		weights[i] = weights[i] + volume.Get(i);
	}
}

void Volume::AddGradientFrom(const Volume& volume) {
	for (int i = 0; i < weight_gradients.GetCount(); i++) {
		weight_gradients[i] += volume.GetGradient(i);
	}
}

void Volume::AddFromScaled(const Volume& volume, double a) {
	for (int i = 0; i < weights.GetCount(); i++) {
		weights[i] = weights[i] + a * volume.Get(i);
	}
}

void Volume::SetConst(double c) {
	for (int i = 0; i < weights.GetCount(); i++) {
		weights[i] = c;
	}
}

void Volume::SetConstGradient(double c) {
	for (int i = 0; i < weight_gradients.GetCount(); i++) {
		weight_gradients[i] = c;
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


void Volume::Augment(int crop, int dx, int dy, bool fliplr) {
	
	// note assumes square outputs of size crop x crop
	if (dx == -1) dx = Random(width - crop);
	if (dy == -1) dy = Random(height - crop);
	
	// randomly sample a crop in the input volume
	if (crop != width || dx != 0 || dy != 0) {
		Volume W;
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
		SwapData(W);
	}
	
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
		SwapData(vol);
	}
}

void Volume::SwapData(Volume& vol) {
	Swap(vol.weight_gradients, weight_gradients);
	Swap(vol.weights, weights);
	Swap(vol.width, width);
	Swap(vol.height, height);
	Swap(vol.depth, depth);
	Swap(vol.length, length);
	
}







void RandomPermutation(int n, Vector<int>& array) {
	int i = n;
	int j = 0;
	
	array.SetCount(n);
	
	for (int q = 0; q < n; q++)
		array[q] = q;
	
	while (i--) {
		j = floor(Randomf() * (i+1));
		int temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
}

}
