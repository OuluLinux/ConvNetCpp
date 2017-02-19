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

}
