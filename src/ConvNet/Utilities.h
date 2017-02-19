#ifndef _ConvNet_Utilities_h_
#define _ConvNet_Utilities_h_

#include <random>
#include <Core/Core.h>
using namespace Upp;


namespace ConvNet
{



// Volume is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width, height, and depth.
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t.
// the data.
class Volume : Moveable<Volume> {
	Vector<double> weight_gradients;
	Vector<double> weights;
	

protected:
	Volume(const Volume& o);// {}
	
	int width;
	int height;
	int depth;
	int length;
	
public:

	Volume& operator=(const Volume& src);
	
	// Volume will be filled with random numbers
	Volume();
	Volume(int width, int height, int depth);
	Volume(int width, int height, int depth, double default_value);
	Volume(const Vector<double>& weights);
	Volume& Init(int width, int height, int depth);
	Volume& Init(int width, int height, int depth, double default_value);
	
	const Vector<double>& GetWeights() const {return weights;}
	
	virtual void Add(int x, int y, int d, double v);
	virtual void AddFrom(const Volume& volume);
	virtual void AddFromScaled(const Volume& volume, double a);
	virtual void AddGradient(int x, int y, int d, double v);
	virtual void AddGradientFrom(const Volume& volume);
	virtual double Get(int x, int y, int d) const;
	virtual double GetGradient(int x, int y, int d) const;
	virtual void Set(int x, int y, int d, double v);
	virtual void SetConst(double c);
	virtual void SetGradient(int x, int y, int d, double v);
	virtual double Get(int i) const;
	virtual void Set(int i, double v);
	virtual double GetGradient(int i) const;
	virtual void SetGradient(int i, double v);
	virtual void AddGradient(int i, double v);
	virtual void ZeroGradients();
	
	int GetWidth()  const {return width;}
	int GetHeight() const {return height;}
	int GetDepth()  const {return depth;}
	int GetLength() const {return length;}
	
};

typedef Volume* VolumePtr;

class ParametersAndGradients : Moveable<ParametersAndGradients> {
	
public:
	ParametersAndGradients() {
		l2_decay_mul = NULL;
		l1_decay_mul = NULL;
		volume = NULL;
	}
	
	ParametersAndGradients(const ParametersAndGradients& o) {
		l2_decay_mul = o.l2_decay_mul;
		l1_decay_mul = o.l1_decay_mul;
		volume = o.volume;
	}
	
	Volume* volume;
	double* l2_decay_mul;
	double* l1_decay_mul;
};

class RandomGaussian {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution;
	
public:

	// weight normalization is done to equalize the output
	// variance of every neuron, otherwise neurons with a lot
	// of incoming connections have outputs of larger variance
	RandomGaussian(int length) : distribution(0, sqrt(1.0 / (double)(length))) {
		generator.seed(Random(1000000000));
	}
	double Get() {return distribution(generator);}
	operator double() {return distribution(generator);}
	
};

inline RandomGaussian& GetRandomGaussian(int length) {
	ArrayMap<int, RandomGaussian>& rands = Single<ArrayMap<int, RandomGaussian> >();
	int i = rands.Find(length);
	if (i == -1) {
		return rands.Add(length, new RandomGaussian(length));
	} else {
		return rands[i];
	}
}

}

#endif
