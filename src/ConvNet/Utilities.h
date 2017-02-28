#ifndef _ConvNet_Utilities_h_
#define _ConvNet_Utilities_h_

#include <random>
#include <Core/Core.h>


namespace ConvNet {
using namespace Upp;



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
	
	int width;
	int height;
	int depth;
	int length;
	
public:

	
	Volume();
	Volume(const Volume& o) {*this = o;}
	Volume(int width, int height, int depth); // Volume will be filled with random numbers
	Volume(int width, int height, int depth, double default_value);
	Volume(const Vector<double>& weights);
	Volume& Init(int width, int height, int depth); // Volume will be filled with random numbers
	Volume& Init(int width, int height, int depth, double default_value);
	
	Volume& operator=(const Volume& src);
	
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
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
	
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

struct MaxMin : Moveable<MaxMin> {
	int maxi, mini;
	double maxv, minv;
	MaxMin() : maxi(-1), mini(-1), maxv(-DBL_MAX), minv(DBL_MAX) {}
	void Set(int mini, double minv, int maxi, double maxv);
};

class Window {
	
	Vector<double> v;
	double sum;
	int size, minsize;
	
public:
	
	// a window stores _size_ number of values
	// and returns averages. Useful for keeping running
	// track of validation or training accuracy during SGD
	Window() {
		size = 100;
		minsize = 20;
		sum = 0;
	}
	
	Window& Init(int size, int minsize) {
		this->size = size;
		this->minsize = minsize;
		sum = 0;
		return *this;
	}
	
	
	void Add(double x) {
		v.Add(x);
		sum += x;
		if (v.GetCount() > size) {
			sum -= v[0];
			v.Remove(0);
		}
	}
	
	double GetAverage() const {
		if (v.GetCount() < minsize)
			return -1;
		else
			return sum / v.GetCount();
	}
	
	void Reset() {
		v.Clear();
		sum = 0;
	}
	
	
	// returns min, max and indeces of an array
	MaxMin GetMaxMin(const Vector<double>& w) const {
		MaxMin res;
		if(w.GetCount() == 0)
			return res;
		
		double maxv = w[0];
		double minv = w[0];
		int maxi = 0;
		int mini = 0;
		for (int i = 1; i < w.GetCount(); i++) {
			if (w[i] > maxv) {
				maxv = w[i];
				maxi = i;
			}
			if (w[i] < minv) {
				minv = w[i];
				mini = i;
			}
		}
		res.Set(mini, minv, maxi, maxv);
		return res;
	}
	
	// returns string representation of float
	// but truncated to length of d digits
	String GetF2T(double x, int d = 5) const {
		double dd = 1.0 * pow(10.0, d);
		return DblStr(floor(x*dd)/dd);
	}

};

}


#endif
