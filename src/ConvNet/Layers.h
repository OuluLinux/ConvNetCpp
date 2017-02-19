#ifndef _ConvNet_Layers_h_
#define _ConvNet_Layers_h_

#include "LayerBase.h"

namespace ConvNet {

typedef Exc ArgumentException;
typedef Exc NotImplementedException;
typedef Exc Exception;

#define ASSERTEXC(x) if (!(x)) {throw Exception();}


class IClassificationLayer {
	
public:
	IClassificationLayer() : class_count(0) {}
	
	int class_count;
};

class IDotProductLayer {
	
public:
	IDotProductLayer() : bias_pref(0) {}
	
	double bias_pref;
};

class ConvLayer : public LayerBase, public IDotProductLayer {
	
protected:
	ConvLayer(const ConvLayer& o) {}
	
public:
	ConvLayer(int width, int height, int filter_count);
	
	int width;
	int height;
	Volume biases;
	Vector<Volume> filters;
	int filter_count;
	double l1_decay_mul;
	double l2_decay_mul;
	int stride;
	int pad;
	
	int GetStride() const {return stride;}
	int GetPad() const {return pad;}
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	void UpdateOutputSize();
	virtual Vector<ParametersAndGradients>& GetParametersAndGradients();
	virtual String GetKey() const {return "conv";}
	
};

class DropOutLayer : public LayerBase {
	
	Vector<bool> dropped;
	
protected:
	DropOutLayer(const DropOutLayer& o) {}
	
public:
	double drop_prob;
	
	DropOutLayer(double drop_prob);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "dropout";}
	
};



class FullyConnLayer : public LayerBase, public IDotProductLayer {
	
	int input_count;
	
protected:
	FullyConnLayer(const FullyConnLayer& o) {}
	
public:
	FullyConnLayer(int neuron_count);
	
	Volume biases;
	Vector<Volume> filters;
	double l1_decay_mul;
	double l2_decay_mul;
	int neuron_count;
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual Vector<ParametersAndGradients>& GetParametersAndGradients();
	virtual String GetKey() const {return "fc";}
	
};

class InputLayer : public LayerBase {
	
protected:
	InputLayer(const InputLayer& o) {}
	
public:
	
	InputLayer(int input_width, int input_height, int input_depth);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual Volume& Forward(bool is_training);
	virtual String GetKey() const {return "input";}
	
};

class LastLayerBase : public LayerBase {
	
protected:
	LastLayerBase(const LastLayerBase& o) {}
	
public:
	LastLayerBase() {}
	
	virtual double Backward(const Vector<double>& y) = 0;
	virtual double Backward(double y) = 0;
	virtual String GetKey() const {return "lastlayerbase";}
	
};



// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size
class MaxoutLayer : public LayerBase {
	
	Vector<int> switches;
	
protected:
	MaxoutLayer(const MaxoutLayer& o) {}
	
public:
	MaxoutLayer(int group_size = 2);
	
	int group_size;
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "maxout";}
	
};

class PoolLayer : public LayerBase {
	Vector<int> switchx;
	Vector<int> switchy;
	
protected:
	PoolLayer(const PoolLayer& o) {}
	
public:
	PoolLayer(int width, int height);
	
	int width;
	int height;
	int stride;
	int pad;
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	void UpdateOutputSize();
	virtual String GetKey() const {return "pool";}
	
};


// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.
class RegressionLayer : public LastLayerBase {
	
protected:
	RegressionLayer(const RegressionLayer& o) {}
	
public:
	RegressionLayer();
	virtual double Backward(double y);
	virtual double Backward(const Vector<double>& y);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "reg";}
	
};


// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
class ReluLayer : public LayerBase {
	
protected:
	ReluLayer(const ReluLayer& o) {}
	
public:
	ReluLayer();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "relu";}
	
};


// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.
class SigmoidLayer : public LayerBase {
	
protected:
	SigmoidLayer(const SigmoidLayer& o) {}
	
public:

	SigmoidLayer();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "sigmoid";}
	
};


// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)
class SoftmaxLayer : public LastLayerBase, public IClassificationLayer {
	Vector<double> es;
	
protected:
	SoftmaxLayer(const SoftmaxLayer& o) {}
	
public:
	SoftmaxLayer(int class_count);
	
	virtual double Backward(double y);
	virtual double Backward(const Vector<double>& y);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "softmax";}
	
};


class SvmLayer : public LastLayerBase, public IClassificationLayer {
	
protected:
	SvmLayer(const SvmLayer& o) {}
	
public:
	SvmLayer(int class_count);
	
	virtual double Backward(double yd);
	virtual double Backward(const Vector<double>& y);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "svm";}
	
};


class TanhLayer : public LayerBase {
	
protected:
	TanhLayer(const TanhLayer& o) {}
	
public:
	
	TanhLayer();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "tanh";}
	
};

}

#endif
