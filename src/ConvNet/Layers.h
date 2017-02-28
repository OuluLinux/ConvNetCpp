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
	ConvLayer(ValueMap values) {Load(values);}
	
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
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};

class DropOutLayer : public LayerBase {
	
	Vector<bool> dropped;
	
protected:
	DropOutLayer(const DropOutLayer& o) {}
	
public:
	double drop_prob;
	
	DropOutLayer(double drop_prob);
	DropOutLayer(ValueMap values) {Load(values);}
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "dropout";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};



class FullyConnLayer : public LayerBase, public IDotProductLayer {
	
	int input_count;
	
protected:
	FullyConnLayer(const FullyConnLayer& o) {Panic("Not allowed");}
	
public:
	FullyConnLayer(int neuron_count);
	FullyConnLayer(ValueMap values) {Load(values);}
	
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
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};

class InputLayer : public LayerBase {
	
protected:
	InputLayer(const InputLayer& o) {}
	
public:
	InputLayer(int input_width, int input_height, int input_depth);
	InputLayer(ValueMap values) {Load(values);}
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual Volume& Forward(bool is_training);
	virtual String GetKey() const {return "input";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
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
	MaxoutLayer(ValueMap values) {Load(values);}
	
	int group_size;
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "maxout";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};

class PoolLayer : public LayerBase {
	Vector<int> switchx;
	Vector<int> switchy;
	
protected:
	PoolLayer(const PoolLayer& o) {}
	
public:
	PoolLayer(int width, int height);
	PoolLayer(ValueMap values) {Load(values);}
	
	int width;
	int height;
	int stride;
	int pad;
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	void UpdateOutputSize();
	virtual String GetKey() const {return "pool";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};


// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.
class RegressionLayer : public LastLayerBase {
	
protected:
	RegressionLayer(const RegressionLayer& o) {}
	
public:
	RegressionLayer();
	RegressionLayer(ValueMap values) {Load(values);}
	
	virtual double Backward(int pos, double y);
	virtual double Backward(const Vector<double>& y);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "regression";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};


// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
class ReluLayer : public LayerBase {
	
protected:
	ReluLayer(const ReluLayer& o) {}
	
public:
	ReluLayer();
	ReluLayer(ValueMap values) {Load(values);}
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "relu";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};


// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.
class SigmoidLayer : public LayerBase {
	
protected:
	SigmoidLayer(const SigmoidLayer& o) {}
	
public:
	SigmoidLayer();
	SigmoidLayer(ValueMap values) {Load(values);}
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "sigmoid";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
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
	SoftmaxLayer(ValueMap values) {Load(values);}
	
	virtual double Backward(int pos, double y);
	virtual double Backward(const Vector<double>& y);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "softmax";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};


class SvmLayer : public LastLayerBase, public IClassificationLayer {
	
protected:
	SvmLayer(const SvmLayer& o) {}
	
public:
	SvmLayer(int class_count);
	SvmLayer(ValueMap values) {Load(values);}
	
	virtual double Backward(int pos, double y);
	virtual double Backward(const Vector<double>& y);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "svm";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};


class TanhLayer : public LayerBase {
	
protected:
	TanhLayer(const TanhLayer& o) {}
	
public:
	TanhLayer();
	TanhLayer(ValueMap values) {Load(values);}
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "tanh";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};


// Local Response Normalization Layer
class LrnLayer : public LayerBase {
	
public:
	LrnLayer();
	LrnLayer(ValueMap values) {Load(values);}
	
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "lrn";}
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
};

}

#endif
