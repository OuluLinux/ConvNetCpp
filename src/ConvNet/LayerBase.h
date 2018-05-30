#ifndef _ConvNet_LayerBase_h_
#define _ConvNet_LayerBase_h_

#include "Utilities.h"


namespace ConvNet {

typedef Exc ArgumentException;
typedef Exc NotImplementedException;
typedef Exc Exception;

#define ASSERTEXC(x) if (!(x)) {throw Exception();}

enum {
	NULL_LAYER,
	FULLYCONN_LAYER,
	LRN_LAYER,
	DROPOUT_LAYER,
	INPUT_LAYER,
	SOFTMAX_LAYER,
	REGRESSION_LAYER,
	CONV_LAYER,
	POOL_LAYER,
	RELU_LAYER,
	SIGMOID_LAYER,
	TANH_LAYER,
	MAXOUT_LAYER,
	SVM_LAYER
};

class LayerBase : Moveable<LayerBase> {
	

protected:
	LayerBase(const LayerBase& o) {}
	
	// Temporary
	Vector<ParametersAndGradients> response;
	Volume* input_activation = NULL;
	
public:
	
	// Persistent
	Volume output_activation;
	int output_depth = 0;
	int output_width = 0;
	int output_height = 0;
	int input_depth = 0;
	int input_width = 0;
	int input_height = 0;
	int input_count = 0;
	int layer_type = NULL_LAYER;
	
	// ClassificationLayer
	int class_count = 0;
	
	// IDotProductLayer
	double bias_pref = 0.0;
	
	// Fully connected, Convolutive layer
	Volume biases;
	Vector<Volume> filters;
	double l1_decay_mul = 0.0;
	double l2_decay_mul = 1.0;
	int neuron_count = 0;
	
	// Local Response Normalization (LRN) Layer
	Volume S_cache;
	double k = 0, alpha = 0, beta = 0;
	int n = 0;
	
	// Dropout layer
	Vector<bool> dropped;
	double drop_prob = 0.0;
	
	// Softmax layer
	Vector<double> es;
	
	// Convolutive layer
	int width;
	int height;
	int filter_count;
	int stride;
	int pad;
	
	// Maxout layer
	Vector<int> switches;
	int group_size = 0;
	
	// Pool layer
	Vector<int> switchx;
	Vector<int> switchy;
	
	
	
	// Fully connected
	int GetInputCount() const {return input_count;}
	Volume& ForwardFullyConn(Volume& input, bool is_training = false);
	void BackwardFullyConn();
	void InitFullyConn(int input_width, int input_height, int input_depth);
	String ToStringFullyConn() const;
	
	// Local Response Normalization (LRN) Layer
	Volume& ForwardLrn(Volume& input, bool is_training = false);
	void BackwardLrn();
	void InitLrn(int input_width, int input_height, int input_depth);
	String ToStringLrn() const;
	
	// Dropout layer
	Volume& ForwardDropOut(Volume& input, bool is_training = false);
	void BackwardDropOut();
	void InitDropOut(int input_width, int input_height, int input_depth);
	String ToStringDropOut() const;
	
	// Input layer
	Volume& ForwardInput(Volume& input, bool is_training = false);
	Volume& ForwardInput(bool is_training = false);
	void BackwardInput();
	String ToStringInput() const;
	
	// Softmax layer
	Volume& ForwardSoftmax(Volume& input, bool is_training = false);
	double BackwardSoftmax(int pos, double y);
	void InitSoftmax(int input_width, int input_height, int input_depth);
	String ToStringSoftmax() const;
	
	// Regression layer
	Volume& ForwardRegression(Volume& input, bool is_training = false);
	double BackwardRegression(const Vector<double>& y);
	double BackwardRegression(int pos, double y);
	double BackwardRegression(int cols, const Vector<int>& posv, const Vector<double>& yv);
	void InitRegression(int input_width, int input_height, int input_depth);
	String ToStringRegression() const;
	
	// Convolutive layer
	Volume& ForwardConv(Volume& input, bool is_training = false);
	void BackwardConv();
	void InitConv(int input_width, int input_height, int input_depth);
	String ToStringConv() const;
	int GetStride() const {return stride;}
	int GetPad() const {return pad;}
	
	// Pool layer
	Volume& ForwardPool(Volume& input, bool is_training = false);
	void BackwardPool();
	void InitPool(int input_width, int input_height, int input_depth);
	String ToStringPool() const;
	
	// Relu layer
	Volume& ForwardRelu(Volume& input, bool is_training = false);
	void BackwardRelu();
	void InitRelu(int input_width, int input_height, int input_depth);
	String ToStringRelu() const;
	
	// Sigmoid layer
	Volume& ForwardSigmoid(Volume& input, bool is_training = false);
	void BackwardSigmoid();
	void InitSigmoid(int input_width, int input_height, int input_depth);
	String ToStringSigmoid() const;
	
	// Tanh layer
	Volume& ForwardTanh(Volume& input, bool is_training = false);
	void BackwardTanh();
	void InitTanh(int input_width, int input_height, int input_depth);
	String ToStringTanh() const;
	
	// Maxout layer
	Volume& ForwardMaxout(Volume& input, bool is_training = false);
	void BackwardMaxout();
	void InitMaxout(int input_width, int input_height, int input_depth);
	String ToStringMaxout() const;
	
	// SVM layer
	Volume& ForwardSVM(Volume& input, bool is_training = false);
	double BackwardSVM(int pos, double yd);
	void InitSVM(int input_width, int input_height, int input_depth);
	String ToStringSVM() const;
	
	
	LayerBase();
	~LayerBase();
	void Serialize(Stream& s);
	Volume& Forward(Volume& input, bool is_training = false);
	Volume& Forward(bool is_training);
	double Backward();
	double Backward(int pos, double y);
	double Backward(const Vector<double>& y);
	double Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	void Init(int input_width, int input_height, int input_depth);
	Vector<ParametersAndGradients>& GetParametersAndGradients();
	bool IsDotProductLayer() const {return layer_type == CONV_LAYER;}
	bool IsClassificationLayer() const {return layer_type == SOFTMAX_LAYER || layer_type == SVM_LAYER;}
	bool IsInputLayer() const {return layer_type == INPUT_LAYER;}
	bool IsFullyConnLayer() const {return layer_type == FULLYCONN_LAYER;}
	bool IsRegressionLayer() const {return layer_type == REGRESSION_LAYER;}
	bool IsReluLayer() const {return layer_type == RELU_LAYER;}
	bool IsSoftMaxLayer() const {return layer_type == SOFTMAX_LAYER;}
	bool IsLastLayer() const {return layer_type == REGRESSION_LAYER || layer_type == SOFTMAX_LAYER || layer_type == SVM_LAYER;}
	String ToString() const;
	String GetKey() const;
	
	void Reset() {Init(input_width, input_height, input_depth);}
	
};





}

#endif
