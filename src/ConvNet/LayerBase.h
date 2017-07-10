#ifndef _ConvNet_LayerBase_h_
#define _ConvNet_LayerBase_h_

#include "Utilities.h"

namespace ConvNet {
	
class LayerBase {
	

protected:
	LayerBase(const LayerBase& o) {}
	
	Vector<ParametersAndGradients> response;
	
public:
	Volume* input_activation;
	Volume output_activation;
	int output_depth;
	int output_width;
	int output_height;
	int input_depth;
	int input_width;
	int input_height;
	
	LayerBase();
	virtual ~LayerBase();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual Volume& Forward(bool is_training);
	virtual void Backward() = 0;
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual Vector<ParametersAndGradients>& GetParametersAndGradients();
	virtual String GetKey() const {return "base";}
	
	virtual void Store(ValueMap& map) const {}
	virtual void Load(const ValueMap& map) {}
	virtual String ToString() const = 0;
	
	
	void Reset() {Init(input_width, input_height, input_depth);}
	
};


typedef LayerBase* LayerBasePtr;


class LastLayerBase : public LayerBase {
	
protected:
	LastLayerBase(const LastLayerBase& o) {}
	
public:
	LastLayerBase() {}
	
	virtual double Backward(int pos, double y) = 0;
	virtual double Backward(const VolumeDataBase& y) = 0;
	virtual double Backward(int cols, const Vector<int>& pos, const Vector<double>& y) = 0;
	virtual String GetKey() const {return "lastlayerbase";}
	
};


}

#endif
