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
	
};

//typedef Ptr<LayerBase> LayerBasePtr;
typedef LayerBase* LayerBasePtr;

}

#endif
