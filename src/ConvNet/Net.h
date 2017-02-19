#ifndef _ConvNet_Net_h_
#define _ConvNet_Net_h_

#include "LayerBase.h"

namespace ConvNet
{

class Net {
	
private:
	Vector<LayerBasePtr> layers;
	Vector<ParametersAndGradients> response;

protected:
	Net(const Net& iv) {}
		
public:
	Net() {}
	
	const Vector<LayerBasePtr>& GetLayers() {return layers;}
	
	virtual void AddLayer(LayerBase& layer);
	virtual Volume& Forward(const Vector<VolumePtr>& inputs, bool is_training = false);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual double GetCostLoss(Volume& input, double y);
	virtual double GetCostLoss(Volume& input, const Vector<double>& y);
	virtual double Backward(double y);
	virtual double Backward(const Vector<double>& y);
	virtual int GetPrediction();
	virtual Vector<ParametersAndGradients>& GetParametersAndGradients();
	
	void Clear() {layers.Clear();}
	
};


}

#endif
