#ifndef _ConvNet_Net_h_
#define _ConvNet_Net_h_

#include "LayerBase.h"

namespace ConvNet
{

class Net {
	
private:
	Vector<LayerBase> layers;
	
	Vector<ParametersAndGradients> response;
	SpinLock lock;
	
protected:
	friend class Session;
	Net(const Net& iv) {}
	
public:
	Net() {}
	
	void Serialize(Stream& s) {
		s % layers;
	}
	
	const Vector<LayerBase>& GetLayers() const {return layers;}
	Vector<LayerBase>& GetLayers() {return layers;}
	Volume& GetOutput() {return layers.Top().output_activation;}
	
	LayerBase& AddLayer() {return layers.Add();}
	void CheckLayer();
	Volume& Forward(const Vector<VolumePtr>& inputs, bool is_training = false);
	Volume& Forward(Volume& input, bool is_training = false);
	double GetCostLoss(Volume& input, int pos, double y);
	double GetCostLoss(Volume& input, const Vector<double>& y);
	double Backward(int pos, double y);
	double Backward(const Vector<double>& y);
	double Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	int GetPrediction();
	Vector<ParametersAndGradients>& GetParametersAndGradients();
	
	void Clear() {layers.Clear();}
	void Enter() {lock.Enter();}
	void Leave() {lock.Leave();}
	
	String ToString() const;
	
};


}

#endif
