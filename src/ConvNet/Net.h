#ifndef _ConvNet_Net_h_
#define _ConvNet_Net_h_

#include "LayerBase.h"

namespace ConvNet
{

class Net {
	
private:
	Vector<LayerBasePtr> layers;
	Vector<ParametersAndGradients> response;
	SpinLock lock;
	
protected:
	friend class Session;
	Net(const Net& iv) {}
		
	void AddLayerPointer(LayerBase& layer) {layers.Add(&layer);}
public:
	Net() {}
	
	const Vector<LayerBasePtr>& GetLayers() const {return layers;}
	Volume& GetOutput() {return layers.Top()->output_activation;}
	
	virtual void AddLayer(LayerBase& layer);
	virtual Volume& Forward(const Vector<VolumePtr>& inputs, bool is_training = false);
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual double GetCostLoss(Volume& input, int pos, double y);
	virtual double GetCostLoss(Volume& input, const VolumeDataBase& y);
	virtual double Backward(int pos, double y);
	virtual double Backward(const VolumeDataBase& y);
	virtual double Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	virtual int GetPrediction();
	virtual Vector<ParametersAndGradients>& GetParametersAndGradients();
	
	void Clear() {layers.Clear();}
	void Enter() {lock.Enter();}
	void Leave() {lock.Leave();}
	
	String ToString() const;
	
};


}

#endif
