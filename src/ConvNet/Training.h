#ifndef _ConvNet_Training_h_
#define _ConvNet_Training_h_

#include "Net.h"

namespace ConvNet {

#define IF_NULL_1(x) (x != NULL ? *x : 1.0)

class TrainerBase {
	
protected:
	Net* net;
	int iter_count;
	Vector<VolumePtr> vec;
	
	TrainerBase(Net& net);
	TrainerBase(const TrainerBase& o) {}
	TrainerBase() {}
	
public:
	virtual ~TrainerBase() {}
	
	double cost_loss;
	int batch_size;
	
	double Beta1;
	double Beta2;
	double l1_decay;
	double l2_decay;
	double l2_decay_loss;
	double l1_decay_loss;
	double learning_rate;
	double momentum;
	double eps;
	double ro;
	
	virtual double GetLoss() {return cost_loss;}
	
	//void Train(Volume& x, double y) {Train(x, 0, y);}
	void Train(Volume& x, int pos, double y);
	void Train(double y, const Vector<VolumePtr>& x);
	void Train(Volume& x, const Vector<double>& y);
	void Train(const Vector<double>& y, const Vector<VolumePtr>& x);
	void Forward(const Vector<VolumePtr>& x);
	
protected:
	virtual void TrainImplem() = 0;
	
	virtual void Backward(int pos, double y);
	virtual void Backward(const Vector<double>& y);
	
};

typedef TrainerBase* TrainerBasePtr;

class AdadeltaTrainer : public TrainerBase {
	Vector<Vector<double> > gsum;
	Vector<Vector<double> > xsum;
	
protected:
	AdadeltaTrainer(const AdadeltaTrainer& o) {}
	
public:
	
	AdadeltaTrainer(Net& net);
	
protected:
	virtual void TrainImplem();
	virtual void Backward(int pos, double y);
	virtual void Backward(const Vector<double>& y);
	
};


class AdagradTrainer : public TrainerBase {
	Vector<Vector<double> > gsum;
	
protected:
	AdagradTrainer(const AdagradTrainer& o) {}
	
public:

	AdagradTrainer(Net& net);
	
protected:
	virtual void TrainImplem();
	virtual void Backward(int pos, double y);
	virtual void Backward(const Vector<double>& y);
	
};


class AdamTrainer : public TrainerBase {
	Vector<Vector<double> > gsum;
	Vector<Vector<double> > xsum;
	
protected:
	AdamTrainer(const AdamTrainer& o) {}
	
public:
	
	AdamTrainer(Net& net);
	
protected:

	virtual void TrainImplem();
	virtual void Backward(int pos, double y);
	virtual void Backward(const Vector<double>& y);
	
};


class NetsterovTrainer : public TrainerBase {
	Vector<Vector<double> > gsum;
	
protected:
	NetsterovTrainer(const NetsterovTrainer& o) {}
	
public:
	
	NetsterovTrainer(Net& net);
	
protected:
	virtual void TrainImplem();
	virtual void Backward(int pos, double y);
	virtual void Backward(const Vector<double>& y);
	
};



// Stochastic gradient descent
class SgdTrainer : public TrainerBase {
	Vector<Vector<double> > gsum;
	
protected:
	SgdTrainer(const SgdTrainer& o) {}
	
public:
	
	SgdTrainer(Net& net);
	
protected:
	virtual void TrainImplem();
	virtual void Backward(int pos, double y);
	virtual void Backward(const Vector<double>& y);
	
};





class WindowgradTrainer : public TrainerBase {
	Vector<Vector<double> > gsum;
	
protected:
	WindowgradTrainer(const WindowgradTrainer& o) {}
	
public:
	
	WindowgradTrainer(Net& net);
	
protected:
	
	virtual void TrainImplem();
	virtual void Backward(int pos, double y);
	virtual void Backward(const Vector<double>& y);
	
};

}

#endif
