#ifndef _ConvNet_Training_h_
#define _ConvNet_Training_h_

#include "Net.h"

namespace ConvNet {

#define IF_NULL_1(x) (x != NULL ? *x : 1.0)

class TrainerBase {
	
protected:
	friend class Session;
	friend class Brain;
	
	Net* net;
	int iter_count;
	Vector<VolumePtr> vec;
	
	// Previously public vars
	int batch_size;
	double cost_loss;
	double cost_reward;
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
	
	TrainerBase(Net& net);
	TrainerBase(const TrainerBase& o) {}
	TrainerBase() {}
	
public:
	virtual ~TrainerBase() {}
	
	// TODO: make these protected
	
	
	int GetBatchSize() const {return batch_size;}
	double GetCostLoss() const {return cost_loss;}
	double GetBeta1() const {return Beta1;}
	double GetBeta2() const {return Beta2;}
	double GetLearningRate() const {return learning_rate;}
	double GetMomentum() const {return momentum;}
	double GetEps() const {return eps;}
	double GetRo() const {return ro;}
	double GetL1Decay() const {return l1_decay;}
	double GetL2Decay() const {return l2_decay;}
	double GetL1DecayLoss() const {return l1_decay_loss;}
	double GetL2DecayLoss() const {return l2_decay_loss;}
	virtual double GetLoss() {return cost_loss;}
	virtual double GetReward() {return cost_reward;}
	
	
	TrainerBase& SetBatchSize(int i) {batch_size = i; return *this;}
	TrainerBase& SetCostLoss(double d) {cost_loss = d; return *this;}
	TrainerBase& SetBeta1(double d) {Beta1 = d; return *this;}
	TrainerBase& SetBeta2(double d) {Beta2 = d; return *this;}
	TrainerBase& SetLearningRate(double d) {learning_rate = d; return *this;}
	TrainerBase& SetMomentum(double d) {momentum = d; return *this;}
	TrainerBase& SetEps(double d) {eps = d; return *this;}
	TrainerBase& SetRo(double d) {ro = d; return *this;}
	TrainerBase& SetL1Decay(double d) {l1_decay = d; return *this;}
	TrainerBase& SetL2Decay(double d) {l2_decay = d; return *this;}
	TrainerBase& SetL1DecayLoss(double d) {l1_decay_loss = d; return *this;}
	TrainerBase& SetL2DecayLoss(double d) {l2_decay_loss = d; return *this;}
	
	//void Train(Volume& x, double y) {Train(x, 0, y);}
	void Train(Volume& x, int pos, double y);
	void Train(double y, const Vector<VolumePtr>& x);
	void Train(Volume& x, const VolumeDataBase& y);
	void Train(const VolumeDataBase& y, const Vector<VolumePtr>& x);
	void Forward(const Vector<VolumePtr>& x);
	
protected:
	virtual void TrainImplem() = 0;
	
	virtual void Backward(int pos, double y);
	virtual void Backward(const VolumeDataBase& y);
	
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
	virtual void Backward(const VolumeDataBase& y);
	
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
	virtual void Backward(const VolumeDataBase& y);
	
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
	virtual void Backward(const VolumeDataBase& y);
	
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
	virtual void Backward(const VolumeDataBase& y);
	
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
	virtual void Backward(const VolumeDataBase& y);
	
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
	virtual void Backward(const VolumeDataBase& y);
	
};

}

#endif
