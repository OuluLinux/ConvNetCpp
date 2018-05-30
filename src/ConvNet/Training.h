#ifndef _ConvNet_Training_h_
#define _ConvNet_Training_h_

#include "Net.h"

namespace ConvNet {

#define IF_NULL_1(x) (x != NULL ? *x : 1.0)

enum {TRAINER_NULL, TRAINER_ADADELTA, TRAINER_ADAGRAD, TRAINER_ADAM, TRAINER_NETSTEROV, TRAINER_SGD, TRAINER_WINDOWGRAD};

class TrainerBase {
	
protected:
	friend class Session;
	friend class Brain;
	
	Net* net = NULL;
	Vector<VolumePtr> vec;
	
	int iter_count;
	Vector<Vector<double> > gsum;
	Vector<Vector<double> > xsum;
	int trainer_type = TRAINER_NULL;
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
	
	
public:
	TrainerBase();
	~TrainerBase() {}
	
	void Serialize(Stream& s);
	
	int GetType() const {return trainer_type;}
	int GetIteration() const {return iter_count;}
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
	double GetLoss() {return cost_loss;}
	double GetReward() {return cost_reward;}
	
	String ToString() const;
	String GetKey() const;
	
	
	TrainerBase& SetNet(Net& net) {this->net = &net; return *this;}
	TrainerBase& SetType(int i) {trainer_type = i; return *this;}
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
	
	void Train(Volume& x, int pos, double y);
	void Train(double y, const Vector<VolumePtr>& x);
	void Train(Volume& x, const Vector<double>& y);
	void Train(const Vector<double>& y, const Vector<VolumePtr>& x);
	void Train(Volume& x, int cols, const Vector<int>& pos, const Vector<double>& y);
	void Forward(const Vector<VolumePtr>& x);
	
	void TrainImplem();
	void Backward(int pos, double y);
	void Backward(const Vector<double>& y);
	void Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	void Reset();
	
	void TrainImplemAdadelta();
	void TrainImplemAdagrad();
	void TrainImplemAdam();
	void TrainImplemNetsterov();
	void TrainImplemSgd();
	void TrainImplemWindowgrad();
	String ToStringAdadelta() const;
	String ToStringAdagrad() const;
	String ToStringAdam() const;
	String ToStringNetsterov() const;
	String ToStringSgd() const;
	String ToStringWindowgrad() const;
};

typedef TrainerBase* TrainerBasePtr;

class AdadeltaTrainer : public TrainerBase {
	
protected:
	AdadeltaTrainer(const AdadeltaTrainer& o) {}
	
public:
	
	AdadeltaTrainer(Net& net);
	
protected:
	virtual void TrainImplem();
	virtual void Backward(int pos, double y);
	virtual void Backward(const Vector<double>& y);
	virtual void Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	virtual void Reset();
	virtual String ToString() const;
	virtual String GetKey() const {return "adadelta";}
	
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
	virtual void Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	virtual void Reset();
	virtual String ToString() const;
	virtual String GetKey() const {return "adagrad";}
	
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
	virtual void Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	virtual void Reset();
	virtual String ToString() const;
	virtual String GetKey() const {return "adam";}
	
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
	virtual void Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	virtual void Reset();
	virtual String ToString() const;
	virtual String GetKey() const {return "netsterov";}
	
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
	virtual void Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	virtual void Reset();
	virtual String ToString() const;
	virtual String GetKey() const {return "sgd";}
	
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
	virtual void Backward(int cols, const Vector<int>& pos, const Vector<double>& y);
	virtual void Reset();
	virtual String ToString() const;
	virtual String GetKey() const {return "windowgrad";}
	
};

}

#endif
