#include "Training.h"


namespace ConvNet {
	
	
TrainerBase::TrainerBase(Net& net) {
	this->net = &net;
	batch_size = 1;
	iter_count = 0;
	cost_loss = 0;
	
	Beta1 = 0;
	Beta2 = 0;
	l1_decay = 0;
	l2_decay = 0;
	l2_decay_loss = 0;
	l1_decay_loss = 0;
	learning_rate = 0.01;
	momentum = 0.9;
	eps = 1e-6;
	ro = 0.95;
	
}

void TrainerBase::Train(Volume& x, int pos, double y) {
	vec.SetCount(1);
	vec[0] = &x;
	Forward(vec);
	
	Backward(pos, y);
	
	TrainImplem();
}

void TrainerBase::Train(Volume& x, const Vector<double>& y) {
	vec.SetCount(1);
	vec[0] = &x;
	Forward(vec);
	
	Backward(y);
	
	TrainImplem();
}

void TrainerBase::Train(const Vector<double>& y, const Vector<VolumePtr>& x) {
	Forward(x);
	
	Backward(y);
	
	TrainImplem();
}

void TrainerBase::Backward(int pos, double y) {
	cost_loss = net->Backward(pos, y);
}

void TrainerBase::Backward(const Vector<double>& y) {
	cost_loss = net->Backward(y);
}

void TrainerBase::Forward(const Vector<VolumePtr>& x) {
	net->Forward(x, true); // also set the flag that lets the net know we're just training
}

}
