#include "Training.h"


namespace ConvNet {
	
	
TrainerBase::TrainerBase() {
	batch_size = 1;
	iter_count = 0;
	cost_loss = 0;
	cost_reward = 0;
	
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

void TrainerBase::Serialize(Stream& s) {
	s % iter_count
	  % gsum
	  % xsum
	  % trainer_type
	  % batch_size
	  % cost_loss
	  % cost_reward
	  % Beta1
	  % Beta2
	  % l1_decay
	  % l2_decay
	  % l2_decay_loss
	  % l1_decay_loss
	  % learning_rate
	  % momentum
	  % eps
	  % ro;
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

void TrainerBase::Train(Volume& x, int cols, const Vector<int>& pos, const Vector<double>& y) {
	vec.SetCount(1);
	vec[0] = &x;
	Forward(vec);
	
	Backward(cols, pos, y);
	
	TrainImplem();
}

void TrainerBase::Backward(int pos, double y) {
	cost_reward = y;
	cost_loss = net->Backward(pos, y);
	
	l2_decay_loss = 0.0;
	l1_decay_loss = 0.0;
}

void TrainerBase::Backward(int cols, const Vector<int>& pos, const Vector<double>& y) {
	double sum = 0;
	for(int i = 0; i < y.GetCount(); i++)
		sum += y[i];
	cost_reward = sum;
	
	cost_loss = net->Backward(cols, pos, y);
	
	l2_decay_loss = 0.0;
	l1_decay_loss = 0.0;
}

void TrainerBase::Backward(const Vector<double>& y) {
	cost_loss = net->Backward(y);
	
	l2_decay_loss = 0.0;
	l1_decay_loss = 0.0;
}

void TrainerBase::Forward(const Vector<VolumePtr>& x) {
	net->Forward(x, true); // also set the flag that lets the net know we're just training
}

void TrainerBase::Reset() {
	iter_count = 0;
	
	gsum.Clear();
	xsum.Clear();
}

void TrainerBase::TrainImplem() {
	switch (trainer_type) {
		case TRAINER_NULL:			Panic("Trainer not set"); return;
		case TRAINER_ADADELTA:		TrainImplemAdadelta(); return;
		case TRAINER_ADAGRAD:		TrainImplemAdagrad(); return;
		case TRAINER_ADAM:			TrainImplemAdam(); return;
		case TRAINER_NETSTEROV:		TrainImplemNetsterov(); return;
		case TRAINER_SGD:			TrainImplemSgd(); return;
		case TRAINER_WINDOWGRAD:	TrainImplemWindowgrad(); return;
		default: Panic("Invalid trainer type");
	}
}

String TrainerBase::ToString() const {
	switch (trainer_type) {
		case TRAINER_NULL:			Panic("Trainer not set");
		case TRAINER_ADADELTA:		return ToStringAdadelta();
		case TRAINER_ADAGRAD:		return ToStringAdagrad();
		case TRAINER_ADAM:			return ToStringAdam();
		case TRAINER_NETSTEROV:		return ToStringNetsterov();
		case TRAINER_SGD:			return ToStringSgd();
		case TRAINER_WINDOWGRAD:	return ToStringWindowgrad();
		default: Panic("Invalid trainer type");
	}
	throw Exc("Never");
}

String TrainerBase::GetKey() const {
	switch (trainer_type) {
		case TRAINER_NULL:			Panic("Trainer not set"); break;
		case TRAINER_ADADELTA:		return "adadelta";
		case TRAINER_ADAGRAD:		return "adagrad";
		case TRAINER_ADAM:			return "adam";
		case TRAINER_NETSTEROV:		return "netsterov";
		case TRAINER_SGD:			return "sgd";
		case TRAINER_WINDOWGRAD:	return "windowgrad";
		default: Panic("Invalid trainer type");
	}
	throw Exc("Never");
}

}
