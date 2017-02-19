#include "Training.h"


namespace ConvNet {


AdagradTrainer::AdagradTrainer(Net& net) : TrainerBase(net) {
	learning_rate = 0.01;
	eps = 1e-6;
	l1_decay = 0;
	l2_decay = 0;
	l2_decay_loss = 0;
	l1_decay_loss = 0;
}

void AdagradTrainer::TrainImplem() {
	
	iter_count++;
	
	if ((iter_count % batch_size) == 0) {
		Vector<ParametersAndGradients>& parametersAndGradients = net->GetParametersAndGradients();
		
		// initialize lists for accumulators. Will only be done once on first iteration
		if (gsum.IsEmpty()) {
			for(int i = 0; i < parametersAndGradients.GetCount(); i++) {
				ParametersAndGradients& t = parametersAndGradients[i];
				gsum.Add().SetCount(t.volume->GetLength(), 0.0);
			}
		}
		
		// perform an update for all sets of weights
		for (int i = 0; i < parametersAndGradients.GetCount(); i++) {
			ParametersAndGradients& parametersAndGradient = parametersAndGradients[i];
			
			// param, gradient, other options in future (custom learning rate etc)
			Volume& vol = *parametersAndGradient.volume;
			
			// learning rate for some parameters.
			double l2_decay_mul = IF_NULL_1(parametersAndGradient.l2_decay_mul);
			double l1_decay_mul = IF_NULL_1(parametersAndGradient.l1_decay_mul);
			double l2_decay = this->l2_decay * l2_decay_mul;
			double l1_decay = this->l1_decay * l1_decay_mul;
			
			int plen = vol.GetLength();
			
			for (int j = 0; j < plen; j++) {
				l2_decay_loss += l2_decay * vol.Get(j) * vol.Get(j) / 2; // accumulate weight decay loss
				l1_decay_loss += l1_decay * fabs(vol.Get(j));
				double l1_grad = l1_decay * (vol.Get(j) > 0 ? 1 : -1);
				double l2_grad = l2_decay * vol.Get(j);
				
				double gij = (l2_grad + l1_grad + vol.GetGradient(j)) / batch_size; // raw batch gradient
				
				Vector<double>& gsumi = gsum[i];
				
				// adagrad update
				gsumi[j] = gsumi[j] + gij * gij;
				double dx = -1.0 * learning_rate / sqrt(gsumi[j] + eps) * gij;
				vol.Set(j, vol.Get(j) + dx);
				
				vol.SetGradient(i, 0.0); // zero out gradient so that we can begin accumulating anew
			}
		}
	}
	
	// in future, TODO: have to completely redo the way loss is done around the network as currently
	// loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
	// and it should all be computed correctly and automatically.
}

void AdagradTrainer::Backward(double y) {
	TrainerBase::Backward(y);
	
	l2_decay_loss = 0.0;
	l1_decay_loss = 0.0;
}

void AdagradTrainer::Backward(const Vector<double>& y) {
	TrainerBase::Backward(y);
	
	l2_decay_loss = 0.0;
	l1_decay_loss = 0.0;
}

}
