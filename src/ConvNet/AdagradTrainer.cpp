#include "Training.h"

namespace ConvNet {

void TrainerBase::TrainImplemAdagrad() {
	
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
				
				vol.SetGradient(j, 0.0); // zero out gradient so that we can begin accumulating anew
			}
		}
	}
	
	// in future, TODO: have to completely redo the way loss is done around the network as currently
	// loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
	// and it should all be computed correctly and automatically.
}

String TrainerBase::ToStringAdagrad() const {
	return Format("Adagrad: batch_size:%d, cost_loss:%2!,n, cost_reward:%2!,n, Beta1:%2!,n, Beta2:%2!,n,"
		"l1_decay:%2!,n, l2_decay:%2!,n, l1_decay_loss:%2!,n, l2_decay_loss:%2!,n, learning_rate:%2!,n, momentum:%2!,n, eps:%2!,n, ro:%2!,n",
		batch_size, cost_loss, cost_reward, Beta1, Beta2,
		l1_decay, l2_decay, l1_decay_loss, l2_decay_loss, learning_rate, momentum, eps, ro);
}

}
