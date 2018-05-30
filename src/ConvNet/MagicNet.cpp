#include "ConvNet.h"

namespace ConvNet {

MagicNet::MagicNet() {
	
	// optional inputs
	train_ratio = 0.7;
	num_folds = 10;
	num_candidates = 50; // we evaluate several in parallel
	
	// how many epochs of data to train every network? for every fold?
	// higher values mean higher accuracy in final results, but more expensive
	num_epochs = 50;
	
	// number of best models to average during prediction. Usually higher = better
	ensemble_size = 10;
	
	// candidate parameters
	batch_size_min = 10;
	batch_size_max = 300;
	l2_decay_min = -4;
	l2_decay_max = 2;
	learning_rate_min = -4;
	learning_rate_max = 0;
	momentum_min = 0.9;
	momentum_max = 0.9;
	neurons_min = 5;
	neurons_max = 30;
	
	foldix = 0;
	datapos = 0;
}

void MagicNet::SampleFolds() {
	SessionData& d = data[0];
	int N = d.GetDataCount();
	int num_train = floor(train_ratio * N);
	int num_test = N - num_train;
	
	train_folds.SetCount(num_folds);
	test_folds.SetCount(num_folds);
	
	Vector<int> p;
	for (int i = 0; i < num_folds; i++) {
		RandomPermutation(N, p);
		
		Vector<int>& train = train_folds[i];
		Vector<int>& test  =  test_folds[i];
		
		train.SetCount(num_train);
		test.SetCount(num_test);
		
		for(int j = 0; j < p.GetCount(); j++) {
			if (j < num_train)
				train[j] = p[j];
			else
				test[j - num_train] = p[j];
		}
	}
}

void MagicNet::SampleCandidate(Session& cand) {
	ASSERT(cand.GetNetwork().GetLayers().IsEmpty());
	
	SessionData& d = data[0];
	
	int input_depth = d.Get(0).GetCount();
	int num_classes = d.GetClassCount();
	
	// sample network topology and hyperparameters
	cand.AddInputLayer(1, 1, input_depth);
	
	//var nl = weightedSample([0,1,2,3], [0.2, 0.3, 0.3, 0.2]); // prefer nets with 1,2 hidden layers
	int nl = 1 + Random(3);
	
	for (int q = 0; q < nl; q++) {
		int ni = neurons_min + Random(neurons_max - neurons_min);
		int act = Random(3); // tanh, maxout, relu
		
		double bias_pref = act == 2 ? 0.1 : 0.0; // 0.1 for relu
		
		LayerBase& fc = cand.AddFullyConnLayer(ni);
		fc.bias_pref = bias_pref;
		
		if (act == 0) {
			cand.AddTanhLayer();
		}
		else if (act == 1) {
			cand.AddMaxoutLayer(2); // 2 is default
		}
		else if (act == 2) {
			cand.AddReluLayer();
		}
		else Panic("What activation");
		
		if (Randomf() < 0.5) {
			cand.AddDropoutLayer(Randomf());
		}
	}
	
	cand.AddFullyConnLayer(num_classes);
	cand.AddSoftmaxLayer(num_classes);
	
	
	// sample training hyperparameters
	int bs = batch_size_min + Random(batch_size_max - batch_size_min); // batch size
	double l2 = pow(10, l2_decay_min + Randomf() + (l2_decay_max - l2_decay_min)); // l2 weight decay
	double lr = pow(10, learning_rate_min + Randomf() * (learning_rate_max - learning_rate_min)); // learning rate
	double mom = momentum_min + Randomf() * (momentum_max - momentum_min); // momentum. Lets just use 0.9, works okay usually ;p
	double tp = Randomf(); // trainer type
	
	
	// add trainer
	TrainerBase& trainer = cand.GetTrainer();;
	if (tp < 1.0/3.0) {
		trainer.SetType(TRAINER_ADADELTA);
	}
	else if (tp < 2.0/3.0) {
		trainer.SetType(TRAINER_ADAGRAD);
	}
	else {
		trainer.SetType(TRAINER_SGD);
	}
	trainer.SetBatchSize(bs);
	trainer.SetLearningRate(lr);
	trainer.SetL2Decay(l2);
	trainer.SetMomentum(mom);
	
	foldix = 0;
	datapos = 0;
}

void MagicNet::SampleCandidates() {
	ClearSessions();
	session.SetCount(num_candidates);
	for (int i = 0; i < num_candidates; i++) {
		Session& ses = session[i];
		SampleCandidate(ses);
	}
}

void MagicNet::Step() {
	SessionData& data = this->data[0];
	
	// run an example through current candidate
	iter++;
	total_iter++;
	
	// step all candidates on a random data point
	Vector<int>& fold = train_folds[foldix]; // active fold
	int dataix = fold[datapos];
	datapos++;
	if (datapos >= fold.GetCount()) datapos = 0;
	
	tmp_in.Init(data.GetDataWidth(), data.GetDataHeight(), data.GetDataDepth(), 0);
	Vector<double>& x = data.Get(datapos);
	tmp_in.SetData(x);
	int l = data.GetLabel(datapos);
	
	for (int k = 0; k < session.GetCount(); k++) {
		session[k].GetTrainer().Train(tmp_in, l, 1.0);
	}
	
	if ((total_iter % 100) == 0) {
		EvaluateValueErrors(val_acc);
		for (int k = 0; k < session.GetCount(); k++) {
			Session& c = session[k];
			c.accuracy_window.Add(val_acc[k]);
		}
		WhenStepInterval(total_iter);
	}
	
	// process consequences: sample new folds, or candidates
	int lastiter = num_epochs * fold.GetCount();
	if (iter >= lastiter) {
		// finished evaluation of this fold. Get final validation
		// accuracies, record them, and go on to next fold.
		EvaluateValueErrors(val_acc);
		for (int k = 0; k < session.GetCount(); k++) {
			Session& c = session[k];
			c.accuracy_result_window.Add(val_acc[k]);
		}
		iter = 0; // reset step number
		foldix++; // increment fold
		datapos = 0;
		
		WhenFinishFold();
		
		if (foldix >= train_folds.GetCount()) {
			// we finished all folds as well! Record these candidates
			// and sample new ones to evaluate.
			for (int k = 0; k < session.GetCount(); k++) {
				evaluated_candidates.Add(session.Detach(k));
			}
			// sort evaluated candidates according to accuracy achieved
			Sort(evaluated_candidates, CandidateSorter());
			
			// and clip only to the top few ones (lets place limit at 3*ensemble_size)
			// otherwise there are concerns with keeping these all in memory
			// if MagicNet is being evaluated for a very long time
			if (evaluated_candidates.GetCount() > 3 * ensemble_size) {
				evaluated_candidates.SetCount(3 * ensemble_size);
			}
			WhenFinishBatch();
			
			SampleCandidates(); // begin with new candidates
			foldix = 0; // reset this
		}
		else {
			// we will go on to another fold. reset all candidates nets
			for(int k = 0; k < session.GetCount(); k++) {
				session[k].Reset();
			}
		}
	}
}

void MagicNet::EvaluateValueErrors(Vector<double>& vals) {
	SessionData& d = data[0];
	tmp_in.Init(d.GetDataWidth(), d.GetDataHeight(), d.GetDataDepth(), 0);
	
	// evaluate candidates on validation data and return performance of current networks
	// as simple list
	vals.SetCount(session.GetCount());
	Vector<int>& fold = test_folds[foldix]; // active fold
	for(int k = 0; k < session.GetCount(); k++) {
		Net& net = session[k].GetNetwork();
		double v = 0.0;
		for (int q = 0; q < fold.GetCount(); q++) {
			Vector<double>& x = d.Get(fold[q]);
			tmp_in.SetData(x);
			int l = d.GetLabel(fold[q]);
			net.Forward(tmp_in);
			int yhat = net.GetPrediction();
			v += (yhat == l ? 1.0 : 0.0); // 0 1 loss
		}
		v /= fold.GetCount(); // normalize
		vals[k] = v;
	}
}

void MagicNet::PredictSoft(Volume& in, Volume& out) {
	// forward prop the best networks
	// and accumulate probabilities at last layer into a an output Vol
	
	Array<Session>* eval_candidates_;
	int nv = 0;
	if (evaluated_candidates.GetCount() == 0) {
		// not sure what to do here, first batch of nets hasnt evaluated yet
		// lets just predict with current candidates.
		nv = session.GetCount();
		eval_candidates_ = &session;
	} else {
		// forward prop the best networks from evaluated_candidates
		nv = min(ensemble_size, evaluated_candidates.GetCount());
		eval_candidates_ = &evaluated_candidates;
	}
	
	Array<Session>& eval_candidates = *eval_candidates_;
	
	// forward nets of all candidates and average the predictions
	int n;
	for (int j = 0; j < nv; j++) {
		Net& net = eval_candidates[j].GetNetwork();
		Volume& x = net.Forward(in);
		if (j == 0) {
			out = x;
			n = x.GetLength();
		} else {
			// add it on
			for(int d = 0; d < n; d++) {
				out.Add(d, x.Get(d));
			}
		}
	}
	// produce average
	for (int d = 0; d < n; d++) {
		out.Set(d, out.Get(d) / nv);
	}
	
}

int MagicNet::PredictSoftLabel(Volume& in) {
	PredictSoft(in, tmp_out);
	
	int predicted_label;
	
	if (tmp_out.GetLength() != 0) {
		predicted_label = tmp_out.GetMaxColumn();
	} else {
		predicted_label = -1; // error out
	}
	
	return predicted_label;
}


}
