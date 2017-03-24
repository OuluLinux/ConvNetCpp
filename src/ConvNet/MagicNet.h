#ifndef _ConvNet_MagicNet_h_
#define _ConvNet_MagicNet_h_

#include "MetaSession.h"

namespace ConvNet {

/*
	A MagicNet takes data: a list of convnetjs.Vol(), and labels
	which for now are assumed to be class indeces 0..K. MagicNet then:
	- creates data folds for cross-validation
	- samples candidate networks
	- evaluates candidate networks on all data folds
	- produces predictions by model-averaging the best networks
*/

class MagicNet : public MetaSession {
	
protected:
	
	// Data-positions in random orders
	Vector<Vector<int> > train_folds, test_folds;
	
	// history of all candidates that were fully evaluated on all folds
	Array<Session> evaluated_candidates;
	
	double train_ratio;
	int num_folds;
	int num_candidates;
	int num_epochs;
	int ensemble_size;
	int foldix, datapos;
	
	double l2_decay_min, l2_decay_max;
	double learning_rate_min, learning_rate_max;
	double momentum_min, momentum_max;
	int batch_size_min, batch_size_max;
	int neurons_min, neurons_max;
	
	
	struct CandidateSorter {
		bool operator() (const Session& a, const Session& b) const {
			return a.accuracy_result_window.GetAverage() > b.accuracy_result_window.GetAverage();
		}
	};
	
	// tmp
	Vector<double> val_acc;
	
	
public:
	typedef MagicNet CLASSNAME;
	MagicNet();
	
	void SetTrainingRatio(double p) {train_ratio = p;}
	void SetFoldsCount(int i) {num_folds = i;}
	void SetCandidateCount(int i) {num_candidates = i;}
	void SetEpochCount(int i) {num_epochs = i;}
	void SetNeuronRange(int min, int max) {neurons_min = min, neurons_max = max;}
	
	// sets folds to a sampling of num_folds folds
	void SampleFolds();
	
	// returns a random candidate network
	void SampleCandidate(Session& cand);
	
	// sets candidates with num_candidates candidate nets
	void SampleCandidates();
	
	void Step();
	
	void EvaluateValueErrors(Vector<double>& vals);
	
	// returns prediction scores for given test data point, as Vol
	// uses an averaged prediction from the best ensemble_size models
	// x is a Vol.
	void PredictSoft(Volume& in, Volume& out);
	void Predict(Volume& in);
	int PredictSoftLabel(Volume& in);
	
	
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
	
	
	Session& GetEvaluatedCandidate(int i) {return evaluated_candidates[i];}
	int GetEvaluatedCandidateCount() const {return evaluated_candidates.GetCount();}
	
	int GetFold() const {return foldix;}
	
};




}

#endif
