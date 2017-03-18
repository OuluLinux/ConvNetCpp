#ifndef _ConvNet_Session_h_
#define _ConvNet_Session_h_

#include "SessionData.h"

namespace ConvNet {


class Session {
	
protected:
	friend class MagicNet;
	friend class MetaSession;
	
	typedef Exc RequiredArg;
	
	SessionData owned_data;
	SessionData* used_data;
	
	// Statistical variables for values during training
	Window loss_window, reward_window, l1_loss_window, l2_loss_window, train_window, accuracy_window, test_window;
	
	// Statistical variables for values for training results
	Window accuracy_result_window;
	
	TrainerBasePtr owned_trainer, trainer;
	SpinLock lock;
	Net net;
	TimeStop ts;
	Volume x;
	Vector<double> session_last_input_array;
	Vector<LayerBasePtr> owned_layers;
	int predict_interval, step_num;
	int train_iter_limit;
	int iter;
	int forward_time, backward_time;
	int step_cb_interal;
	int iter_cb_interal;
	int augmentation;
	bool is_training, is_training_stopped;
	bool test_predict;
	bool augmentation_do_flip;
	
	const Value& ChkNotNull(const String& key, const Value& v);
	void Train();
	
public:
	typedef Session CLASSNAME;
	Session();
	~Session();
	
	virtual const Vector<double>& GetLastInput() const {return session_last_input_array;}
	
	void StartTraining();
	void StopTraining();
	void TrainBegin();
	void TrainIteration();
	void TrainEnd();
	void Enter() {lock.Enter();}
	void Leave() {lock.Leave();}
	void ClearOwnedLayers();
	void ClearOwnedTrainer();
	void Clear();
	void ClearData();
	void Reset();
	void ResetTraining();
	
	SessionData& GetData()	{return *used_data;}
	SessionData& Data()		{return *used_data;}
	
	FullyConnLayer&		AddFullyConnLayer(int neuron_count, double l1_decay_mul=0.0, double l2_decay_mul=1.0, double bias_pref=0.0);
	LrnLayer&			AddLrnLayer();
	DropOutLayer&		AddDropoutLayer(double drop_prob);
	InputLayer&			AddInputLayer(int input_width, int input_height, int input_depth);
	SoftmaxLayer&		AddSoftmaxLayer(int class_count);
	RegressionLayer&	AddRegressionLayer();
	ConvLayer&			AddConvLayer(int width, int height, int filter_count, double l1_decay_mul=0.0, double l2_decay_mul=1.0, int stride=1, int pad=0, double bias_pref=0.0);
	PoolLayer&			AddPoolLayer(int width, int height, int stride=2, int pad=0);
	ReluLayer&			AddReluLayer();
	SigmoidLayer&		AddSigmoidLayer();
	TanhLayer&			AddTanhLayer();
	MaxoutLayer&		AddMaxoutLayer(int group_size);
	SvmLayer&			AddSVMLayer(int class_count);
	
	template <class T>
	T& LoadLayer(ValueMap values) {
		T* t = new T(values);
		owned_layers.Add(t);
		net.AddLayerPointer(*t);
		return *t;
	}
	
	Net& GetNetwork();
	InputLayer* GetInput() const;
	TrainerBase* GetTrainer() const {return trainer;}
	const Window& GetLossWindow() const {return loss_window;}
	const Window& GetAccuracyWindow() const {return accuracy_window;}
	const Window& GetTrainingWindow() const {return train_window;}
	const Window& GetTestingAccuracyWindow() const {return test_window;}
	double GetL1DecayLossAverage() const {return l1_loss_window.GetAverage();}
	double GetL2DecayLossAverage() const {return l2_loss_window.GetAverage();}
	double GetTrainingAccuracyAverage() const {return accuracy_window.GetAverage();}
	double GetValidationAccuracyAverage() const {return accuracy_window.GetAverage();}
	int GetForwardTime() const {return forward_time;}
	int GetBackwardTime() const {return backward_time;}
	int GetStepCount() const {return step_num;}
	int GetIteration() const {return iter;}
	bool IsTraining() const {return !is_training_stopped || is_training;}
	
	virtual double GetLossAverage() const {return loss_window.GetAverage();}
	virtual double GetRewardAverage() const {return reward_window.GetAverage();}
	
	bool MakeLayers(const String& json);
	bool LoadJSON(const String& json);
	bool StoreJSON(String& json);
	void SetMaxTrainIters(int count) {train_iter_limit = count;}
	void SetPredictInterval(int i) {predict_interval = i;}
	void SetTestPredict(bool b) {test_predict = b;}
	void SetAugmentation(int i=0, bool flip=false) {augmentation = 0; augmentation_do_flip = flip;}
	Session& SetTrainer(TrainerBase& trainer) {this->trainer = &trainer; return *this;}
	Session& AttachTrainer(TrainerBase* trainer) {ASSERT(!owned_trainer); this->trainer = trainer; owned_trainer = trainer; return *this;}
	Session& SetWindowSize(int size, int min_size=1);
	
	Callback WhenSessionLoaded;
	Callback1<int> WhenStepInterval, WhenIterationInterval;
	
	
	
};

}

#endif
