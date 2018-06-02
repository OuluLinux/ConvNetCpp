#ifndef _ConvNet_Session_h_
#define _ConvNet_Session_h_

#include "SessionData.h"

namespace ConvNet {


class Session {
	
public:
	// Statistical variables for values during training
	Window loss_window, reward_window, l1_loss_window, l2_loss_window, train_window, accuracy_window, test_window;
	
	// Statistical variables for values for training results
	Window accuracy_result_window;
	
protected:
	friend class MagicNet;
	friend class MetaSession;
	
	typedef Exc RequiredArg;
	
	
	// Persistent
	SessionData owned_data;
	TrainerBase trainer;
	Net net;
	Volume x;
	Vector<double> session_last_input_array;
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
	
	
	// Temp vars
	TimeStop ts;
	SessionData* used_data = NULL;
	SpinLock lock;
	
	const Value& ChkNotNull(const String& key, const Value& v);
	void Train();
	
public:
	typedef Session CLASSNAME;
	Session();
	~Session();
	
	void CopyFrom(Session& session);
	
	virtual const Vector<double>& GetLastInput() const {return session_last_input_array;}
	
	void StartTraining();
	void StopTraining();
	void TrainBegin();
	void TrainIteration();
	void TrainEnd();
	void TrainOnce(Volume& x, const Vector<double>& y);
	void Enter() {lock.Enter();}
	void Leave() {lock.Leave();}
	void ClearLayers();
	void Clear();
	void ClearData();
	void Reset();
	void ResetTraining();
	
	SessionData& GetData()	{return *used_data;}
	SessionData& Data()		{return *used_data;}
	
	LayerBase&			AddFullyConnLayer(int neuron_count, double l1_decay_mul=0.0, double l2_decay_mul=1.0, double bias_pref=0.0);
	LayerBase&			AddLrnLayer(double k, int n, double alpha, double beta);
	LayerBase&			AddDropoutLayer(double drop_prob);
	LayerBase&			AddInputLayer(int input_width, int input_height, int input_depth);
	LayerBase&			AddSoftmaxLayer(int class_count);
	LayerBase&			AddRegressionLayer();
	LayerBase&			AddConvLayer(int width, int height, int filter_count, double l1_decay_mul=0.0, double l2_decay_mul=1.0, int stride=1, int pad=0, double bias_pref=0.0);
	LayerBase&			AddDeconvLayer(int width, int height, int filter_count, double l1_decay_mul=0.0, double l2_decay_mul=1.0, int stride=1, int pad=0, double bias_pref=0.0);
	LayerBase&			AddPoolLayer(int width, int height, int stride=2, int pad=0);
	LayerBase&			AddUnpoolLayer(int width, int height, int stride=2, int pad=0);
	LayerBase&			AddReluLayer();
	LayerBase&			AddSigmoidLayer();
	LayerBase&			AddTanhLayer();
	LayerBase&			AddMaxoutLayer(int group_size);
	LayerBase&			AddSVMLayer(int class_count);
	
	
	Net& GetNetwork();
	LayerBase* GetInput();
	TrainerBase& GetTrainer() {return trainer;}
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
	void Serialize(Stream& s);
	void Xmlize(XmlIO& xml);
	void SetMaxTrainIters(int count) {train_iter_limit = count;}
	void SetPredictInterval(int i) {predict_interval = i;}
	void SetTestPredict(bool b) {test_predict = b;}
	void SetAugmentation(int i=0, bool flip=false) {augmentation = 0; augmentation_do_flip = flip;}
	Session& SetWindowSize(int size, int min_size=1);
	
	Callback WhenSessionLoaded;
	Callback1<int> WhenStepInterval, WhenIterationInterval;
	
	
	
};

}

#endif
