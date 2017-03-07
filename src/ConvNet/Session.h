#ifndef _ConvNet_Session_h_
#define _ConvNet_Session_h_

#include "Net.h"

namespace ConvNet {

class Session {
	
protected:
	typedef Exc RequiredArg;
	
	Vector<VolumeDataBase*> data, test_data, result_data;
	Vector<double> mins, maxs, session_last_input_array;
	Vector<int> labels, test_labels;
	Vector<LayerBasePtr> owned_layers;
	Vector<String> classes;
	TrainerBasePtr owned_trainer, trainer;
	SpinLock lock;
	Net net;
	TimeStop ts;
	Volume x;
	Window loss_window, reward_window, l1_loss_window, l2_loss_window, train_window, accuracy_window;
	int predict_interval, step_num;
	int train_iter_limit;
	int iter;
	int data_w, data_h, data_d, data_len;
	int forward_time, backward_time;
	int step_cb_interal;
	int augmentation;
	bool is_training, is_training_stopped;
	bool test_predict;
	bool augmentation_do_flip;
	bool is_data_result;
	
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
	void ClearData();
	void ClearOwnedLayers();
	void ClearOwnedTrainer();
	void Clear();
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
	VolumeDataBase& Get(int i) {return *data[i];}
	VolumeDataBase& GetTest(int i) {return *test_data[i];}
	VolumeDataBase& GetResult(int i) {return *result_data[i];}
	String GetClass(int i) const {return classes[i];}
	double GetData(int i, int col) const;
	double GetTestData(int i, int col) const;
	double GetMax(int col) const {return maxs[col];}
	double GetMin(int col) const {return mins[col];}
	double GetL1DecayLossAverage() const {return l1_loss_window.GetAverage();}
	double GetL2DecayLossAverage() const {return l2_loss_window.GetAverage();}
	double GetTrainingAccuracyAverage() const {return train_window.GetAverage();}
	double GetValidationAccuracyAverage() const {return accuracy_window.GetAverage();}
	int GetLabel(int i) const {return labels[i];}
	int GetDataCount() const;
	int GetDataLength() const {return data_w * data_h * data_d;}
	int GetForwardTime() const {return forward_time;}
	int GetBackwardTime() const {return backward_time;}
	int GetDataWidth() const {return data_w;}
	int GetDataHeight() const {return data_h;}
	int GetDataDepth() const {return data_d;}
	int GetClassCount() const {return classes.GetCount();}
	int GetStepCount() const {return step_num;}
	bool IsTraining() const {return !is_training_stopped || is_training;}
	void GetUniformClassData(int per_class, Vector<VolumeDataBase*>& volumes, Vector<int>& labels);
	
	virtual double GetLossAverage() const {return loss_window.GetAverage();}
	virtual double GetRewardAverage() const {return reward_window.GetAverage();}
	
	bool MakeLayers(const String& json);
	bool LoadOriginalJSON(const String& json);
	bool StoreOriginalJSON(String& json);
	void BeginData(int cls_count, int count, int column_count, int test_count=0) {BeginDataClass<VolumeData<double> >(cls_count, count, 1, 1, column_count, test_count);}
	void BeginData(int cls_count, int count, int width, int height, int depth, int test_count=0) {BeginDataClass<VolumeData<double> >(cls_count, count, width, height, depth, test_count);}
	void BeginDataResult(int result_length, int count, int column_count, int test_count=0) {BeginDataResult<VolumeData<double> >(result_length, count, 1, 1, column_count, test_count);}
	void EndData();
	void SetMaxTrainIters(int count) {train_iter_limit = count;}
	void SetPredictInterval(int i) {predict_interval = i;}
	void SetTestPredict(bool b) {test_predict = b;}
	void SetAugmentation(int i=0, bool flip=false) {augmentation = 0; augmentation_do_flip = flip;}
	Session& SetData(int i, int col, double value) {data[i]->Set(col, value); return *this;}
	Session& SetResult(int i, int col, double value) {result_data[i]->Set(col, value); return *this;}
	Session& SetLabel(int i, int label) {labels[i] = label; return *this;}
	Session& SetTestData(int i, int col, double value) {test_data[i]->Set(col, value); return *this;}
	Session& SetTestLabel(int i, int label) {test_labels[i] = label; return *this;}
	Session& SetClass(int i, const String& cls) {classes[i] = cls; return *this;}
	Session& SetTrainer(TrainerBase& trainer) {this->trainer = &trainer; return *this;}
	Session& SetWindowSize(int size, int min_size=1);
	
	Callback WhenSessionLoaded;
	Callback1<int> WhenStepInterval;
	
	template <class T>
	void BeginDataClass(int cls_count, int count, int width, int height, int depth, int test_count=0) {
		ClearData();
		
		data_len = width * height * depth;
		data_w = width;
		data_h = height;
		data_d = depth;
		
		is_data_result = false;
		
		data.SetCount(count, NULL);
		for(int i = 0; i < data.GetCount(); i++)
			data[i] = new T(data_len);
		
		test_data.SetCount(test_count, NULL);
		for(int i = 0; i < test_data.GetCount(); i++)
			test_data[i] = new T(data_len);
		
		labels.SetCount(0);
		labels.SetCount(count, 0);
		test_labels.SetCount(0);
		test_labels.SetCount(test_count, -1);
		
		mins.Clear();
		mins.SetCount(data_len, DBL_MAX);
		maxs.Clear();
		maxs.SetCount(data_len, -DBL_MAX);
		
		classes.SetCount(cls_count);
		
	}
	
	template <class T>
	void BeginDataResult(int result_length, int count, int width, int height, int depth, int test_count=0) {
		ClearData();
		
		data_len = width * height * depth;
		data_w = width;
		data_h = height;
		data_d = depth;
		
		is_data_result = true;
		
		data.SetCount(count, NULL);
		for(int i = 0; i < data.GetCount(); i++)
			data[i] = new T(data_len);
		
		result_data.SetCount(count, NULL);
		for(int i = 0; i < result_data.GetCount(); i++)
			result_data[i] = new T(result_length);
		
		test_data.SetCount(test_count, NULL);
		for(int i = 0; i < test_data.GetCount(); i++)
			test_data[i] = new T(data_len);
		
		mins.Clear();
		mins.SetCount(data_len, DBL_MAX);
		maxs.Clear();
		maxs.SetCount(data_len, -DBL_MAX);
		
	}
	
	
	
};

}

#endif
