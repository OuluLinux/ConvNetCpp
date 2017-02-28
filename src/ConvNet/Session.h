#ifndef _ConvNet_Session_h_
#define _ConvNet_Session_h_

#include "Net.h"

namespace ConvNet {

class Session {
	
protected:
	typedef Exc RequiredArg;
	
	Vector<Vector<double> > data;
	Vector<double> mins, maxs, session_last_input_array;
	Vector<int> labels;
	Vector<LayerBasePtr> owned_layers;
	TrainerBasePtr owned_trainer, trainer;
	SpinLock lock;
	Net net;
	TimeStop ts;
	Volume x;
	double avloss;
	int train_iter_limit;
	int cols, iter;
	bool is_training, is_training_stopped;
	
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
	void ClearData() {data.Clear(); labels.Clear();}
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
	double GetData(int i, int col) const;
	double GetMax(int col) const {return maxs[col];}
	double GetMin(int col) const {return mins[col];}
	int GetLabel(int i) const {return labels[i];}
	int GetDataCount() const;
	int GetColumnCount() const {return data.IsEmpty() ? 0 : data[0].GetCount();}
	bool IsTraining() const {return !is_training_stopped || is_training;}
	
	
	bool MakeLayers(const String& json);
	bool LoadOriginalJSON(const String& json);
	bool StoreOriginalJSON(String& json);
	void BeginData(int count, int column_count);
	void EndData();
	void SetMaxTrainIters(int count) {train_iter_limit = count;}
	Session& SetData(int i, int col, double value) {data[i][col] = value; return *this;}
	Session& SetLabel(int i, int label) {labels[i] = label; return *this;}
	Session& SetTrainer(TrainerBase& trainer) {this->trainer = &trainer; return *this;}
	
	Callback WhenSessionLoaded;
	
};

}

#endif
