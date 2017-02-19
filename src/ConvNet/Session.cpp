#include "ConvNet.h"

namespace ConvNet {

Session::Session() {
	is_training = false;
	is_training_stopped = true;
	train_iter_limit = 0;
	owned_trainer = NULL;
}

Session::~Session() {
	StopTraining();
	ClearOwnedLayers();
}

void Session::Clear() {
	StopTraining();
	ClearOwnedLayers();
	net.Clear();
	train_iter_limit = 0;
}

void Session::ClearOwnedLayers() {
	lock.Enter();
	for(int i = 0; i < owned_layers.GetCount(); i++)
		if (owned_layers[i])
			delete owned_layers[i];
	owned_layers.Clear();
	if (owned_trainer)
		delete owned_trainer;
	owned_trainer = NULL;
	lock.Leave();
}

void Session::StartTraining() {
	if (!is_training_stopped) return;
	is_training = true;
	#ifdef flagMT
	Thread::Start(THISBACK(Train));
	#else
	TrainBegin();
	#endif
}

void Session::StopTraining() {
	#ifdef flagMT
	is_training = false;
	while (!is_training_stopped)
		Sleep(100);
	#else
	TrainEnd();
	#endif
}

void Session::Train() {
	TrainBegin();
	
	while (is_training) {
		TrainIteration();
		if (iter == train_iter_limit) // zero means no limit
			break;
	}
	
	TrainEnd();
}

void Session::TrainBegin() {
	if (!owned_trainer) {
		LOG("Can't train, because trainer has not been set");
		return;
	}
	
	is_training = true;
	is_training_stopped = false;
	
	ts.Reset();
	
	avloss = 0.0;
	iter = 0;
	
	cols = GetColumnCount();
	x.Init(1,1,cols);
}

void Session::TrainIteration() {
	TrainerBase& trainer = *owned_trainer;
	
	for(int i = 0; i < GetDataCount() && is_training; i++) {
		
		for(int j = 0; j < cols; j++) {
			double d = GetData(i, j);
			x.Set(0, 0, j, d);
		}
		
		int label = GetLabel(i);
		
		lock.Enter();
		trainer.Train(x, label);
		double loss = trainer.GetLoss();
		lock.Leave();
		
		avloss += loss;
	}
	iter++;
}

void Session::TrainEnd() {
	avloss /= GetDataCount() * iter;
	
	LOG("loss = " << avloss << ", " << iter << " cycles through data in " << ts.ToString() << "ms");
	is_training_stopped = true;
	is_training = false;
}

Net& Session::GetNetwork() {
	return net;
}

InputLayer* Session::GetInput() {
	if (net.GetLayers().IsEmpty()) return NULL;
	return dynamic_cast<InputLayer*>(&*net.GetLayers()[0]);
}

double Session::GetData(int i, int col) const {
	return data[i][col];
}

int Session::GetDataCount() const {
	return data.GetCount();
}

const Value& Session::ChkNotNull(const String& key, const Value& v) {
	if (v.IsNull()) throw RequiredArg(key);
	return v;
}

bool Session::LoadJSON(const String& json) {
	Clear();
	
	Value js = ParseJSON(json);
	LOG(AsJSON(js));
	DUMP(js);
	
	try {
		
		// Read layer settings
		for(int i = 0; i < js.GetCount(); i++) {
			Value row = js[i];
			
			String type = row.GetAdd("type");
			if (type.IsEmpty()) {
				LOG("Invalid JSON");
				return false;
			}
			
			// Read trainer
			bool trainer_loaded = false;
			#define LOAD_LAYER(key, layer) \
				if (type == key) {\
					if (owned_trainer) {LOG("Only one trainer can be loaded"); return false;} \
					owned_trainer = new layer (net); \
					trainer_loaded = true; \
				}
			
			LOAD_LAYER("adadelta", AdadeltaTrainer);
			LOAD_LAYER("adagrad", AdagradTrainer);
			LOAD_LAYER("adam", AdamTrainer);
			LOAD_LAYER("netsterov", NetsterovTrainer);
			LOAD_LAYER("sgd", SgdTrainer);
			LOAD_LAYER("windowgrad", WindowgradTrainer);
			
			
			if (trainer_loaded) {
				#define OPT(x) {Value x = row.GetAdd(#x); if (!x.IsNull()) {owned_trainer->x = x;}}
				OPT(Beta1);
				OPT(Beta2);
				OPT(l1_decay);
				OPT(l2_decay);
				OPT(l1_decay_loss);
				OPT(l2_decay_loss);
				OPT(learning_rate);
				OPT(batch_size);
				OPT(momentum);
				OPT(eps);
				OPT(ro);
				continue;
			}
			
			// Read layers
			if (net.GetLayers().IsEmpty() && type != "input") {
				LOG("Error! First layer must be the input layer, to declare size of inputs. Trying to create layer " + type);
				return false;
			}
			
			// Reference all possible arguments while there is only a few of them.
			#define ARG(x) Value x = row.GetAdd(#x);
			#define REQ(x) ChkNotNull(#x, x)
			#define DEF(x, y) x.IsNull() ? y : (double)x
			
			ARG(class_count);
			ARG(neuron_count);
			ARG(activation);
			ARG(bias_pref);
			ARG(group_size);
			ARG(drop_prob);
			ARG(input_width);
			ARG(input_height);
			ARG(input_depth);
			ARG(width);
			ARG(height);
			ARG(filter_count);
			ARG(l1_decay_mul);
			ARG(l2_decay_mul);
			ARG(stride);
			ARG(pad);
			
			if(type == "softmax" || type == "svm") {
				// add an fc layer here, there is no reason the user should
				// have to worry about this and we almost always want to
				//new_defs.push({type:'fc', neuron_count: def.num_classes});
				AddFullyConnLayer(REQ(class_count));
			}
			
			if(type == "regression") {
				// add an fc layer here, there is no reason the user should
				// have to worry about this and we almost always want to
				//new_defs.push({type:'fc', neuron_count: def.neuron_count});
				AddFullyConnLayer(REQ(neuron_count));
			}
			
			if((type == "fc" || type == "conv") && bias_pref.IsNull()) {
				bias_pref = 0.0;
				if (activation == "relu" ) {
					bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
					// otherwise it's technically possible that a relu unit will never turn on (by chance)
					// and will never get any gradient and never contribute any computation. Dead relu.
				}
			}
			
			if      (type == "fc")			AddFullyConnLayer(REQ(neuron_count), DEF(l1_decay_mul, 0.0), DEF(l2_decay_mul, 1.0), DEF(bias_pref, 0.0));
			else if (type == "lrn")			AddLocalResponseNormalizationLayer();
			else if (type == "dropout")		AddDropoutLayer(REQ(drop_prob));
			else if (type == "input")		AddInputLayer(REQ(input_width), REQ(input_height), REQ(input_depth));
			else if (type == "softmax")		AddSoftmaxLayer(REQ(class_count));
			else if (type == "regression")	AddRegressionLayer();
			else if (type == "conv")		AddConvLayer(REQ(width), REQ(height), REQ(filter_count), DEF(l1_decay_mul, 0.0), DEF(l2_decay_mul, 1.0), DEF(stride, 1), DEF(pad, 0), DEF(bias_pref, 0.0));
			else if (type == "pool")		AddPoolLayer(REQ(width), REQ(height), DEF(stride, 2), DEF(pad, 0));
			else if (type == "relu")		AddReluLayer();
			else if (type == "sigmoid")		AddSigmoidLayer();
			else if (type == "tanh")		AddTanhLayer();
			else if (type == "maxout")		AddMaxoutLayer(REQ(group_size));
			else if (type == "svm")			AddSVMLayer(REQ(class_count));
			else {
				LOG("ERROR: UNRECOGNIZED LAYER TYPE: " + type);
				return false;
			}
			
			
			
			if (!activation.IsNull()) {
				String act_str = activation;
				if (act_str == "relu") {
					//new_defs.push({type:'relu'});
					AddReluLayer();
				}
				else if (act_str == "sigmoid") {
					//new_defs.push({type:'sigmoid'});
					AddSigmoidLayer();
				}
				else if (act_str == "tanh") {
					//new_defs.push({type:'tanh'});
					AddTanhLayer();
				}
				else if (act_str == "maxout") {
					// create maxout activation, and pass along group size, if provided
					//new_defs.push({type:'maxout', group_size:gs});
					AddMaxoutLayer(DEF(group_size, 2));
				}
				else {
					LOG("ERROR unsupported activation " + act_str);
					return false;
				}
			}
			if (!drop_prob.IsNull() && type != "dropout") {
				//new_defs.push({type:'dropout', drop_prob: def.drop_prob});
				AddDropoutLayer(REQ(drop_prob));
			}
	
		}
	}
	catch (RequiredArg a) {
		LOG("Required argument " + a + " was missing");
		return false;
	}
	
	if (net.GetLayers().GetCount() < 2) {
		LOG("Error! At least one input layer and one loss layer are required.");
		return false;
	}
	
	WhenSessionLoaded();
	
	return true;
}

void Session::BeginData(int count, int column_count) {
	Vector<double> cols;
	cols.SetCount(column_count, 0.0);
	data.Clear();
	data.SetCount(count, cols);
	labels.Clear();
	labels.SetCount(count, 0);
	mins.Clear();
	mins.SetCount(column_count, DBL_MAX);
	maxs.Clear();
	maxs.SetCount(column_count, -DBL_MAX);
}

void Session::EndData() {
	for(int i = 0; i < data.GetCount(); i++) {
		Vector<double>& cols = data[i];
		for(int j = 0; j < cols.GetCount(); j++) {
			double d = cols[j];
			double& mind = mins[j];
			double& maxd = maxs[j];
			mind = min(mind, d);
			maxd = max(maxd, d);
		}
	}
}

Session& Session::AddFullyConnLayer(int neuron_count, double l1_decay_mul, double l2_decay_mul, double bias_pref) {
	FullyConnLayer* fc = new FullyConnLayer(neuron_count);
	fc->l1_decay_mul = l1_decay_mul;
	fc->l2_decay_mul = l2_decay_mul;
	fc->bias_pref = bias_pref;
	owned_layers.Add(fc);
	net.AddLayer(*fc);
	return *this;
}

Session& Session::AddLocalResponseNormalizationLayer() {
	
	
	
	throw NotImplementedException("LocalResponseNormalizationLayer");
	
	
	
	return *this;
}

Session& Session::AddDropoutLayer(double drop_prob) {
	DropOutLayer* dol = new DropOutLayer(drop_prob);
	owned_layers.Add(dol);
	net.AddLayer(*dol);
	return *this;
}

Session& Session::AddInputLayer(int input_width, int input_height, int input_depth) {
	InputLayer* in = new InputLayer(input_width, input_height, input_depth);
	owned_layers.Add(in);
	net.AddLayer(*in);
	return *this;
}

Session& Session::AddSoftmaxLayer(int class_count) {
	SoftmaxLayer* sm = new SoftmaxLayer(class_count);
	owned_layers.Add(sm);
	net.AddLayer(*sm);
	return *this;
}

Session& Session::AddRegressionLayer() {
	RegressionLayer* reg = new RegressionLayer();
	owned_layers.Add(reg);
	net.AddLayer(*reg);
	return *this;
}

Session& Session::AddConvLayer(int width, int height, int filter_count, double l1_decay_mul, double l2_decay_mul, int stride, int pad, double bias_pref) {
	ConvLayer* conv = new ConvLayer(width, height, filter_count);
	conv->l1_decay_mul = l1_decay_mul;
	conv->l2_decay_mul = l2_decay_mul;
	conv->stride = stride;
	conv->pad = pad;
	conv->bias_pref = bias_pref;
	owned_layers.Add(conv);
	net.AddLayer(*conv);
	return *this;
}

Session& Session::AddPoolLayer(int width, int height, int stride, int pad) {
	PoolLayer* pool = new PoolLayer(width, height);
	pool->stride = stride;
	pool->pad = pad;
	owned_layers.Add(pool);
	net.AddLayer(*pool);
	return *this;
}

Session& Session::AddReluLayer() {
	ReluLayer* relu = new ReluLayer();
	owned_layers.Add(relu);
	net.AddLayer(*relu);
	return *this;
}

Session& Session::AddSigmoidLayer() {
	SigmoidLayer* sig = new SigmoidLayer();
	owned_layers.Add(sig);
	net.AddLayer(*sig);
	return *this;
}

Session& Session::AddTanhLayer() {
	TanhLayer* tanh = new TanhLayer();
	owned_layers.Add(tanh);
	net.AddLayer(*tanh);
	return *this;
}

Session& Session::AddMaxoutLayer(int group_size) {
	MaxoutLayer* mo = new MaxoutLayer(group_size);
	owned_layers.Add(mo);
	net.AddLayer(*mo);
	return *this;
}

Session& Session::AddSVMLayer(int class_count) {
	SvmLayer* svm = new SvmLayer(class_count);
	owned_layers.Add(svm);
	net.AddLayer(*svm);
	return *this;
}

}
