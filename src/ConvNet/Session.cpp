#include "ConvNet.h"

namespace ConvNet {

Session::Session() {
	is_training = false;
	is_training_stopped = true;
	train_iter_limit = 0;
	owned_trainer = NULL;
	trainer = NULL;
	step_num = 0;
	predict_interval = 10;
	test_predict = false;
	forward_time = 0;
	backward_time = 0;
	step_cb_interal = 100;
	augmentation = 0;
	augmentation_do_flip = false;
	data_w = 0;
	data_h = 0;
	data_d = 0;
	
	SetWindowSize(100);
}

Session::~Session() {
	StopTraining();
	ClearOwnedLayers();
	ClearOwnedTrainer();
	ClearData();
}

Session& Session::SetWindowSize(int size, int min_size) {
	loss_window.Init(size, min_size);
	reward_window.Init(size, min_size);
	l1_loss_window.Init(size, min_size);
	l2_loss_window.Init(size, min_size);
	train_window.Init(size, min_size);
	accuracy_window.Init(size, min_size);
	return *this;
}

void Session::Clear() {
	StopTraining();
	ClearOwnedLayers();
	ClearOwnedTrainer();
	train_iter_limit = 0;
}

void Session::ClearOwnedLayers() {
	lock.Enter();
	net.Clear();
	for(int i = 0; i < owned_layers.GetCount(); i++)
		if (owned_layers[i])
			delete owned_layers[i];
	owned_layers.Clear();
	lock.Leave();
}

void Session::ClearOwnedTrainer() {
	lock.Enter();
	if (owned_trainer)
		delete owned_trainer;
	owned_trainer = NULL;
	trainer = NULL;
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
	is_training = false;
	#ifdef flagMT
	while (!is_training_stopped)
		Sleep(100);
	#else
	if (is_training)
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
	if (!trainer) {
		LOG("Can't train, because trainer has not been set");
		return;
	}
	
	is_training = true;
	is_training_stopped = false;
	
	ts.Reset();
	
	iter = 0;
	
	x.Init(data_w, data_h, data_d, 0.0);
	
	// reinit windows that keep track of val/train accuracies
	loss_window.Clear();
	reward_window.Clear();
	l1_loss_window.Clear();
	l2_loss_window.Clear();
	train_window.Clear();
	accuracy_window.Clear();
	
}

void Session::TrainIteration() {
	TrainerBase& trainer = *this->trainer;
	
	const Vector<LayerBasePtr>& layers = net.GetLayers();
	bool train_regression = dynamic_cast<RegressionLayer*>(layers[layers.GetCount()-1]);
	
	try {
	
		for(int i = 0; i < GetDataCount() && is_training; i++) {
			ASSERT(data[i]);
			
			x.SetData(Get(i));
			int label = GetLabel(i);
			
			if (augmentation)
				x.Augment(augmentation, -1, -1, augmentation_do_flip);
			
			lock.Enter();
			
			// use x to build our estimate of validation error
			if (test_predict && (step_num % predict_interval) == 0) {
				TimeStop ts;
				net.Forward(x);
				forward_time = ts.Elapsed();
				
				int cls = net.GetPrediction();
				accuracy_window.Add(cls == label ? 1.0 : 0.0);
			}
			
			TimeStop ts;
			if (train_regression)
				trainer.Train(x, x.GetWeights()); // value
			else
				trainer.Train(x, label, 1.0); // value
			backward_time = ts.Elapsed();
			double reward = trainer.GetReward();
			double loss = trainer.GetLoss();
			double loss_l1d = trainer.GetL1DecayLoss();
			double loss_l2d = trainer.GetL2DecayLoss();
			step_num++;
			lock.Leave();
			
			// keep track of stats such as the average training error and loss
			// if last layer is softmax, then add prediction value to the average
			if (test_predict) {
				int cls = net.GetPrediction();
				train_window.Add(cls == label ? 1.0 : 0.0); // add 1 when label is correct
			}
			
			reward_window.Add(reward);
			loss_window.Add(loss);
			l1_loss_window.Add(loss_l1d);
			l2_loss_window.Add(loss_l2d);
			
			
			if ((step_num % step_cb_interal) == 0)
				WhenStepInterval(step_num);
			
		}
		
		iter++;
	}
	catch (Exc e) {
		lock.Leave();
		LOG("Exception: " + e);
		TrainEnd();
	}
}

void Session::TrainEnd() {
	
	LOG("loss = " << loss_window.GetAverage() << ", " << iter << " cycles through data in " << ts.ToString() << "ms");
	is_training_stopped = true;
	is_training = false;
}

Net& Session::GetNetwork() {
	return net;
}

InputLayer* Session::GetInput() const {
	if (net.GetLayers().IsEmpty()) return NULL;
	return dynamic_cast<InputLayer*>(&*net.GetLayers()[0]);
}

double Session::GetData(int i, int col) const {
	return data[i]->Get(col);
}

double Session::GetTestData(int i, int col) const {
	return test_data[i]->Get(col);
}

int Session::GetDataCount() const {
	return data.GetCount();
}

const Value& Session::ChkNotNull(const String& key, const Value& v) {
	if (v.IsNull()) throw RequiredArg(key);
	return v;
}

bool Session::MakeLayers(const String& json) {
	Clear();
	
	Value js = ParseJSON(json);
	if (js.IsNull()) {
		LOG("JSON parse failed");
		return false;
	}
	
	Enter();
	
	try {
		
		// Read layer settings
		for(int i = 0; i < js.GetCount(); i++) {
			Value row = js[i];
			
			String type = row.GetAdd("type");
			if (type.IsEmpty()) {
				LOG("Invalid JSON");
				Leave();
				return false;
			}
			
			// Read trainer
			bool trainer_loaded = false;
			#define LOAD_LAYER(key, layer) \
				if (type == key) {\
					if (owned_trainer) {LOG("Only one trainer can be loaded"); Leave(); return false;} \
					owned_trainer = new layer (net); \
					trainer = owned_trainer; \
					trainer_loaded = true; \
				}
			
			LOAD_LAYER("adadelta", AdadeltaTrainer);
			LOAD_LAYER("adagrad", AdagradTrainer);
			LOAD_LAYER("adam", AdamTrainer);
			LOAD_LAYER("netsterov", NetsterovTrainer);
			LOAD_LAYER("sgd", SgdTrainer);
			LOAD_LAYER("windowgrad", WindowgradTrainer);
			
			
			if (trainer_loaded) {
				#define OPT(x) {Value x = row.GetAdd(#x); if (!x.IsNull()) {trainer->x = x;}}
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
				Leave();
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
			else if (type == "lrn")			AddLrnLayer();
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
				Leave();
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
					Leave();
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
		Leave();
		return false;
	}
	
	if (net.GetLayers().GetCount() < 2) {
		LOG("Error! At least one input layer and one loss layer are required.");
		Leave();
		return false;
	}
	
	Leave();
	
	WhenSessionLoaded();
	
	return true;
}

bool Session::LoadOriginalJSON(const String& json) {
	ClearOwnedLayers();
	
	Enter();
	
	ValueMap js = ParseJSON(json);
	
	int layers_id = js.Find("layers");
	if (layers_id == -1) {Leave(); return false;}
	
	ValueMap layers = js.GetValue(layers_id);
	
	for(int i = 0; i < layers.GetCount(); i++) {
		ValueMap layer = layers.GetValue(i);
		
		String type = layer.GetAdd("layer_type");
		if (type.IsEmpty()) return false;
		
		if      (type == "fc")			LoadLayer<FullyConnLayer>(layer);
		else if (type == "lrn")			LoadLayer<LrnLayer>(layer);
		else if (type == "dropout")		LoadLayer<DropOutLayer>(layer);
		else if (type == "input")		LoadLayer<InputLayer>(layer);
		else if (type == "softmax")		LoadLayer<SoftmaxLayer>(layer);
		else if (type == "regression")	LoadLayer<RegressionLayer>(layer);
		else if (type == "conv")		LoadLayer<ConvLayer>(layer);
		else if (type == "pool")		LoadLayer<PoolLayer>(layer);
		else if (type == "relu")		LoadLayer<ReluLayer>(layer);
		else if (type == "sigmoid")		LoadLayer<SigmoidLayer>(layer);
		else if (type == "tanh")		LoadLayer<TanhLayer>(layer);
		else if (type == "maxout")		LoadLayer<MaxoutLayer>(layer);
		else if (type == "svm")			LoadLayer<SvmLayer>(layer);
		else {
			LOG("ERROR: UNRECOGNIZED LAYER TYPE: " + type);
			return false;
		}
	}
	
	Leave();
	
	WhenSessionLoaded();
	
	return true;
}

bool Session::StoreOriginalJSON(String& json) {
	Enter();
	
	Value new_layers;
	for(int i = 0; i < this->owned_layers.GetCount(); i++) {
		ValueMap map;
		this->owned_layers[i]->Store(map);
		new_layers.Add(map);
	}
	
	ValueMap js;
	js.GetAdd("layers") = new_layers;
	
	json = AsJSON(js, true);
	
	Leave();
	
	// HOTFIX: Bug in older U++: test if json puts , instead of .
	String s = Format("%.16g", 7.56);
	if (s.Find(",") != -1) {
		String fix_tmp;
		
		char prev = json[0];
		fix_tmp.Cat(prev);
		
		int count = json.GetCount()-1;
		for(int i = 1; i < count; i++) {
			char cur = json[i];
			char next = json[i+1];
			if (cur == ',' && IsDigit(prev) && IsDigit(next))
				fix_tmp.Cat('.');
			else
				fix_tmp.Cat(cur);
			prev = cur;
		}
		fix_tmp.Cat(json[count]);
		
		json = fix_tmp;
	}
	
	return true;
}

void Session::ClearData() {
	Enter();
	for(int i = 0; i < data.GetCount(); i++) {
		delete data[i];
	}
	data.Clear();
	for(int i = 0; i < test_data.GetCount(); i++) {
		delete test_data[i];
	}
	test_data.Clear();
	labels.Clear();
	test_labels.Clear();
	classes.Clear();
	step_num = 0;
	Leave();
}

void Session::EndData() {
	for(int i = 0; i < data.GetCount(); i++) {
		VolumeDataBase* vol = data[i];
		if (!vol) {
			data.Remove(i);
			labels.Remove(i);
			i--;
			continue;
		}
		for(int j = 0; j < vol->GetCount(); j++) {
			double d = vol->Get(j);
			double& mind = mins[j];
			double& maxd = maxs[j];
			mind = min(mind, d);
			maxd = max(maxd, d);
		}
	}
	
	// Randomize data
	int count = data.GetCount() / 2;
	for(int i = 0; i < count; i++) {
		int a = Random(data.GetCount());
		int b = Random(data.GetCount());
		Swap(data[a],	data[b]);
		Swap(labels[a],	labels[b]);
	}
	
}

FullyConnLayer& Session::AddFullyConnLayer(int neuron_count, double l1_decay_mul, double l2_decay_mul, double bias_pref) {
	FullyConnLayer* fc = new FullyConnLayer(neuron_count);
	fc->l1_decay_mul = l1_decay_mul;
	fc->l2_decay_mul = l2_decay_mul;
	fc->bias_pref = bias_pref;
	owned_layers.Add(fc);
	net.AddLayer(*fc);
	return *fc;
}

LrnLayer& Session::AddLrnLayer() {
	
	
	
	throw NotImplementedException("LrnLayer");
	
	LrnLayer* lrn = new LrnLayer();
	owned_layers.Add(lrn);
	net.AddLayer(*lrn);
	return *lrn;
}

DropOutLayer& Session::AddDropoutLayer(double drop_prob) {
	DropOutLayer* dol = new DropOutLayer(drop_prob);
	owned_layers.Add(dol);
	net.AddLayer(*dol);
	return *dol;
}

InputLayer& Session::AddInputLayer(int input_width, int input_height, int input_depth) {
	InputLayer* in = new InputLayer(input_width, input_height, input_depth);
	owned_layers.Add(in);
	net.AddLayer(*in);
	return *in;
}

SoftmaxLayer& Session::AddSoftmaxLayer(int class_count) {
	SoftmaxLayer* sm = new SoftmaxLayer(class_count);
	owned_layers.Add(sm);
	net.AddLayer(*sm);
	return *sm;
}

RegressionLayer& Session::AddRegressionLayer() {
	RegressionLayer* reg = new RegressionLayer();
	owned_layers.Add(reg);
	net.AddLayer(*reg);
	return *reg;
}

ConvLayer& Session::AddConvLayer(int width, int height, int filter_count, double l1_decay_mul, double l2_decay_mul, int stride, int pad, double bias_pref) {
	ConvLayer* conv = new ConvLayer(width, height, filter_count);
	conv->l1_decay_mul = l1_decay_mul;
	conv->l2_decay_mul = l2_decay_mul;
	conv->stride = stride;
	conv->pad = pad;
	conv->bias_pref = bias_pref;
	owned_layers.Add(conv);
	net.AddLayer(*conv);
	return *conv;
}

PoolLayer& Session::AddPoolLayer(int width, int height, int stride, int pad) {
	PoolLayer* pool = new PoolLayer(width, height);
	pool->stride = stride;
	pool->pad = pad;
	owned_layers.Add(pool);
	net.AddLayer(*pool);
	return *pool;
}

ReluLayer& Session::AddReluLayer() {
	ReluLayer* relu = new ReluLayer();
	owned_layers.Add(relu);
	net.AddLayer(*relu);
	return *relu;
}

SigmoidLayer& Session::AddSigmoidLayer() {
	SigmoidLayer* sig = new SigmoidLayer();
	owned_layers.Add(sig);
	net.AddLayer(*sig);
	return *sig;
}

TanhLayer& Session::AddTanhLayer() {
	TanhLayer* tanh = new TanhLayer();
	owned_layers.Add(tanh);
	net.AddLayer(*tanh);
	return *tanh;
}

MaxoutLayer& Session::AddMaxoutLayer(int group_size) {
	MaxoutLayer* mo = new MaxoutLayer(group_size);
	owned_layers.Add(mo);
	net.AddLayer(*mo);
	return *mo;
}

SvmLayer& Session::AddSVMLayer(int class_count) {
	SvmLayer* svm = new SvmLayer(class_count);
	owned_layers.Add(svm);
	net.AddLayer(*svm);
	return *svm;
}

void Session::GetUniformClassData(int per_class, Vector<VolumeDataBase*>& volumes, Vector<int>& labels) {
	ASSERT(per_class >= 0);
	Vector<int> counts;
	counts.SetCount(classes.GetCount(), 0);
	int remaining = per_class * classes.GetCount();
	volumes.SetCount(remaining);
	labels.SetCount(remaining);
	
	for(int i = 0; i < this->labels.GetCount() && remaining > 0; i++) {
		int label = this->labels[i];
		int& count = counts[label];
		if (count < per_class) {
			count++;
			remaining--;
			volumes[remaining] = data[i];
			labels[remaining] = label;
		}
	}
	
}

}
