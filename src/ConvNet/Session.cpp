#include "ConvNet.h"

namespace ConvNet {

Session::Session() {
	used_data = &owned_data;
	
	trainer.SetNet(net);
	
	is_training = false;
	is_training_stopped = true;
	train_iter_limit = -1;
	step_num = 0;
	predict_interval = 10;
	test_predict = false;
	forward_time = 0;
	backward_time = 0;
	step_cb_interal = 100;
	iter_cb_interal = 100;
	augmentation = 0;
	augmentation_do_flip = false;
	
	SetWindowSize(100);
}

Session::~Session() {
	StopTraining();
	net.Enter();
	net.Leave();
	ClearLayers();
}

void Session::CopyFrom(Session& session) {
	StringStream ss;
	ss.SetStoring();
	ss % session;
	ss.Seek(0);
	ss.SetLoading();
	ss % *this;
}

Session& Session::SetWindowSize(int size, int min_size) {
	loss_window.Init(size, min_size);
	reward_window.Init(size, min_size);
	l1_loss_window.Init(size, min_size);
	l2_loss_window.Init(size, min_size);
	train_window.Init(size, min_size);
	accuracy_window.Init(size, min_size);
	test_window.Init(size, min_size);
	return *this;
}

void Session::Clear() {
	StopTraining();
	ClearLayers();
	Reset();
	ResetTraining();
	train_iter_limit = -1;
	step_num = 0;
}

void Session::ClearLayers() {
	lock.Enter();
	net.Clear();
	lock.Leave();
}

void Session::StartTraining() {
	if (!is_training_stopped) return;
	is_training = true;
	is_training_stopped = false;
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
	if (trainer.GetType() == TRAINER_NULL) {
		LOG("Can't train, because trainer has not been set");
		return;
	}
	
	is_training = true;
	is_training_stopped = false;
	
	ts.Reset();
	
	iter = 0;
	
	SessionData& d = Data();
	x.Init(d.data_w, d.data_h, d.data_d, 0.0);
	
	// reinit windows that keep track of val/train accuracies
	loss_window.Clear();
	reward_window.Clear();
	l1_loss_window.Clear();
	l2_loss_window.Clear();
	train_window.Clear();
	accuracy_window.Clear();
	test_window.Clear();
	
}

void Session::TrainIteration() {
	SessionData& d = Data();
	
	const Vector<LayerBase>& layers = net.GetLayers();
	bool train_regression = d.is_data_result ? false : (layers.Top().IsRegressionLayer() || layers.Top().IsDeconvLayer());
	
	try {
	
		for(int i = 0; i < d.GetDataCount() && is_training; i++) {
			ASSERT(d.data[i]);
			
			x.SetData(d.Get(i));
			
			if (augmentation)
				x.Augment(augmentation, -1, -1, augmentation_do_flip);
			
			lock.Enter();
			
			// use x to build our estimate of validation error
			if (test_predict && (step_num % predict_interval) == 0) {
				TimeStop ts;
				Volume& v = net.Forward(x);
				forward_time = ts.Elapsed();
				
				if (train_regression || d.is_data_result) {
					// Mean squared error
					const Vector<double>& correct = train_regression ? x.GetWeights() : d.GetResult(i);
					double mse = 0.0;
					for (int i = 0; i < v.GetLength(); i++) {
						double diff = correct[i] - v.Get(i);
						mse += diff * diff;
					}
					mse /= v.GetLength();
					accuracy_window.Add(-mse);
				}
				else {
					// Is correct prediction or not?
					int cls = net.GetPrediction();
					accuracy_window.Add(cls == d.GetLabel(i) ? 1.0 : 0.0);
				}
			}
			
			TimeStop ts;
			if (d.is_data_result)
				trainer.Train(x, d.GetResult(i));
			else if (train_regression)
				trainer.Train(x, x.GetWeights()); // value
			else
				trainer.Train(x, d.GetLabel(i), 1.0); // value
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
				if (train_regression || d.is_data_result) {
					// Mean squared error
					Volume& v = net.GetOutput();
					const Vector<double>& correct = train_regression ? x.GetWeights() : d.GetResult(i);
					double mse = 0.0;
					for (int i = 0; i < v.GetLength(); i++) {
						double diff = correct[i] - v.Get(i);
						mse += diff * diff;
					}
					mse /= v.GetLength();
					train_window.Add(-mse);
				}
				else {
					// Is correct prediction or not?
					int cls = net.GetPrediction();
					train_window.Add(cls == d.GetLabel(i) ? 1.0 : 0.0); // add 1 when label is correct
				}
			}
			
			reward_window.Add(reward);
			loss_window.Add(loss);
			l1_loss_window.Add(loss_l1d);
			l2_loss_window.Add(loss_l2d);
			
			
			if ((step_num % step_cb_interal) == 0)
				WhenStepInterval(step_num);
			
		}
		
		iter++;
		
	
		if ((iter % iter_cb_interal) == 0)
			WhenIterationInterval(iter);
		
	}
	catch (Exc e) {
		lock.Leave();
		LOG("Exception: " + e);
		TrainEnd();
	}
	catch (...) {
		lock.Leave();
		LOG("Unknown exception");
		TrainEnd();
	}
}

void Session::TrainEnd() {
	
	LOG("loss = " << loss_window.GetAverage() << ", " << iter << " cycles through data in " << ts.ToString() << "ms");
	is_training_stopped = true;
	is_training = false;
}

void Session::TrainOnce(Volume& x, const Vector<double>& y) {
	lock.Enter();
	
	// use x to build our estimate of validation error
	if (test_predict && (step_num % predict_interval) == 0) {
		TimeStop ts;
		Volume& v = net.Forward(x);
		forward_time = ts.Elapsed();
		
		// Mean squared error
		double mse = 0.0;
		for (int i = 0; i < v.GetLength(); i++) {
			double diff = y[i] - v.Get(i);
			mse += diff * diff;
		}
		mse /= v.GetLength();
		accuracy_window.Add(-mse);
	}
	
	TimeStop ts;
	trainer.Train(x, y);
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
		// Mean squared error
		Volume& v = net.GetOutput();
		double mse = 0.0;
		for (int i = 0; i < v.GetLength(); i++) {
			double diff = y[i] - v.Get(i);
			mse += diff * diff;
		}
		mse /= v.GetLength();
		train_window.Add(-mse);
	}
	
	reward_window.Add(reward);
	loss_window.Add(loss);
	l1_loss_window.Add(loss_l1d);
	l2_loss_window.Add(loss_l2d);
	
	
	if ((step_num % step_cb_interal) == 0)
		WhenStepInterval(step_num);
	
}

Net& Session::GetNetwork() {
	return net;
}

LayerBase* Session::GetInput() {
	if (net.GetLayers().IsEmpty()) return NULL;
	ASSERT(net.GetLayers()[0].IsInputLayer());
	return &net.GetLayers()[0];
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
					trainer.SetType(layer); \
					trainer_loaded = true; \
				}
			
			LOAD_LAYER("adadelta", TRAINER_ADADELTA);
			LOAD_LAYER("adagrad", TRAINER_ADAGRAD);
			LOAD_LAYER("adam", TRAINER_ADAM);
			LOAD_LAYER("netsterov", TRAINER_NETSTEROV);
			LOAD_LAYER("sgd", TRAINER_SGD);
			LOAD_LAYER("windowgrad", TRAINER_WINDOWGRAD);
			
			
			if (trainer_loaded) {
				#define OPT(x) {Value x = row.GetAdd(#x); if (!x.IsNull()) {trainer.x = x;}}
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
			ARG(k);
			ARG(n);
			ARG(alpha);
			ARG(beta);
			
			if(type == "softmax" || type == "svm") {
				// add an fc layer here, there is no reason the user should
				// have to worry about this and we almost always want to
				//new_defs.push({type:'fc', neuron_count: def.num_classes});
				AddFullyConnLayer(REQ(class_count));
			}
			
			if(type == "regression" || type == "heteroscedastic_regression") {
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
			else if (type == "lrn")			AddLrnLayer(REQ(k), REQ(n), REQ(alpha), REQ(beta));
			else if (type == "dropout")		AddDropoutLayer(REQ(drop_prob));
			else if (type == "input")		AddInputLayer(REQ(input_width), REQ(input_height), REQ(input_depth));
			else if (type == "softmax")		AddSoftmaxLayer(REQ(class_count));
			else if (type == "regression")	AddRegressionLayer();
			else if (type == "heteroscedastic_regression")	AddHeteroscedasticRegressionLayer();
			else if (type == "conv")		AddConvLayer(REQ(width), REQ(height), REQ(filter_count), DEF(l1_decay_mul, 0.0), DEF(l2_decay_mul, 1.0), DEF(stride, 1), DEF(pad, 0), DEF(bias_pref, 0.0));
			else if (type == "deconv")		AddDeconvLayer(REQ(width), REQ(height), REQ(filter_count), DEF(l1_decay_mul, 0.0), DEF(l2_decay_mul, 1.0), DEF(stride, 1), DEF(pad, 0), DEF(bias_pref, 0.0));
			else if (type == "pool")		AddPoolLayer(REQ(width), REQ(height), DEF(stride, 2), DEF(pad, 0));
			else if (type == "unpool")		AddUnpoolLayer(REQ(width), REQ(height), DEF(stride, 2), DEF(pad, 0));
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

void Session::Serialize(Stream& s) {
	s % loss_window %reward_window % l1_loss_window % l2_loss_window % train_window % accuracy_window % test_window
	  % accuracy_result_window
	  % owned_data
	  % trainer
	  % net
	  % x
	  % session_last_input_array
	  % predict_interval % step_num
	  % train_iter_limit
	  % iter
	  % forward_time % backward_time
	  % step_cb_interal
	  % iter_cb_interal
	  % augmentation
	  % is_training % is_training_stopped
	  % test_predict
	  % augmentation_do_flip;
}

void Session::Xmlize(XmlIO& xml) {
	
}

void Session::ClearData() {
	Enter();
	Data().ClearData();
	Leave();
}

// Variables, what are being used during training iterations
void Session::Reset() {
	session_last_input_array.Clear();
	
	loss_window.Clear();
	reward_window.Clear();
	l1_loss_window.Clear();
	l2_loss_window.Clear();
	train_window.Clear();
	accuracy_window.Clear();
	step_num = 0;
	
	for(int i = 0; i < net.GetLayers().GetCount(); i++) {
		net.GetLayers()[i].Reset();
	}
	trainer.Reset();
}

// Variables, what are being used at the end or beginning of training
void Session::ResetTraining() {
	
	accuracy_result_window.Clear();
	
}

LayerBase& Session::AddFullyConnLayer(int neuron_count, double l1_decay_mul, double l2_decay_mul, double bias_pref) {
	LayerBase& fc = net.AddLayer();
	fc.layer_type = FULLYCONN_LAYER;
	fc.neuron_count = neuron_count;
	if (neuron_count < 1)
		throw ArgumentException("Neuron count must be more than 0");
	fc.l1_decay_mul = l1_decay_mul;
	fc.l2_decay_mul = l2_decay_mul;
	fc.bias_pref = bias_pref;
	net.CheckLayer();
	return fc;
}

LayerBase& Session::AddLrnLayer(double k, int n, double alpha, double beta) {
	LayerBase& lrn = net.AddLayer();
	lrn.layer_type = LRN_LAYER;
	lrn.k = k;
	lrn.n = n;
	lrn.alpha = alpha;
	lrn.beta = beta;
	if (n % 2 == 0)
		throw ArgumentException("n should be odd for LRN layer");
	net.CheckLayer();
	return lrn;
}

LayerBase& Session::AddDropoutLayer(double drop_prob) {
	LayerBase& dol = net.AddLayer();
	dol.layer_type = DROPOUT_LAYER;
	dol.drop_prob = drop_prob;
	if (!(drop_prob >= 0.0 && drop_prob <= 1.0))
		throw ArgumentException("DropOutLayer probability is not valid");
	net.CheckLayer();
	return dol;
}

LayerBase& Session::AddInputLayer(int input_width, int input_height, int input_depth) {
	LayerBase& in = net.AddLayer();
	in.layer_type = INPUT_LAYER;
	if (!(input_width > 0 && input_height > 0 && input_depth > 0))
		throw ArgumentException("All volume components must be positive integers");
	in.Init(input_width, input_height, input_depth);
	in.output_width = input_width;
	in.output_height = input_height;
	in.output_depth = input_depth;
	net.CheckLayer();
	return in;
}

LayerBase& Session::AddSoftmaxLayer(int class_count) {
	LayerBase& sm = net.AddLayer();
	sm.layer_type = SOFTMAX_LAYER;
	if (class_count < 1)
		throw ArgumentException("SoftmaxLayer class_count must be a positive integer");
	sm.class_count = class_count;
	net.CheckLayer();
	return sm;
}

LayerBase& Session::AddRegressionLayer() {
	LayerBase& reg = net.AddLayer();
	reg.layer_type = REGRESSION_LAYER;
	net.CheckLayer();
	return reg;
}

LayerBase& Session::AddHeteroscedasticRegressionLayer() {
	LayerBase& reg = net.AddLayer();
	reg.layer_type = HETEROSCEDASTICREGRESSION_LAYER;
	net.CheckLayer();
	return reg;
}

LayerBase& Session::AddConvLayer(int width, int height, int filter_count, double l1_decay_mul, double l2_decay_mul, int stride, int pad, double bias_pref) {
	LayerBase& conv = net.AddLayer();
	conv.layer_type = CONV_LAYER;
	conv.filter_count = filter_count;
	conv.width = width;
	conv.height = height;
	conv.l1_decay_mul = l1_decay_mul;
	conv.l2_decay_mul = l2_decay_mul;
	conv.stride = stride;
	conv.pad = pad;
	conv.bias_pref = bias_pref;
	net.CheckLayer();
	return conv;
}

LayerBase& Session::AddDeconvLayer(int width, int height, int filter_count, double l1_decay_mul, double l2_decay_mul, int stride, int pad, double bias_pref) {
	LayerBase& conv = net.AddLayer();
	conv.layer_type = DECONV_LAYER;
	conv.filter_count = filter_count;
	conv.width = width;
	conv.height = height;
	conv.l1_decay_mul = l1_decay_mul;
	conv.l2_decay_mul = l2_decay_mul;
	conv.stride = stride;
	conv.pad = pad;
	conv.bias_pref = bias_pref;
	net.CheckLayer();
	return conv;
}

LayerBase& Session::AddPoolLayer(int width, int height, int stride, int pad) {
	LayerBase& pool = net.AddLayer();
	pool.layer_type = POOL_LAYER;
	pool.width = width;
	pool.height = height;
	pool.stride = stride;
	pool.pad = pad;
	net.CheckLayer();
	return pool;
}

LayerBase& Session::AddUnpoolLayer(int width, int height, int stride, int pad) {
	LayerBase& pool = net.AddLayer();
	pool.layer_type = UNPOOL_LAYER;
	pool.width = width;
	pool.height = height;
	pool.stride = stride;
	pool.pad = pad;
	net.CheckLayer();
	return pool;
}

LayerBase& Session::AddReluLayer() {
	LayerBase& relu = net.AddLayer();
	relu.layer_type = RELU_LAYER;
	net.CheckLayer();
	return relu;
}

LayerBase& Session::AddSigmoidLayer() {
	LayerBase& sig = net.AddLayer();
	sig.layer_type = SIGMOID_LAYER;
	net.CheckLayer();
	return sig;
}

LayerBase& Session::AddTanhLayer() {
	LayerBase& tanh = net.AddLayer();
	tanh.layer_type = TANH_LAYER;
	net.CheckLayer();
	return tanh;
}

LayerBase& Session::AddMaxoutLayer(int group_size) {
	LayerBase& mo = net.AddLayer();
	mo.layer_type = MAXOUT_LAYER;
	mo.group_size = group_size;
	net.CheckLayer();
	return mo;
}

LayerBase& Session::AddSVMLayer(int class_count) {
	LayerBase& svm = net.AddLayer();
	svm.layer_type = SVM_LAYER;
	svm.class_count = class_count;
	net.CheckLayer();
	return svm;
}






}

