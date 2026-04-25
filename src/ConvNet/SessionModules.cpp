#include "SessionModules.h"
#include "Training.h"
#include "Session.h"
#include <initializer_list>

namespace ConvNet {

SessionModuleRegistry& SessionModuleRegistry::Get() {
	static SessionModuleRegistry inst;
	return inst;
}

void SessionModuleRegistry::RegisterLayer(const String& type, SessionLayerModuleFn fn,
                                          const SessionLayerModuleSpec& spec) {
	int i = layer_modules.Find(type);
	if (i >= 0) layer_modules[i] = fn;
	else layer_modules.Add(type, fn);

	SessionLayerModuleSpec normalized = spec;
	if (normalized.type.IsEmpty()) normalized.type = type;
	if (normalized.label.IsEmpty()) normalized.label = type;
	if (normalized.category.IsEmpty()) normalized.category = "layers";

	int si = layer_specs.Find(type);
	if (si >= 0) layer_specs[si] = pick(normalized);
	else layer_specs.Add(type, pick(normalized));
}

void SessionModuleRegistry::RegisterTrainer(const String& type, SessionTrainerModuleFn fn,
                                            const SessionTrainerModuleSpec& spec) {
	int i = trainer_modules.Find(type);
	if (i >= 0) trainer_modules[i] = fn;
	else trainer_modules.Add(type, fn);

	SessionTrainerModuleSpec normalized = spec;
	if (normalized.type.IsEmpty()) normalized.type = type;
	if (normalized.label.IsEmpty()) normalized.label = type;
	if (normalized.category.IsEmpty()) normalized.category = "trainers";

	int si = trainer_specs.Find(type);
	if (si >= 0) trainer_specs[si] = pick(normalized);
	else trainer_specs.Add(type, pick(normalized));
}

int SessionModuleRegistry::ApplyLayer(const String& type, Session& session, Value& row, String& error) const {
	int i = layer_modules.Find(type);
	if (i < 0) return 0;
	SessionLayerModuleFn fn = layer_modules[i];
	if (!fn) {
		error = "Layer module '" + type + "' is not callable";
		return -1;
	}
	bool ok = fn(session, row, error);
	if (!ok && error.IsEmpty())
		error = "Layer module '" + type + "' failed";
	return ok ? 1 : -1;
}

int SessionModuleRegistry::ApplyTrainer(const String& type, TrainerBase& trainer, Value& row, String& error) const {
	int i = trainer_modules.Find(type);
	if (i < 0) return 0;
	SessionTrainerModuleFn fn = trainer_modules[i];
	if (!fn) {
		error = "Trainer module '" + type + "' is not callable";
		return -1;
	}
	bool ok = fn(trainer, row, error);
	if (!ok && error.IsEmpty())
		error = "Trainer module '" + type + "' failed";
	return ok ? 1 : -1;
}

Vector<String> SessionModuleRegistry::GetLayerTypes() const {
	Vector<String> out;
	out <<= layer_modules.GetKeys();
	return out;
}

Vector<String> SessionModuleRegistry::GetTrainerTypes() const {
	Vector<String> out;
	out <<= trainer_modules.GetKeys();
	return out;
}

Vector<SessionLayerModuleSpec> SessionModuleRegistry::GetLayerSpecs() const {
	Vector<SessionLayerModuleSpec> out;
	out.Reserve(layer_specs.GetCount());
	for(int i = 0; i < layer_specs.GetCount(); i++)
		out.Add(layer_specs[i]);
	return out;
}

Vector<SessionTrainerModuleSpec> SessionModuleRegistry::GetTrainerSpecs() const {
	Vector<SessionTrainerModuleSpec> out;
	out.Reserve(trainer_specs.GetCount());
	for(int i = 0; i < trainer_specs.GetCount(); i++)
		out.Add(trainer_specs[i]);
	return out;
}

bool SessionModuleRegistry::GetLayerSpec(const String& type, SessionLayerModuleSpec& out) const {
	int i = layer_specs.Find(type);
	if(i < 0)
		return false;
	out = layer_specs[i];
	return true;
}

bool SessionModuleRegistry::GetTrainerSpec(const String& type, SessionTrainerModuleSpec& out) const {
	int i = trainer_specs.Find(type);
	if(i < 0)
		return false;
	out = trainer_specs[i];
	return true;
}

namespace {

Value GetArg(Value& row, const char* key) {
	Value v = row.GetAdd(key);
	if (v.IsNull()) {
		if (!strcmp(key, "input_width"))		v = row.GetAdd("out_sx");
		else if (!strcmp(key, "input_height"))	v = row.GetAdd("out_sy");
		else if (!strcmp(key, "input_depth"))	v = row.GetAdd("out_depth");
		else if (!strcmp(key, "neuron_count"))	v = row.GetAdd("num_neurons");
		else if (!strcmp(key, "class_count"))	v = row.GetAdd("num_classes");
	}
	return v;
}

Value RequireArg(Value& row, const char* key, const char* module_type, String& error) {
	Value v = GetArg(row, key);
	if (v.IsNull())
		error = String("Required argument '") + key + "' missing for module '" + module_type + "'";
	return v;
}

double GetDouble(Value& row, const char* key, double def) {
	Value v = GetArg(row, key);
	return v.IsNull() ? def : (double)v;
}

bool GetBool(Value& row, const char* key, bool def) {
	Value v = GetArg(row, key);
	return v.IsNull() ? def : (bool)v;
}

int GetInt(Value& row, const char* key, int def) {
	Value v = GetArg(row, key);
	return v.IsNull() ? def : (int)v;
}

bool ParseIntList(const String& text, Vector<int>& out) {
	out.Clear();
	String s = TrimBoth(text);
	if(s.IsEmpty())
		return false;
	if(s.StartsWith("[") && s.EndsWith("]"))
		s = TrimBoth(s.Mid(1, s.GetCount() - 2));
	if(s.IsEmpty())
		return false;
	Vector<String> parts = Split(s, ',');
	for(String part : parts) {
		part = TrimBoth(part);
		if(part.IsEmpty())
			continue;
		const char* end = nullptr;
		int x = ScanInt(~part, &end);
		if(!end)
			return false;
		while(*end && (byte)*end <= ' ')
			end++;
		if(*end)
			return false;
		out.Add(x);
	}
	return !out.IsEmpty();
}

bool GetIntPair(Value& row, const char* key, int out[2]) {
	Value raw = GetArg(row, key);
	if(raw.IsNull())
		return false;

	Vector<int> vec;
	if(IsValueArray(raw)) {
		ValueArray arr = raw;
		for(int i = 0; i < arr.GetCount(); i++)
			vec.Add((int)arr[i]);
	}
	else {
		String s = raw.ToString();
		if(!ParseIntList(s, vec))
			return false;
	}

	if(vec.GetCount() < 2)
		return false;
	out[0] = vec[0];
	out[1] = vec[1];
	return true;
}

SessionModuleParamSpec ParamSpec(const char* key, const char* widget, const Value& def = Value()) {
	SessionModuleParamSpec p;
	p.key = key;
	p.widget = widget;
	p.default_value = def;
	return p;
}

SessionLayerModuleSpec LayerSpec(const char* type, const char* label,
                                 bool has_input, bool has_output,
                                 std::initializer_list<SessionModuleParamSpec> params) {
	SessionLayerModuleSpec s;
	s.type = type;
	s.label = label;
	s.category = "layers";
	s.has_layer_stack_input = has_input;
	s.has_layer_stack_output = has_output;
	for(const SessionModuleParamSpec& p : params)
		s.params.Add(p);
	return s;
}

SessionTrainerModuleSpec TrainerSpec(const char* type, const char* label,
                                     std::initializer_list<SessionModuleParamSpec> params) {
	SessionTrainerModuleSpec s;
	s.type = type;
	s.label = label;
	s.category = "trainers";
	for(const SessionModuleParamSpec& p : params)
		s.params.Add(p);
	return s;
}

bool AddFullyConn(Session& s, Value& row, String& error) {
	Value neuron_count = RequireArg(row, "neuron_count", "fc", error);
	if (!error.IsEmpty()) return false;
	s.AddFullyConnLayer(
		(int)neuron_count,
		GetDouble(row, "l1_decay_mul", 0.0),
		GetDouble(row, "l2_decay_mul", 1.0),
		GetDouble(row, "bias_pref", 0.0));
	return true;
}

bool AddLrn(Session& s, Value& row, String& error) {
	Value k = RequireArg(row, "k", "lrn", error); if (!error.IsEmpty()) return false;
	Value n = RequireArg(row, "n", "lrn", error); if (!error.IsEmpty()) return false;
	Value alpha = RequireArg(row, "alpha", "lrn", error); if (!error.IsEmpty()) return false;
	Value beta = RequireArg(row, "beta", "lrn", error); if (!error.IsEmpty()) return false;
	s.AddLrnLayer((double)k, (int)n, (double)alpha, (double)beta);
	return true;
}

bool AddDropout(Session& s, Value& row, String& error) {
	Value drop_prob = RequireArg(row, "drop_prob", "dropout", error);
	if (!error.IsEmpty()) return false;
	s.AddDropoutLayer((double)drop_prob);
	return true;
}

bool AddInput(Session& s, Value& row, String& error) {
	Value input_width = RequireArg(row, "input_width", "input", error); if (!error.IsEmpty()) return false;
	Value input_height = RequireArg(row, "input_height", "input", error); if (!error.IsEmpty()) return false;
	Value input_depth = RequireArg(row, "input_depth", "input", error); if (!error.IsEmpty()) return false;
	s.AddInputLayer((int)input_width, (int)input_height, (int)input_depth);
	return true;
}

bool AddSoftmax(Session& s, Value& row, String& error) {
	Value class_count = RequireArg(row, "class_count", "softmax", error);
	if (!error.IsEmpty()) return false;
	s.AddSoftmaxLayer((int)class_count);
	return true;
}

bool AddRegression(Session& s, Value&, String&) {
	s.AddRegressionLayer();
	return true;
}

bool AddHeteroscedasticRegression(Session& s, Value&, String&) {
	s.AddHeteroscedasticRegressionLayer();
	return true;
}

bool AddConv(Session& s, Value& row, String& error) {
	Value width = RequireArg(row, "width", "conv", error); if (!error.IsEmpty()) return false;
	Value height = RequireArg(row, "height", "conv", error); if (!error.IsEmpty()) return false;
	Value filter_count = RequireArg(row, "filter_count", "conv", error); if (!error.IsEmpty()) return false;
	s.AddConvLayer(
		(int)width, (int)height, (int)filter_count,
		GetDouble(row, "l1_decay_mul", 0.0),
		GetDouble(row, "l2_decay_mul", 1.0),
		GetInt(row, "stride", 1),
		GetInt(row, "pad", 0),
		GetDouble(row, "bias_pref", 0.0));
	return true;
}

bool AddDeconv(Session& s, Value& row, String& error) {
	Value width = RequireArg(row, "width", "deconv", error); if (!error.IsEmpty()) return false;
	Value height = RequireArg(row, "height", "deconv", error); if (!error.IsEmpty()) return false;
	Value filter_count = RequireArg(row, "filter_count", "deconv", error); if (!error.IsEmpty()) return false;
	s.AddDeconvLayer(
		(int)width, (int)height, (int)filter_count,
		GetDouble(row, "l1_decay_mul", 0.0),
		GetDouble(row, "l2_decay_mul", 1.0),
		GetInt(row, "stride", 1),
		GetInt(row, "pad", 0),
		GetDouble(row, "bias_pref", 0.0));
	return true;
}

bool AddPool(Session& s, Value& row, String& error) {
	Value width = RequireArg(row, "width", "pool", error); if (!error.IsEmpty()) return false;
	Value height = RequireArg(row, "height", "pool", error); if (!error.IsEmpty()) return false;
	s.AddPoolLayer((int)width, (int)height, GetInt(row, "stride", 2), GetInt(row, "pad", 0));
	return true;
}

bool AddUnpool(Session& s, Value& row, String& error) {
	Value width = RequireArg(row, "width", "unpool", error); if (!error.IsEmpty()) return false;
	Value height = RequireArg(row, "height", "unpool", error); if (!error.IsEmpty()) return false;
	s.AddUnpoolLayer((int)width, (int)height, GetInt(row, "stride", 2), GetInt(row, "pad", 0));
	return true;
}

bool AddRelu(Session& s, Value&, String&) {
	s.AddReluLayer();
	return true;
}

bool AddSigmoid(Session& s, Value&, String&) {
	s.AddSigmoidLayer();
	return true;
}

bool AddTanh(Session& s, Value&, String&) {
	s.AddTanhLayer();
	return true;
}

bool AddMaxout(Session& s, Value& row, String& error) {
	Value group_size = RequireArg(row, "group_size", "maxout", error);
	if (!error.IsEmpty()) return false;
	s.AddMaxoutLayer((int)group_size);
	return true;
}

bool AddSvm(Session& s, Value& row, String& error) {
	Value class_count = RequireArg(row, "class_count", "svm", error);
	if (!error.IsEmpty()) return false;
	s.AddSVMLayer((int)class_count);
	return true;
}

bool AddViTPatchEmbedding(Session& s, Value& row, String& error) {
	Value patch_size = RequireArg(row, "patch_size", "vit_patch_embed", error); if (!error.IsEmpty()) return false;
	Value embed_dim = RequireArg(row, "embed_dim", "vit_patch_embed", error); if (!error.IsEmpty()) return false;
	Value num_patches = RequireArg(row, "num_patches", "vit_patch_embed", error); if (!error.IsEmpty()) return false;
	s.AddViTPatchEmbeddingLayer((int)patch_size, (int)embed_dim, (int)num_patches);
	return true;
}

bool AddViTEncoder(Session& s, Value& row, String& error) {
	Value embed_dim = RequireArg(row, "embed_dim", "vit_encoder", error); if (!error.IsEmpty()) return false;
	Value num_heads = RequireArg(row, "num_heads", "vit_encoder", error); if (!error.IsEmpty()) return false;
	Value ff_dim = RequireArg(row, "ff_dim", "vit_encoder", error); if (!error.IsEmpty()) return false;
	Value num_layers = RequireArg(row, "num_layers", "vit_encoder", error); if (!error.IsEmpty()) return false;
	s.AddViTEncoderLayer((int)embed_dim, (int)num_heads, (int)ff_dim, (int)num_layers, GetDouble(row, "dropout_rate", 0.1));
	return true;
}

bool AddViTClassifier(Session& s, Value& row, String& error) {
	Value num_classes = RequireArg(row, "num_classes", "vit_classifier", error); if (!error.IsEmpty()) return false;
	Value embed_dim = RequireArg(row, "embed_dim", "vit_classifier", error); if (!error.IsEmpty()) return false;
	s.AddViTClassifierLayer((int)num_classes, (int)embed_dim);
	return true;
}

bool AddSwinPatchMerging(Session& s, Value& row, String& error) {
	Value dim = RequireArg(row, "dim", "swin_patch_merge", error); if (!error.IsEmpty()) return false;
	Value out_dim = RequireArg(row, "out_dim", "swin_patch_merge", error); if (!error.IsEmpty()) return false;
	s.AddSwinPatchMergingLayer((int)dim, (int)out_dim);
	return true;
}

bool AddWindowAttention(Session& s, Value& row, String& error) {
	Value window_size = RequireArg(row, "window_size", "window_attention", error); if (!error.IsEmpty()) return false;
	Value num_heads = RequireArg(row, "num_heads", "window_attention", error); if (!error.IsEmpty()) return false;
	Value input_dim = RequireArg(row, "input_dim", "window_attention", error); if (!error.IsEmpty()) return false;
	s.AddWindowAttentionLayer((int)window_size, (int)num_heads, (int)input_dim);
	return true;
}

bool AddSwinBlock(Session& s, Value& row, String& error) {
	Value dim = RequireArg(row, "dim", "swin_block", error); if (!error.IsEmpty()) return false;
	Value num_heads = RequireArg(row, "num_heads", "swin_block", error); if (!error.IsEmpty()) return false;

	int input_resolution[2] = {0, 0};
	if (!GetIntPair(row, "input_resolution", input_resolution)) {
		error = "Required argument 'input_resolution' missing or invalid for module 'swin_block'";
		return false;
	}

	s.AddSwinTransformerBlockLayer(
		(int)dim,
		input_resolution,
		(int)num_heads,
		GetInt(row, "window_size", 7),
		GetInt(row, "shift_size", 0),
		GetInt(row, "mlp_ratio", 4),
		GetBool(row, "mlp_bias", true),
		GetDouble(row, "mlp_dropout", 0.0));
	return true;
}

bool AddMaskedAttention(Session& s, Value& row, String& error) {
	Value embed_dim = RequireArg(row, "embed_dim", "masked_attention", error); if (!error.IsEmpty()) return false;
	Value num_heads = RequireArg(row, "num_heads", "masked_attention", error); if (!error.IsEmpty()) return false;
	s.AddMaskedMultiHeadAttentionLayer((int)embed_dim, (int)num_heads);
	return true;
}

bool SetTrainerAdadelta(TrainerBase& trainer, Value&, String&) {
	trainer.SetType(TRAINER_ADADELTA);
	return true;
}

bool SetTrainerAdagrad(TrainerBase& trainer, Value&, String&) {
	trainer.SetType(TRAINER_ADAGRAD);
	return true;
}

bool SetTrainerAdam(TrainerBase& trainer, Value&, String&) {
	trainer.SetType(TRAINER_ADAM);
	return true;
}

bool SetTrainerNetsterov(TrainerBase& trainer, Value&, String&) {
	trainer.SetType(TRAINER_NETSTEROV);
	return true;
}

bool SetTrainerSgd(TrainerBase& trainer, Value&, String&) {
	trainer.SetType(TRAINER_SGD);
	return true;
}

bool SetTrainerWindowgrad(TrainerBase& trainer, Value&, String&) {
	trainer.SetType(TRAINER_WINDOWGRAD);
	return true;
}

}

void RegisterBuiltinSessionModules() {
	static bool initialized = false;
	if (initialized) return;
	initialized = true;

	SessionModuleRegistry& reg = SessionModuleRegistry::Get();

	reg.RegisterLayer("fc", AddFullyConn, LayerSpec("fc", "Fully Connected", true, true, {
		ParamSpec("neuron_count", "EditIntSpin", 128),
		ParamSpec("l1_decay_mul", "EditDoubleSpin", 0.0),
		ParamSpec("l2_decay_mul", "EditDoubleSpin", 1.0),
		ParamSpec("bias_pref", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterLayer("fully_connected", AddFullyConn, LayerSpec("fully_connected", "Fully Connected", true, true, {
		ParamSpec("neuron_count", "EditIntSpin", 128),
		ParamSpec("l1_decay_mul", "EditDoubleSpin", 0.0),
		ParamSpec("l2_decay_mul", "EditDoubleSpin", 1.0),
		ParamSpec("bias_pref", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterLayer("fullyconn", AddFullyConn, LayerSpec("fullyconn", "Fully Connected", true, true, {
		ParamSpec("neuron_count", "EditIntSpin", 128),
		ParamSpec("l1_decay_mul", "EditDoubleSpin", 0.0),
		ParamSpec("l2_decay_mul", "EditDoubleSpin", 1.0),
		ParamSpec("bias_pref", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterLayer("dense", AddFullyConn, LayerSpec("dense", "Dense", true, true, {
		ParamSpec("neuron_count", "EditIntSpin", 128),
		ParamSpec("l1_decay_mul", "EditDoubleSpin", 0.0),
		ParamSpec("l2_decay_mul", "EditDoubleSpin", 1.0),
		ParamSpec("bias_pref", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterLayer("lrn", AddLrn, LayerSpec("lrn", "Local Response Norm", true, true, {
		ParamSpec("k", "EditDoubleSpin", 2.0),
		ParamSpec("n", "EditIntSpin", 5),
		ParamSpec("alpha", "EditDoubleSpin", 0.0001),
		ParamSpec("beta", "EditDoubleSpin", 0.75),
	}));
	reg.RegisterLayer("dropout", AddDropout, LayerSpec("dropout", "Dropout", true, true, {
		ParamSpec("drop_prob", "EditDoubleSpin", 0.5),
	}));
	reg.RegisterLayer("input", AddInput, LayerSpec("input", "Input", false, true, {
		ParamSpec("input_width", "EditIntSpin", 28),
		ParamSpec("input_height", "EditIntSpin", 28),
		ParamSpec("input_depth", "EditIntSpin", 1),
	}));
	reg.RegisterLayer("softmax", AddSoftmax, LayerSpec("softmax", "Softmax", true, true, {
		ParamSpec("class_count", "EditIntSpin", 10),
	}));
	reg.RegisterLayer("regression", AddRegression,
	                  LayerSpec("regression", "Regression", true, true, {}));
	reg.RegisterLayer("heteroscedastic_regression", AddHeteroscedasticRegression,
	                  LayerSpec("heteroscedastic_regression", "Heteroscedastic Regression", true, true, {}));
	reg.RegisterLayer("conv", AddConv, LayerSpec("conv", "Convolution", true, true, {
		ParamSpec("width", "EditIntSpin", 3),
		ParamSpec("height", "EditIntSpin", 3),
		ParamSpec("filter_count", "EditIntSpin", 16),
		ParamSpec("stride", "EditIntSpin", 1),
		ParamSpec("pad", "EditIntSpin", 0),
		ParamSpec("l1_decay_mul", "EditDoubleSpin", 0.0),
		ParamSpec("l2_decay_mul", "EditDoubleSpin", 1.0),
		ParamSpec("bias_pref", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterLayer("deconv", AddDeconv, LayerSpec("deconv", "Deconvolution", true, true, {
		ParamSpec("width", "EditIntSpin", 3),
		ParamSpec("height", "EditIntSpin", 3),
		ParamSpec("filter_count", "EditIntSpin", 16),
		ParamSpec("stride", "EditIntSpin", 1),
		ParamSpec("pad", "EditIntSpin", 0),
		ParamSpec("l1_decay_mul", "EditDoubleSpin", 0.0),
		ParamSpec("l2_decay_mul", "EditDoubleSpin", 1.0),
		ParamSpec("bias_pref", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterLayer("conv_transpose", AddDeconv, LayerSpec("conv_transpose", "Conv Transpose", true, true, {
		ParamSpec("width", "EditIntSpin", 3),
		ParamSpec("height", "EditIntSpin", 3),
		ParamSpec("filter_count", "EditIntSpin", 16),
		ParamSpec("stride", "EditIntSpin", 1),
		ParamSpec("pad", "EditIntSpin", 0),
		ParamSpec("l1_decay_mul", "EditDoubleSpin", 0.0),
		ParamSpec("l2_decay_mul", "EditDoubleSpin", 1.0),
		ParamSpec("bias_pref", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterLayer("pool", AddPool, LayerSpec("pool", "Pooling", true, true, {
		ParamSpec("width", "EditIntSpin", 2),
		ParamSpec("height", "EditIntSpin", 2),
		ParamSpec("stride", "EditIntSpin", 2),
		ParamSpec("pad", "EditIntSpin", 0),
	}));
	reg.RegisterLayer("unpool", AddUnpool, LayerSpec("unpool", "Unpool", true, true, {
		ParamSpec("width", "EditIntSpin", 2),
		ParamSpec("height", "EditIntSpin", 2),
		ParamSpec("stride", "EditIntSpin", 2),
		ParamSpec("pad", "EditIntSpin", 0),
	}));
	reg.RegisterLayer("relu", AddRelu, LayerSpec("relu", "ReLU", true, true, {}));
	reg.RegisterLayer("sigmoid", AddSigmoid, LayerSpec("sigmoid", "Sigmoid", true, true, {}));
	reg.RegisterLayer("tanh", AddTanh, LayerSpec("tanh", "Tanh", true, true, {}));
	reg.RegisterLayer("maxout", AddMaxout, LayerSpec("maxout", "Maxout", true, true, {
		ParamSpec("group_size", "EditIntSpin", 2),
	}));
	reg.RegisterLayer("svm", AddSvm, LayerSpec("svm", "SVM", true, true, {
		ParamSpec("class_count", "EditIntSpin", 10),
	}));
	reg.RegisterLayer("vit_patch_embed", AddViTPatchEmbedding, LayerSpec("vit_patch_embed", "ViT Patch Embedding", true, true, {
		ParamSpec("patch_size", "EditIntSpin", 4),
		ParamSpec("embed_dim", "EditIntSpin", 64),
		ParamSpec("num_patches", "EditIntSpin", 49),
	}));
	reg.RegisterLayer("vit_encoder", AddViTEncoder, LayerSpec("vit_encoder", "ViT Encoder", true, true, {
		ParamSpec("embed_dim", "EditIntSpin", 64),
		ParamSpec("num_heads", "EditIntSpin", 4),
		ParamSpec("ff_dim", "EditIntSpin", 128),
		ParamSpec("num_layers", "EditIntSpin", 4),
		ParamSpec("dropout_rate", "EditDoubleSpin", 0.1),
	}));
	reg.RegisterLayer("vit_classifier", AddViTClassifier, LayerSpec("vit_classifier", "ViT Classifier", true, true, {
		ParamSpec("num_classes", "EditIntSpin", 10),
		ParamSpec("embed_dim", "EditIntSpin", 64),
	}));
	reg.RegisterLayer("swin_patch_merge", AddSwinPatchMerging, LayerSpec("swin_patch_merge", "Swin Patch Merge", true, true, {
		ParamSpec("dim", "EditIntSpin", 96),
		ParamSpec("out_dim", "EditIntSpin", 192),
	}));
	reg.RegisterLayer("window_attention", AddWindowAttention, LayerSpec("window_attention", "Window Attention", true, true, {
		ParamSpec("window_size", "EditIntSpin", 7),
		ParamSpec("num_heads", "EditIntSpin", 3),
		ParamSpec("input_dim", "EditIntSpin", 96),
	}));
	reg.RegisterLayer("swin_block", AddSwinBlock, LayerSpec("swin_block", "Swin Block", true, true, {
		ParamSpec("dim", "EditIntSpin", 96),
		ParamSpec("input_resolution", "EditString", "[7,7]"),
		ParamSpec("num_heads", "EditIntSpin", 3),
		ParamSpec("window_size", "EditIntSpin", 7),
		ParamSpec("shift_size", "EditIntSpin", 0),
		ParamSpec("mlp_ratio", "EditIntSpin", 4),
		ParamSpec("mlp_bias", "Option", true),
		ParamSpec("mlp_dropout", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterLayer("masked_attention", AddMaskedAttention, LayerSpec("masked_attention", "Masked Attention", true, true, {
		ParamSpec("embed_dim", "EditIntSpin", 64),
		ParamSpec("num_heads", "EditIntSpin", 4),
	}));

	reg.RegisterTrainer("adadelta", SetTrainerAdadelta, TrainerSpec("adadelta", "Adadelta", {
		ParamSpec("learning_rate", "EditDoubleSpin", 1.0),
		ParamSpec("eps", "EditDoubleSpin", 0.000001),
		ParamSpec("ro", "EditDoubleSpin", 0.95),
		ParamSpec("batch_size", "EditIntSpin", 1),
		ParamSpec("l2_decay", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterTrainer("adagrad", SetTrainerAdagrad, TrainerSpec("adagrad", "Adagrad", {
		ParamSpec("learning_rate", "EditDoubleSpin", 0.01),
		ParamSpec("eps", "EditDoubleSpin", 0.000001),
		ParamSpec("batch_size", "EditIntSpin", 1),
		ParamSpec("l2_decay", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterTrainer("adam", SetTrainerAdam, TrainerSpec("adam", "Adam", {
		ParamSpec("learning_rate", "EditDoubleSpin", 0.001),
		ParamSpec("Beta1", "EditDoubleSpin", 0.9),
		ParamSpec("Beta2", "EditDoubleSpin", 0.999),
		ParamSpec("eps", "EditDoubleSpin", 0.00000001),
		ParamSpec("batch_size", "EditIntSpin", 1),
		ParamSpec("l2_decay", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterTrainer("netsterov", SetTrainerNetsterov, TrainerSpec("netsterov", "Netsterov", {
		ParamSpec("learning_rate", "EditDoubleSpin", 0.01),
		ParamSpec("momentum", "EditDoubleSpin", 0.9),
		ParamSpec("batch_size", "EditIntSpin", 1),
		ParamSpec("l2_decay", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterTrainer("nesterov", SetTrainerNetsterov, TrainerSpec("nesterov", "Nesterov", {
		ParamSpec("learning_rate", "EditDoubleSpin", 0.01),
		ParamSpec("momentum", "EditDoubleSpin", 0.9),
		ParamSpec("batch_size", "EditIntSpin", 1),
		ParamSpec("l2_decay", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterTrainer("sgd", SetTrainerSgd, TrainerSpec("sgd", "SGD", {
		ParamSpec("learning_rate", "EditDoubleSpin", 0.01),
		ParamSpec("momentum", "EditDoubleSpin", 0.0),
		ParamSpec("batch_size", "EditIntSpin", 1),
		ParamSpec("l2_decay", "EditDoubleSpin", 0.0),
	}));
	reg.RegisterTrainer("windowgrad", SetTrainerWindowgrad, TrainerSpec("windowgrad", "Windowgrad", {
		ParamSpec("learning_rate", "EditDoubleSpin", 0.01),
		ParamSpec("eps", "EditDoubleSpin", 0.000001),
		ParamSpec("ro", "EditDoubleSpin", 0.95),
		ParamSpec("batch_size", "EditIntSpin", 1),
		ParamSpec("l2_decay", "EditDoubleSpin", 0.0),
	}));
}

}
