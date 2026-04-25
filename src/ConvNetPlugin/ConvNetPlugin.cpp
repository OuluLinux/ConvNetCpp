#include "ConvNetPlugin.h"

NAMESPACE_UPP

namespace {

using namespace ConvNet;

Vector<double> PyListToDoubleVec(const PyValue& pv) {
	Vector<double> out;
	if(pv.GetType() == PY_LIST || pv.GetType() == PY_TUPLE) {
		const Vector<PyValue>& arr = pv.GetArray();
		out.SetCount(arr.GetCount());
		for(int i = 0; i < arr.GetCount(); i++)
			out[i] = arr[i].AsDouble();
	}
	return out;
}

// convnet.net_create(json_layers_string)
// json_layers_string: JSON array of layer descriptors, e.g.:
//   '[{"type":"input","out_sx":1,"out_sy":1,"out_depth":2}, ...]'
// Optional second arg: trainer config JSON
//   '{"type":"sgd","learning_rate":0.01,"momentum":0.9}'
PyValue FnNetCreate(const Vector<PyValue>& args, void* ud) {
	ConvNetPlugin* p = (ConvNetPlugin*)ud;
	if(!p || args.GetCount() < 1) return PyValue();
	Session& s = p->GetSession();
	s.Clear();
	s.MakeLayers(args[0].ToString());
	s.GetTrainer().SetType(TRAINER_SGD);
	if(args.GetCount() >= 2) {
		Value cfg = ParseJSON(args[1].ToString());
		if(cfg.Is<ValueMap>()) {
			const ValueMap& m = cfg;
			int ti = m.Find("type");
			if(ti >= 0) {
				String type = ToLower((String)m.GetValue(ti));
				if(type == "adadelta") s.GetTrainer().SetType(TRAINER_ADADELTA);
				else if(type == "adagrad") s.GetTrainer().SetType(TRAINER_ADAGRAD);
				else if(type == "adam") s.GetTrainer().SetType(TRAINER_ADAM);
				else if(type == "netsterov") s.GetTrainer().SetType(TRAINER_NETSTEROV);
				else if(type == "windowgrad") s.GetTrainer().SetType(TRAINER_WINDOWGRAD);
				else s.GetTrainer().SetType(TRAINER_SGD);
			}
			auto dget = [&](const char* k, double def) {
				int i = m.Find(k); return i >= 0 ? (double)m.GetValue(i) : def;
			};
			auto iget = [&](const char* k, int def) {
				int i = m.Find(k); return i >= 0 ? (int)m.GetValue(i) : def;
			};
			s.GetTrainer().SetLearningRate(dget("learning_rate", s.GetTrainer().GetLearningRate()));
			s.GetTrainer().SetMomentum(dget("momentum",       s.GetTrainer().GetMomentum()));
			s.GetTrainer().SetBatchSize(iget("batch_size",    s.GetTrainer().GetBatchSize()));
			s.GetTrainer().SetL2Decay(dget("l2_decay",        s.GetTrainer().GetL2Decay()));
			s.GetTrainer().SetL1Decay(dget("l1_decay",        s.GetTrainer().GetL1Decay()));
		}
	}
	return PyValue();
}

// convnet.train(x_list, class_label)  — classification: label is int
// convnet.train(x_list, target_value) — regression: target is float
// Returns loss.
PyValue FnTrain(const Vector<PyValue>& args, void* ud) {
	ConvNetPlugin* p = (ConvNetPlugin*)ud;
	if(!p || args.GetCount() < 2) return PyValue(0.0);
	Session& s = p->GetSession();

	Vector<double> xvec = PyListToDoubleVec(args[0]);
	if(xvec.IsEmpty()) return PyValue(0.0);

	Volume vol;
	vol.Init(xvec.GetCount(), 1, 1, 0.0);
	for(int i = 0; i < xvec.GetCount(); i++) vol.Set(i, xvec[i]);

	if(args[1].GetType() == PY_INT) {
		int label = args[1].AsInt();
		s.GetTrainer().Train(vol, label, 1.0);
	} else {
		Vector<double> y;
		y.Add(args[1].AsDouble());
		s.GetTrainer().Train(vol, y);
	}
	return PyValue(s.GetTrainer().GetLoss());
}

// convnet.forward(x_list) — returns list of output values
PyValue FnForward(const Vector<PyValue>& args, void* ud) {
	ConvNetPlugin* p = (ConvNetPlugin*)ud;
	if(!p || args.GetCount() < 1) return PyValue::List();
	Session& s = p->GetSession();

	Vector<double> xvec = PyListToDoubleVec(args[0]);
	if(xvec.IsEmpty()) return PyValue::List();

	Vector<double> result = s.Predict(xvec);

	PyValue out = PyValue::List();
	for(int i = 0; i < result.GetCount(); i++)
		out.Add(PyValue(result[i]));
	return out;
}

// convnet.get_loss() — returns current loss average
PyValue FnGetLoss(const Vector<PyValue>&, void* ud) {
	ConvNetPlugin* p = (ConvNetPlugin*)ud;
	return p ? PyValue(p->GetSession().GetLossAverage()) : PyValue(0.0);
}

// convnet.save(path) — serialize session to file
PyValue FnSave(const Vector<PyValue>& args, void* ud) {
	ConvNetPlugin* p = (ConvNetPlugin*)ud;
	if(!p || args.GetCount() < 1) return PyValue(0);
	FileOut out(args[0].ToString());
	if(!out.IsOpen()) return PyValue(0);
	p->GetSession().Serialize(out);
	return PyValue(out.IsError() ? 0 : 1);
}

// convnet.load(path) — deserialize session from file
PyValue FnLoad(const Vector<PyValue>& args, void* ud) {
	ConvNetPlugin* p = (ConvNetPlugin*)ud;
	if(!p || args.GetCount() < 1) return PyValue(0);
	FileIn in(args[0].ToString());
	if(!in.IsOpen()) return PyValue(0);
	p->GetSession().Serialize(in);
	return PyValue(in.IsError() ? 0 : 1);
}

// convnet.reset() — clear all layers and trainer state
PyValue FnReset(const Vector<PyValue>&, void* ud) {
	ConvNetPlugin* p = (ConvNetPlugin*)ud;
	if(p) p->GetSession().Clear();
	return PyValue();
}

} // namespace

void ConvNetPlugin::Init(IPluginContext& ctx) {
	context = &ctx;
	if(auto* reg = dynamic_cast<IPluginRegistry*>(&ctx))
		reg->RegisterPythonBindingProvider(*this);
}

void ConvNetPlugin::Shutdown() {
	context = nullptr;
	session.Clear();
}

void ConvNetPlugin::SyncBindings(PyVM& vm) {
	PyValue m = PyValue::Dict();
	m.SetItem(PyValue("net_create"), PyValue::Function("net_create", &FnNetCreate, this));
	m.SetItem(PyValue("train"),      PyValue::Function("train",      &FnTrain,     this));
	m.SetItem(PyValue("forward"),    PyValue::Function("forward",    &FnForward,   this));
	m.SetItem(PyValue("get_loss"),   PyValue::Function("get_loss",   &FnGetLoss,   this));
	m.SetItem(PyValue("save"),       PyValue::Function("save",       &FnSave,      this));
	m.SetItem(PyValue("load"),       PyValue::Function("load",       &FnLoad,      this));
	m.SetItem(PyValue("reset"),      PyValue::Function("reset",      &FnReset,     this));
	vm.GetGlobalsRW().SetItem(PyValue("convnet"), m);
}

END_UPP_NAMESPACE
