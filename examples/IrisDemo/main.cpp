// IrisDemo — trains a 3-layer neural net on the Iris dataset,
// prints a live loss/accuracy training curve, and reports final confusion matrix.

#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

// ---- data loading -------------------------------------------------------

struct Sample {
	double x[4];
	int    label;
};

static bool LoadIris(const String& path, Vector<Sample>& out, Vector<String>& class_names) {
	String data = LoadFile(path);
	if(data.IsEmpty()) return false;
	Index<String> names;
	for(String& line : Split(data, '\n')) {
		line = TrimBoth(line);
		if(line.IsEmpty()) continue;
		Vector<String> parts = Split(line, ',');
		if(parts.GetCount() < 5) continue;
		Sample s;
		for(int i = 0; i < 4; i++) s.x[i] = ScanDouble(parts[i]);
		s.label = names.FindAdd(TrimBoth(parts[4]));
		out.Add(s);
	}
	class_names.Clear();
	for(int i = 0; i < names.GetCount(); i++) class_names.Add(names[i]);
	return !out.IsEmpty();
}

static void Normalise(Vector<Sample>& data) {
	double mn[4] = {1e9,1e9,1e9,1e9}, mx[4] = {-1e9,-1e9,-1e9,-1e9};
	for(auto& s : data)
		for(int i = 0; i < 4; i++) { mn[i]=min(mn[i],s.x[i]); mx[i]=max(mx[i],s.x[i]); }
	for(auto& s : data)
		for(int i = 0; i < 4; i++) {
			double r = mx[i] - mn[i];
			s.x[i] = r > 1e-9 ? (s.x[i] - mn[i]) / r : 0.0;
		}
}

// ---- bar helpers --------------------------------------------------------

static String Bar(double v, int width = 40) {
	int filled = max(0, min(width, (int)(v * width)));
	String s;
	s << "[";
	for(int i = 0; i < width; i++) s << (i < filled ? '#' : '.');
	s << "]";
	return s;
}

static String Pad(const String& s, int w) {
	String r = s;
	while(r.GetCount() < w) r << ' ';
	return r;
}

// ---- main ---------------------------------------------------------------

CONSOLE_APP_MAIN {
	String iris_path = AppendFileName(GetFileDirectory(GetExeFilePath()),
	                                  "../../examples/NetworkOptimization/DataIris.txt");
	if(!FileExists(iris_path))
		iris_path = "examples/NetworkOptimization/DataIris.txt";

	Vector<Sample> data;
	Vector<String> class_names;
	if(!LoadIris(iris_path, data, class_names)) {
		Cout() << "ERROR: could not load " << iris_path << "\n";
		SetExitCode(1);
		return;
	}

	Normalise(data);

	// Shuffle deterministically
	for(int i = data.GetCount()-1; i > 0; i--) {
		int j = (i * 6364136223846793005LL + 1442695040888963407LL) % (i+1);
		if(j < 0) j += i+1;
		Swap(data[i], data[j]);
	}

	// 80/20 split
	int n_train = (int)(data.GetCount() * 0.8);
	Vector<Sample> train_data, test_data;
	for(int i = 0; i < data.GetCount(); i++)
		(i < n_train ? train_data : test_data).Add(data[i]);

	int n_class = class_names.GetCount();
	Cout() << "\n";
	Cout() << "═══════════════════════════════════════════════════════════\n";
	Cout() << "  Iris Classifier Demo — ConvNet C++\n";
	Cout() << "═══════════════════════════════════════════════════════════\n";
	Cout() << "  Dataset : " << data.GetCount() << " samples ("
	       << train_data.GetCount() << " train / " << test_data.GetCount() << " test)\n";
	Cout() << "  Classes : ";
	for(int i = 0; i < class_names.GetCount(); i++)
		Cout() << class_names[i] << (i+1<class_names.GetCount() ? ", " : "\n");
	Cout() << "  Network : 4→16→relu→16→relu→3→softmax\n";
	Cout() << "  Trainer : SGD  lr=0.05  momentum=0.9  batch=10\n";
	Cout() << "═══════════════════════════════════════════════════════════\n\n";

	// Build network
	Session s;
	s.MakeLayers(Format(
		"[{\"type\":\"input\",\"input_width\":1,\"input_height\":1,\"input_depth\":4},"
		"{\"type\":\"fc\",\"neuron_count\":16,\"activation\":\"relu\"},"
		"{\"type\":\"fc\",\"neuron_count\":16,\"activation\":\"relu\"},"
		"{\"type\":\"softmax\",\"class_count\":%d}]", n_class));
	s.GetTrainer().SetType(TRAINER_SGD);
	s.GetTrainer().SetLearningRate(0.05);
	s.GetTrainer().SetMomentum(0.9);
	s.GetTrainer().SetBatchSize(10);

	int epochs = 200;
	int report_every = 20;

	Cout() << Pad("Epoch", 7)
	       << Pad("Loss", 10)
	       << Pad("Train acc", 11)
	       << Pad("Test acc", 10)
	       << "Progress\n";
	Cout() << String('-', 70) << "\n";

	Volume vol;
	vol.Init(1, 1, 4, 0.0);

	auto EvalAccuracy = [&](const Vector<Sample>& ds) -> double {
		int correct = 0;
		for(auto& sample : ds) {
			for(int i = 0; i < 4; i++) vol.Set(i, sample.x[i]);
			Vector<double> probs = s.Predict(vol.GetWeights());
			int pred = 0;
			for(int i = 1; i < probs.GetCount(); i++)
				if(probs[i] > probs[pred]) pred = i;
			if(pred == sample.label) correct++;
		}
		return ds.IsEmpty() ? 0.0 : (double)correct / ds.GetCount();
	};

	double best_test_acc = 0;
	for(int epoch = 1; epoch <= epochs; epoch++) {
		// Shuffle training data each epoch
		for(int i = train_data.GetCount()-1; i > 0; i--) {
			int j = (epoch * 6364136223846793005LL * (i+1) + 1442695040888963407LL) % (i+1);
			if(j < 0) j += i+1;
			Swap(train_data[i], train_data[j]);
		}
		double epoch_loss = 0;
		for(auto& sample : train_data) {
			for(int i = 0; i < 4; i++) vol.Set(i, sample.x[i]);
			s.GetTrainer().Train(vol, sample.label, 1.0);
			epoch_loss += s.GetTrainer().GetLoss();
		}
		epoch_loss /= train_data.GetCount();

		if(epoch % report_every == 0 || epoch == 1 || epoch == epochs) {
			double train_acc = EvalAccuracy(train_data);
			double test_acc  = EvalAccuracy(test_data);
			if(test_acc > best_test_acc) best_test_acc = test_acc;
			Cout() << Pad(Format("%d", epoch), 7)
			       << Pad(Format("%.4f", epoch_loss), 10)
			       << Pad(Format("%.1f%%", train_acc*100), 11)
			       << Pad(Format("%.1f%%", test_acc*100), 10)
			       << Bar(test_acc) << "\n";
		}
	}

	// Confusion matrix
	double test_acc = EvalAccuracy(test_data);
	Cout() << "\n";
	Cout() << "═══════════════════════════════════════════════════════════\n";
	Cout() << "  Final test accuracy : " << Format("%.1f%%", test_acc*100)
	       << "  (best: " << Format("%.1f%%", best_test_acc*100) << ")\n";
	Cout() << "═══════════════════════════════════════════════════════════\n";

	// Confusion matrix
	int cm[3][3] = {};
	for(auto& sample : test_data) {
		for(int i = 0; i < 4; i++) vol.Set(i, sample.x[i]);
		Vector<double> probs = s.Predict(vol.GetWeights());
		int pred = 0;
		for(int i = 1; i < probs.GetCount(); i++)
			if(probs[i] > probs[pred]) pred = i;
		if(sample.label < n_class && pred < n_class)
			cm[sample.label][pred]++;
	}
	Cout() << "\n  Confusion matrix (rows=actual, cols=predicted):\n\n";
	int name_w = 18;
	Cout() << Pad("", name_w);
	for(int j = 0; j < n_class; j++) Cout() << Pad(class_names[j].Left(12), 14);
	Cout() << "\n";
	for(int i = 0; i < n_class; i++) {
		Cout() << Pad(class_names[i].Left(16), name_w);
		for(int j = 0; j < n_class; j++) {
			String cell = Format("%d", cm[i][j]);
			if(i == j) cell = "[" + cell + "]";
			Cout() << Pad(cell, 14);
		}
		Cout() << "\n";
	}
	Cout() << "\n";

	SetExitCode(test_acc >= 0.85 ? 0 : 1);
}
