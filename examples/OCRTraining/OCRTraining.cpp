#include "OCRTraining.h"
#include <OCR/OCRTesting.h>

namespace Upp {

OCRTraining::OCRTraining() {
	Title("OCR Training Tool");
	Sizeable().Zoomable();
	
	SetRect(0, 0, 800, 600);
	
	running = false;
	stopped = true;
	mode = MODE_CLASSIFIER;
	
	// Network Editor
	net_ctrl.Add(net_edit.SizePos());
	net_edit.SetFont(Monospace(12));
	
	// Control Panel
	control_panel.Add(mode_select.LeftPos(5, 120).TopPos(5, 25));
	mode_select.Add(MODE_CLASSIFIER, "Classifier");
	mode_select.Add(MODE_SPLITTER, "Splitter");
	mode_select.SetData(MODE_CLASSIFIER);
	mode_select << [this] { SetMode(mode_select.GetData()); };
	
	control_panel.Add(btn_load_dataset.LeftPos(5, 120).TopPos(35, 25));
	btn_load_dataset.SetLabel("Load Dataset...");
	btn_load_dataset << THISBACK(LoadDatasetProject);
	
	control_panel.Add(btn_start_training.LeftPos(5, 120).TopPos(65, 25));
	btn_start_training.SetLabel("Start Training");
	btn_start_training << THISBACK(StartTraining);
	
	control_panel.Add(btn_stop_training.LeftPos(5, 120).TopPos(95, 25));
	btn_stop_training.SetLabel("Stop Training");
	btn_stop_training << THISBACK(StopTraining);
	
	control_panel.Add(btn_reset_network.LeftPos(5, 120).TopPos(125, 25));
	btn_reset_network.SetLabel("Reset Network");
	btn_reset_network << THISBACK(ResetNetwork);
	
	control_panel.Add(btn_export.LeftPos(5, 120).TopPos(155, 25));
	btn_export.SetLabel("Export Model...");
	btn_export << THISBACK(ExportModel);
	
	control_panel.Add(btn_test.LeftPos(5, 120).TopPos(185, 25));
	btn_test.SetLabel("Run Batch Test");
	btn_test << THISBACK(RunBatchTest);
	
	control_panel.Add(status_label.LeftPos(5, 200).TopPos(220, 60));
	status_label.SetText("No dataset loaded");
	
	// Layer Visualizer
	lctrl.SetSession(session);
	lctrl.WhenViewLayer = THISBACK(ViewLayer);
	
	PostCallback(THISBACK(DockInit));
	
	SetMode(MODE_CLASSIFIER);
}

OCRTraining::~OCRTraining() {
	StopTraining();
}

void OCRTraining::MainMenu(Bar& bar) {
	bar.Add("File", THISBACK(FileMenu));
	bar.Add("Training", THISBACK(TrainingMenu));
}

void OCRTraining::FileMenu(Bar& bar) {
	bar.Add("Load Dataset...", CtrlImg::open(), THISBACK(LoadDatasetProject));
	bar.Add("Export Model...", CtrlImg::save(), THISBACK(ExportModel));
}

void OCRTraining::TrainingMenu(Bar& bar) {
	bar.Add("Start Training", CtrlImg::new_doc(), THISBACK(StartTraining)).Key(K_F5);
	bar.Add("Stop Training", CtrlImg::remove(), THISBACK(StopTraining)).Key(K_F6);
	bar.Separator();
	bar.Add("Reset Network", THISBACK(ResetNetwork));
	bar.Add("Reload Network Architecture", THISBACK(ReloadNetwork));
	bar.Separator();
	bar.Add("Run Batch Test", THISBACK(RunBatchTest)).Key(K_F7);
}

void OCRTraining::DockInit() {
	if (docking_initialized) return;
	RLOG("OCRTraining::DockInit size=" << GetSize());
	
	Add(graph.SizePos());
	
	DockLeft(Dockable(net_ctrl, "Network Editor").SizeHint(Size(300, 400)));
	DockRight(Dockable(control_panel, "Controls").SizeHint(Size(200, 400)));
	DockBottom(Dockable(lctrl, "Layer Visualizer").SizeHint(Size(600, 200)));
	
	DockRight(Dockable(confusion, "Confusion Matrix").SizeHint(Size(300, 300)));
	DockRight(Dockable(samples, "Sample Predictions").SizeHint(Size(300, 300)));
	DockRight(Dockable(splitter_vis, "Splitter Test").SizeHint(Size(300, 100)));
	DockRight(Dockable(dist_ctrl, "Class Distribution").SizeHint(Size(300, 150)));
	
	docking_initialized = true;
	Layout();
	Refresh();
}

void OCRTraining::SetMode(int m) {
	mode = m;
	StopTraining();
	graph.Clear();
	confusion.Clear();
	samples.Clear();
	splitter_vis.Clear();
	
	if (mode == MODE_CLASSIFIER) {
		String arch = R"([
  {"type":"input", "input_width":28, "input_height":28, "input_depth":1},
  {"type":"conv", "filter_count":20, "width":5, "height":5, "pad":0, "activation":"relu"},
  {"type":"pool", "width":2, "height":2, "stride":2},
  {"type":"conv", "filter_count":50, "width":5, "height":5, "pad":0, "activation":"relu"},
  {"type":"pool", "width":2, "height":2, "stride":2},
  {"type":"fc", "neuron_count":500, "activation":"relu"},
  {"type":"softmax", "class_count":62},
  {"type":"sgd", "learning_rate":0.01, "momentum":0.9, "batch_size":32}
])";
		net_edit.SetData(arch);
	} else {
		String arch = R"([
  {"type":"input", "input_width":3, "input_height":28, "input_depth":1},
  {"type":"conv", "filter_count":16, "width":3, "height":3, "pad":1, "activation":"relu"},
  {"type":"pool", "width":2, "height":2, "stride":2},
  {"type":"conv", "filter_count":32, "width":3, "height":3, "pad":1, "activation":"relu"},
  {"type":"pool", "width":2, "height":2, "stride":2},
  {"type":"fc", "neuron_count":64, "activation":"relu"},
  {"type":"softmax", "class_count":2},
  {"type":"sgd", "learning_rate":0.01, "momentum":0.9, "batch_size":64}
])";
		net_edit.SetData(arch);
	}
	
	ReloadNetwork();
}

void OCRTraining::ReloadNetwork() {
	StopTraining();
	String json = net_edit.GetData();
	if (session.MakeLayers(json)) {
		lctrl.UpdateLayers();
		UpdateStatus();
	} else {
		Exclamation("Failed to create network from JSON!");
	}
}

void OCRTraining::LoadProject(const String& path) {
	RLOG("OCRTraining::LoadProject: " << path);
	if (dataset.Load(path)) {
		SyncDatasetToSession();
	} else {
		RLOG("  FAILED to load project in trainer");
	}
}

void OCRTraining::LoadDatasetProject() {
	FileSel fs;
	fs.Type("OCR Dataset (.ocrdata)", "*.ocrdata");
	if (fs.ExecuteOpen("Open Dataset Project")) {
		if (dataset.Load(fs.Get())) {
			SyncDatasetToSession();
		} else {
			Exclamation("Failed to load dataset project!");
		}
	}
}

void OCRTraining::SyncDatasetToSession() {
	if (mode == MODE_CLASSIFIER) SyncClassifier();
	else SyncSplitter();
}

void OCRTraining::SyncClassifier() {
	RLOG("OCRTraining::SyncClassifier starting");
	StopTraining();
	
	int total_chars = 0;
	for (int i = 0; i < dataset.GetCount(); i++) {
		const Vector<OCR::OCRSegment>& segs = dataset.GetSegments(i);
		for(const auto& seg : segs)
			total_chars += seg.chars.GetCount();
	}
	
	RLOG("  Total characters: " << total_chars);
	if (total_chars == 0) {
		RLOG("  WARNING: Dataset is empty!");
		Exclamation("Dataset is empty!");
		return;
	}
	
	// Add original + 2 augmented versions
	int total_samples = total_chars * 3;
	
	SessionData& data = session.Data();
	data.BeginData(62, total_samples, 28 * 28);
	
	int idx = 0;
	for (int i = 0; i < dataset.GetCount(); i++) {
		const Image& img = dataset.GetImage(i);
		const Vector<OCR::OCRSegment>& segs = dataset.GetSegments(i);
		
		for (const auto& seg : segs) {
			for (const auto& ch : seg.chars) {
				Image char_img = OCR::CropImage(img, ch.bounds);
				char_img = OCR::NormalizeChar(char_img, Size(28, 28));
				
				// 1. Original
				for (int y = 0; y < 28; y++) {
					for (int x = 0; x < 28; x++) {
						data.SetData(idx, y * 28 + x, OCR::GetGrayValue(char_img[y][x]) / 255.0);
					}
				}
				data.SetLabel(idx, ch.label);
				idx++;
				
				// 2. Augmented 1
				Image aug1 = OCR::AugmentChar(char_img);
				for (int y = 0; y < 28; y++) {
					for (int x = 0; x < 28; x++) {
						data.SetData(idx, y * 28 + x, OCR::GetGrayValue(aug1[y][x]) / 255.0);
					}
				}
				data.SetLabel(idx, ch.label);
				idx++;

				// 3. Augmented 2
				Image aug2 = OCR::AugmentChar(char_img);
				for (int y = 0; y < 28; y++) {
					for (int x = 0; x < 28; x++) {
						data.SetData(idx, y * 28 + x, OCR::GetGrayValue(aug2[y][x]) / 255.0);
					}
				}
				data.SetLabel(idx, ch.label);
				idx++;
			}
		}
	}
	
	data.EndData();
	
	Vector<int> counts;
	dataset.GetClassDistribution(counts);
	dist_ctrl.SetDistribution(counts);
	
	UpdateStatus();
}

void OCRTraining::SyncSplitter() {
	RLOG("OCRTraining::SyncSplitter starting");
	StopTraining();
	
	OCR::SplitterTrainingData sd = OCR::TextSplitter::GenerateTrainingData(dataset);
	RLOG("  Generated windows: " << sd.windows.GetCount());
	
	if (sd.windows.IsEmpty()) {
		RLOG("  WARNING: No splitting data generated!");
		Exclamation("No splitting data generated!");
		return;
	}
	
	SessionData& data = session.Data();
	data.BeginData(2, sd.windows.GetCount(), 3 * 28);
	
	for (int i = 0; i < sd.windows.GetCount(); i++) {
		const Image& window = sd.windows[i];
		for (int y = 0; y < 28; y++) {
			for (int x = 0; x < 3; x++) {
				data.SetData(i, y * 3 + x, OCR::GetGrayValue(window[y][x]) / 255.0);
			}
		}
		data.SetLabel(i, sd.is_boundary[i] ? 1 : 0);
	}
	
	data.EndData();
	UpdateStatus();
}

void OCRTraining::StartTraining() {
	RLOG("OCRTraining::StartTraining starting");
	if (session.GetLayerCount() < 2) {
		RLOG("  WARNING: Network not initialized!");
		Exclamation("Network not initialized!");
		return;
	}
	if (session.Data().GetDataCount() == 0) {
		RLOG("  WARNING: No training data loaded!");
		Exclamation("No training data loaded!");
		return;
	}
	
	running = true;
	stopped = false;
	session.StartTraining();
	Thread::Start(THISBACK(Refresher));
	RLOG("OCRTraining::StartTraining thread started");
}

void OCRTraining::StopTraining() {
	running = false;
	session.StopTraining();
	while (!stopped) Sleep(10);
}

void OCRTraining::ResetNetwork() {
	ReloadNetwork();
}

void OCRTraining::ExportModel() {
	StopTraining();
	FileSel fs;
	fs.Type("OCR Model (.ocrmodel)", "*.ocrmodel");
	if (fs.ExecuteSaveAs("Export Model")) {
		OCR::OCRModel model;
		model.SetSession(session);
		model.SetArchitecture(net_edit.GetData());
		model.SetType(mode == MODE_CLASSIFIER ? OCR::OCR_MODEL_CLASSIFIER : OCR::OCR_MODEL_SPLITTER);
		model.SetMetadata("accuracy", session.GetTrainingAccuracyAverage());
		model.SetMetadata("loss", session.GetLossAverage());
		model.SetMetadata("iterations", session.GetIteration());
		model.SetMetadata("created_at", Format(GetSysTime()));
		model.SetMetadata("version", "1.0");
		
		if (model.Save(fs.Get())) {
			PromptOK("Model exported successfully!");
		} else {
			Exclamation("Failed to export model!");
		}
	}
}

void OCRTraining::RunBatchTest() {
	StopTraining();
	if (dataset.IsEmpty()) {
		Exclamation("Load a dataset first!");
		return;
	}
	
	OCR::OCRModel model;
	model.SetSession(session);
	model.SetArchitecture(net_edit.GetData());
	model.SetType(mode == MODE_CLASSIFIER ? OCR::OCR_MODEL_CLASSIFIER : OCR::OCR_MODEL_SPLITTER);
	
	OCR::BatchTester tester;
	tester.SetModel(&model);
	tester.SetDataset(&dataset);
	
	OCR::TestResults res = tester.Run();
	
	confusion.SetMatrix(res.confusion_matrix);
	PromptOK(Format("Test Results:\nAccuracy: %.2f%%\nSamples: %d", res.accuracy * 100.0, res.total_samples));
}

void OCRTraining::ViewLayer(int i) {
	// Task 04 will implement detailed layer view
}

void OCRTraining::Refresher() {
	while (running) {
		helper_mutex.Enter();
		if (!helper_running) {
			helper_running = true;
			helper_mutex.Leave();
			Thread::Start(THISBACK(HelperThread));
		} else {
			helper_mutex.Leave();
		}
		Sleep(100);
	}
	stopped = true;
}

void OCRTraining::HelperThread() {
	if (!session.TryEnter()) {
		helper_mutex.Enter();
		helper_running = false;
		helper_mutex.Leave();
		return;
	}
	
	vis_iter = session.GetIteration();
	vis_loss = session.GetLossAverage();
	vis_acc = session.GetTrainingAccuracyAverage();
	
	SessionData& d = session.Data();
	vis_samples = d.GetDataCount();
	int n = d.GetClassCount();
	if (n == 0) n = (mode == MODE_SPLITTER) ? 2 : 62;
	
	vis_m.SetCount(n);
	for(int i = 0; i < n; i++) vis_m[i].SetCount(n, 0);
	vis_sr.Clear();
	
	if (vis_iter % 10 == 0) {
		for (int i = 0; i < min(100, vis_samples); i++) {
			Vector<double> input;
			for(int j = 0; j < d.GetDataLength(); j++)
				input.Add(d.GetData(i, j));
			
			Vector<double> probs = session.Predict0(input);
			int pred = 0;
			if (!probs.IsEmpty()) {
				for(int j = 1; j < probs.GetCount(); j++)
					if(probs[j] > probs[pred]) pred = j;
			}
			
			int actual = d.GetLabel(i);
			if (actual >= 0 && actual < n && pred >= 0 && pred < n)
				vis_m[actual][pred]++;
			
			if (vis_sr.GetCount() < 20) {
				OCR::SamplePredictionsCtrl::SampleResult& s = vis_sr.Add();
				s.true_class = actual;
				s.pred_class = pred;
				s.correct = (actual == pred);
				s.confidence = !probs.IsEmpty() ? probs[pred] : 0.0;
				
				// Reconstruct image for preview
				int dw = d.GetDataWidth();
				int dh = d.GetDataHeight();
				ImageBuffer ib(dw, dh);
				for(int y = 0; y < dh; y++)
					for(int x = 0; x < dw; x++) {
						byte g = (byte)(d.GetData(i, y * dw + x) * 255);
						ib[y][x] = Color(g, g, g);
					}
				s.image = ib;
			}
		}
	}
	session.Leave();

	Ctrl::PostCallback(THISBACK(ApplyVisuals));
}

void OCRTraining::ApplyVisuals() {
	static int last_visual_iter = -1;
	if (vis_iter != last_visual_iter) {
		last_visual_iter = vis_iter;
		
		String s;
		s << "Mode: " << (mode == MODE_CLASSIFIER ? "Classifier" : "Splitter") << "\n";
		s << "Samples: " << vis_samples << "\n";
		s << "Iteration: " << vis_iter << "\n";
		s << "Loss: " << Format("%.4f", vis_loss) << "\n";
		s << "Accuracy: " << Format("%.2f%%", vis_acc * 100.0);
		status_label.SetText(s);
		lctrl.Refresh();
		
		graph.AddDataPoint(vis_loss, vis_acc);
		
		if (vis_iter % 10 == 0) {
			confusion.SetMatrix(vis_m);
			samples.SetSamples(vis_sr);
		}
	}
	
	helper_mutex.Enter();
	helper_running = false;
	helper_mutex.Leave();
}

void OCRTraining::UpdateStatus() {
	String s;
	s << "Mode: " << (mode == MODE_CLASSIFIER ? "Classifier" : "Splitter") << "\n";
	s << "Samples: " << session.Data().GetDataCount() << "\n";
	s << "Iteration: " << session.GetIteration() << "\n";
	s << "Loss: " << Format("%.4f", session.GetLossAverage()) << "\n";
	s << "Accuracy: " << Format("%.2f%%", session.GetTrainingAccuracyAverage() * 100.0);
	status_label.SetText(s);
	lctrl.Refresh();
}

void OCRTraining::SetEmbedded(bool b) {
	if (b) {
		// OCRTraining doesn't have its own menu/toolbar variables, 
		// it might be using frames added to itself.
	}
}

} // namespace Upp
