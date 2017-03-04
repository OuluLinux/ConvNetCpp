#include "ClassifyImages.h"

#define IMAGECLASS ClassifyImagesImg
#define IMAGEFILE <ClassifyImages/ClassifyImages.iml>
#include <Draw/iml_source.h>



ClassifyImages::ClassifyImages()
{
	Icon(ClassifyImagesImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	
	#if defined flagMNIST
	Title("Classify MNIST digits");
	t =		"[\n"
			"\t{\"type\":\"input\", \"input_width\":24, \"input_height\":24, \"input_depth\":1},\n"
			"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":8, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
			"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
			"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":16, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
			"\t{\"type\":\"pool\", \"width\":3, \"height\":3, \"stride\":3},\n"
			"\t{\"type\":\"softmax\", \"class_count\":10},\n"
			"\t{\"type\":\"adadelta\", \"batch_size\":20, \"l2_decay\":0.001}\n"
			"]\n";
	img_sz = Size(28,28);
	augmentation = 24;
	do_flip = false;
	has_colors = false;
	#elif defined flagCIFAR10
	Title("Classify CIFAR-10 images");
	t =		"[\n"
			"\t{\"type\":\"input\", \"input_width\":30, \"input_height\":30, \"input_depth\":3},\n"
			"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":16, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
			"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
			"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":20, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
			"\t{\"type\":\"pool\", \"width\":3, \"height\":2, \"stride\":2},\n"
			"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":20, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
			"\t{\"type\":\"pool\", \"width\":3, \"height\":2, \"stride\":2},\n"
			"\t{\"type\":\"softmax\", \"class_count\":10},\n"
			"\t{\"type\":\"adadelta\", \"batch_size\":20, \"l2_decay\":0.001}\n"
			"]\n";
	img_sz = Size(32,32);
	augmentation = 30;
	do_flip = true;
	has_colors = true;
	layer_view.SetColor();
	#endif
	
	net_edit.SetData(t);
	
	layer_view.SetSession(ses);
	
	running = false;
	stopped = true;
	ticking_running = false;
	ticking_stopped = true;
	use_validation_data = true;
	
	loss_graph.SetMode(PLOT_AA).SetLimits(-5,5,-5,5);
	loss_graph.data.Add();
	loss_graph.data.Add();
	loss_graph.data[0].SetTitle("Loss").SetThickness(1.5).SetColor(Red());
	loss_graph.data[1].SetDash("1.5").SetTitle("Average").SetThickness(1.0).SetColor(Blue());
	
	average_size = 10;
	forward_time = 0.0;
	backward_time = 0.0;
	max_diff_imgs = 100000; // not limiting currently
	
	UpdateNetParamDisplay();
	
	xLossWindow.Init(100);
	wLossWindow.Init(100);
	trainAccWindow.Init(100);
	valAccWindow.Init(100);
	step_num = 0;
	
	Add(v_split.SizePos());
	v_split.Vert();
	v_split << layer_view << pred_view;
	v_split.SetPos(6400);
	
	
	
	net_ctrl.Add(net_edit.HSizePos().VSizePos(0,30));
	net_ctrl.Add(reload_btn.HSizePos().BottomPos(0,30));
	reload_btn.SetLabel("Reload Network");
	reload_btn <<= THISBACK(Reload);
	
	lrate.SetLabel("Learning rate:");
	lmom.SetLabel("Momentum:");
	lbatch.SetLabel("Batch size:");
	ldecay.SetLabel("Weight decay:");
	apply.SetLabel("Apply");
	save_net.SetLabel("Save network");
	load_net.SetLabel("Load network");
	apply <<= THISBACK(ApplySettings);
	save_net <<= THISBACK(SaveFile);
	load_net <<= THISBACK(OpenFile);
	int row = 20;
	settings.Add(lrate.HSizePos(4,4).TopPos(0,row));
	settings.Add(rate.HSizePos(4,4).TopPos(1*row,row));
	settings.Add(lmom.HSizePos(4,4).TopPos(2*row,row));
	settings.Add(mom.HSizePos(4,4).TopPos(3*row,row));
	settings.Add(lbatch.HSizePos(4,4).TopPos(4*row,row));
	settings.Add(batch.HSizePos(4,4).TopPos(5*row,row));
	settings.Add(ldecay.HSizePos(4,4).TopPos(6*row,row));
	settings.Add(decay.HSizePos(4,4).TopPos(7*row,row));
	settings.Add(apply.HSizePos(4,4).TopPos(8*row,row));
	settings.Add(save_net.HSizePos(4,4).TopPos(9*row,row));
	settings.Add(load_net.HSizePos(4,4).TopPos(10*row,row));
	rate.SetData(0.01);
	mom.SetData(0.9);
	batch.SetData(20);
	decay.SetData(0.001);
	
	Reload();
    Start();
}

ClassifyImages::~ClassifyImages() {
	ticking_running = false;
	while (!ticking_stopped) Sleep(100);
}

void ClassifyImages::DockInit() {
	DockLeft(Dockable(settings, "Settings").SizeHint(Size(320, 11*20)));
	DockLeft(Dockable(loss_graph, "Loss").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void ClassifyImages::UpdateNetParamDisplay() {
	TrainerBase* t = ses.GetTrainer();
	if (!t) return;
	TrainerBase& trainer = *t;
	rate.SetData(trainer.GetLearningRate());
	mom.SetData(trainer.GetMomentum());
	batch.SetData(trainer.GetBatchSize());
	decay.SetData(trainer.GetL2Decay());
}

void ClassifyImages::ApplySettings() {
	TrainerBase* t = ses.GetTrainer();
	if (!t) return;
	TrainerBase& trainer = *t;
	trainer.SetLearningRate(rate.GetData());
	trainer.SetMomentum(mom.GetData());
	trainer.SetBatchSize(batch.GetData());
	trainer.SetL2Decay(decay.GetData());
}

void ClassifyImages::OpenFile() {
	String file = SelectFileOpen("JSON files\t*.json\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	if (!FileExists(file)) {
		PromptOK("File does not exists");
		return;
	}
	
	// Load json
	String json = LoadFile(file);
	if (json.IsEmpty()) {
		PromptOK("File is empty");
		return;
	}
	
	StopTicking();
	
	ticking_lock.Enter();
	bool res = ses.LoadOriginalJSON(json);
	if (!res) {
		ticking_running = false;
	}
	ticking_lock.Leave();
	
	if (!res) {
		PromptOK("Loading failed.");
		return;
	}
	
	ResetAll();
	Start();
}

void ClassifyImages::SaveFile() {
	String file = SelectFileSaveAs("JSON files\t*.json\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	// Save json
	String json;
	if (!ses.StoreOriginalJSON(json)) {
		PromptOK("Error: Getting JSON failed");
		return;
	}
	
	FileOut fout(file);
	if (!fout.IsOpen()) {
		PromptOK("Error: could not open file " + file);
		return;
	}
	fout << json;
}

void ClassifyImages::Reload() {
	StopTicking();
	
	String net_str = net_edit.GetData();
	
	ticking_lock.Enter();
	
	bool success = ses.MakeLayers(net_str);
	
	ticking_lock.Leave();
	
	ResetAll();
	layer_view.Layout();
	
	if (success) Start();
}

void ClassifyImages::Tick() {
	SampleTrainingInstance(tmp_sample);
	Step(tmp_sample);
}

void ClassifyImages::Ticking() {
	while (ticking_running) {
		if (!ses.GetTrainer()) {Sleep(100); continue;}
		ticking_lock.Enter();
		Tick();
		ticking_lock.Leave();
	}
	ticking_stopped = true;
}

void ClassifyImages::Start() {
	if (!running) {
		running = true;
		stopped = false;
		PostCallback(THISBACK(Refresher));
	}
	
	if (!ticking_running) {
		ticking_running = true;
		ticking_stopped = false;
		Thread::Start(THISBACK(Ticking));
	}
}

void ClassifyImages::RefreshStatus() {
	String s;
	s << "   Forward time per example: " << forward_time << "\n";
	s << "   Backprop time per example: " << backward_time << "\n";
	s << "   Classification loss: " << xLossWindow.GetAverage() << "\n";
	s << "   L2 Weight decay loss: " << wLossWindow.GetAverage() << "\n";
	s << "   Training accuracy: " << trainAccWindow.GetAverage() << "\n";
	s << "   Validation accuracy: " << valAccWindow.GetAverage() << "\n";
	s << "   Examples seen: " << step_num;
	status.SetLabel(s);
}

void ClassifyImages::Refresher() {
	layer_view.Refresh();
	loss_graph.Sync();
	loss_graph.Refresh();
	RefreshStatus();

	if (running) PostCallback(THISBACK(Refresher));
	else stopped = true;
}

void ClassifyImages::AddLoss() {
	int id = loss_graph.data[0].GetCount();
	
	double xa = xLossWindow.GetAverage();
	double xw = wLossWindow.GetAverage();
	if (xa < 0 || xw < 0) return;
	double av = xa + xw;
	
	loss_graph.data[0].AddXY(id, av);
	int count = id + 1;
	if (count < 2) return;
	
	int pos = id;
	double sum = 0;
	int av_count = 0;
	for(int i = 0; i < average_size && pos >= 0; i++) {
		sum += loss_graph.data[0][pos].y;
		av_count++;
		pos--;
	}
	double avav = sum / av_count;
	loss_graph.data[1].AddXY(id, avav);
	
	
	double min = +DBL_MAX;
	double max = -DBL_MAX;
	for(int i = 0; i < count; i++) {
		double d = loss_graph.data[0][i].y;
		if (d > max) max = d;
		if (d < min) min = d;
	}
	double diff = max - min;
	if (diff <= 0) return;
	double center = min + diff / 2;
	loss_graph.SetLimits(0, id, min, max);
	loss_graph.SetModify();
}

void ClassifyImages::SampleTrainingInstance(Sample& sample) {
	ImageBank& bank = GetImageBank();
	ASSERT(bank.images.GetCount() == bank.labels.GetCount());
	
	// find an unloaded batch
	int n = Random(min(max_diff_imgs, bank.images.GetCount()));
	
	sample.label = bank.labels[n];
	sample.isval = use_validation_data && (((n % 10) == 0) ? true : false);
	
	// fetch the appropriate row of the training image and reshape into a Vol
	Image& p = bank.images[n];
	int length = img_sz.cx * img_sz.cy;
	const RGBA* it = p.Begin();
	if (!has_colors) {
		sample.x.Init(img_sz.cx, img_sz.cy, 1, 0.0);
		for (int i = 0; i < length; i++, it++) {
			sample.x.Set(i, (it->r) /255.0);
		}
	} else {
		sample.x.Init(img_sz.cx, img_sz.cy, 3, 0.0);
		for (int y = 0; y < img_sz.cy; y++) {
			for (int x = 0; x < img_sz.cx; x++) {
				sample.x.Set(x, y, 0, (it->r) /255.0);
				sample.x.Set(x, y, 1, (it->g) /255.0);
				sample.x.Set(x, y, 2, (it->b) /255.0);
				it++;
			}
		}
	}
	sample.x.Augment(augmentation, -1, -1, do_flip);
	
}

// sample a random testing instance
void ClassifyImages::SampleTestInstance(Vector<Sample>& samples) {
	ImageBank& bank = GetImageBank();
	ASSERT(bank.test_images.GetCount() == bank.test_labels.GetCount());
	
	int n = Random(min(max_diff_imgs, bank.test_images.GetCount()));
	
	Image& p = bank.test_images[n];
	int label = bank.test_labels[n];
	
	int length = img_sz.cx * img_sz.cy;
	const RGBA* it = p.Begin();
	if (!has_colors) {
		tmp_vol.Init(img_sz.cx, img_sz.cy, 1, 0.0);
		for (int i = 0; i < length; i++, it++) {
			tmp_vol.Set(i, (it->r) /255.0);
		}
	} else {
		tmp_vol.Init(img_sz.cx, img_sz.cy, 3, 0.0);
		for (int y = 0; y < img_sz.cy; y++) {
			for (int x = 0; x < img_sz.cx; x++) {
				tmp_vol.Set(x, y, 0, (it->r) /255.0);
				tmp_vol.Set(x, y, 1, (it->g) /255.0);
				tmp_vol.Set(x, y, 2, (it->b) /255.0);
				it++;
			}
		}
	}
	
	int aug_sample_count = 4;
	
	samples.SetCount(aug_sample_count);
	for (int i = 0; i < aug_sample_count; i++) {
		Sample& s = samples[i];
		s.x = tmp_vol;
		s.x.Augment(augmentation, -1, -1, do_flip);
		s.label = label;
		s.img_id = n;
	}
	
	// return multiple augmentations, and we will average the network over them
	// to increase performance
}



void ClassifyImages::TestPredict() {
	ImageBank& bank = GetImageBank();
	
	Net& net = ses.GetNetwork();
	int layer_count = net.GetLayers().GetCount();
	ASSERT(layer_count);
	
	int num_classes = net.GetLayers()[layer_count-1]->output_depth;
	
	//document.getElementById('testset_acc').innerHTML = '';
	// grab a random test image
	for (int num = 0; num < 50; num++) {
		SampleTestInstance(tmp_samples);
		int y = tmp_samples[0].label;  // ground truth label
		
		// forward prop it through the network
		aavg.Init(1,1,num_classes,0.0);
		
		// ensures we always have a list, regardless if above returns single item or list
		int n = tmp_samples.GetCount();
		for (int i = 0; i < n; i++) {
			Volume& a = net.Forward(tmp_samples[i].x);
			aavg.AddFrom(a);
		}
		
		double y_val = aavg.Get(y);
		VectorMap<int, double> preds;
		for(int i = 0; i < num_classes; i++) {
			if (i == y) continue;
			preds.Add(i, aavg.Get(i) / n);
		}
		SortByValue(preds, StdGreater<double>());
		
		int img_id = tmp_samples[0].img_id;
		Image& img = bank.test_images[img_id];
		pred_view.Add(img, bank.classes[y], y_val, bank.classes[preds.GetKey(0)], preds[0], bank.classes[preds.GetKey(1)], preds[1]);
		
	}
	
	PostCallback(THISBACK(RefreshPredictions));
}

void ClassifyImages::Step(Sample& sample) {
	Volume& x = sample.x;
	int y = sample.label;
	Net& net = ses.GetNetwork();
	
	
	if (sample.isval) {
		// use x to build our estimate of validation error
		TimeStop ts;
		net.Forward(x);
		forward_time = ts.Elapsed();
		
		int yhat = net.GetPrediction();
		double val_acc = yhat == y ? 1.0 : 0.0;
		valAccWindow.Add(val_acc);
		return; // get out
	}
	
	// train on it with network
	TrainerBase& trainer = *ses.GetTrainer();
	TimeStop ts;
	trainer.Train(x, y, 1);
	backward_time = ts.Elapsed();
	double lossx = trainer.GetLoss();
	double lossw = trainer.GetL2DecayLoss();
	
	// keep track of stats such as the average training error and loss
	double yhat = net.GetPrediction();
	double train_acc = yhat == y ? 1.0 : 0.0;
	xLossWindow.Add(lossx);
	wLossWindow.Add(lossw);
	trainAccWindow.Add(train_acc);
	
	step_num++;
	
	// log progress to graph, (full loss)
	if (step_num % 200 == 0) {
		AddLoss();
	}
	
	// run prediction on test set
	if (step_num == 100 || (step_num % 1000) == 0) {
		TestPredict();
	}
}

void ClassifyImages::ResetAll() {
	UpdateNetParamDisplay();
	
	// reinit windows that keep track of val/train accuracies
	xLossWindow.Clear();
	wLossWindow.Clear();
	trainAccWindow.Clear();
	valAccWindow.Clear();
	loss_graph.data[0].Clear();
	loss_graph.data[1].Clear();
	step_num = 0;

}


