#include "ClassifyImages.h"

#define IMAGECLASS ClassifyImagesImg
#define IMAGEFILE <ClassifyImages/ClassifyImages.iml>
#include <Draw/iml_source.h>


ClassifyImages::ClassifyImages(int loader, int type)
{
	this->loader = loader;
	this->type = type;
	
	Icon(ClassifyImagesImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	
	if (loader == LOADER_MNIST) {
		if (type == TYPE_LEARNER) {
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
			augmentation = 24;
		}
		else if (type == TYPE_AUTOENCODER) {
			Title("MNIST digits autoencoder");
			t =		"[\n"
					"\t{\"type\":\"input\", \"input_width\":28, \"input_height\":28, \"input_depth\":1},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":2},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"regression\", \"neuron_count\":784},\n" // 24*24=576, 28*28=784
					"\t{\"type\":\"adadelta\", \"learning_rate\":1, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
			augmentation = 0;
		}
		else if (type == TYPE_CONV) {
			Title("MNIST convolutive autoencoder");
			/*t =		"[\n"
					"\t{\"type\":\"input\", \"input_width\":28, \"input_height\":28, \"input_depth\":1},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":1, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":1, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"unpool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"deconv\", \"width\":5, \"height\":5, \"filter_count\":1, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"unpool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"deconv\", \"width\":5, \"height\":5, \"filter_count\":1, \"stride\":1, \"pad\":2},\n"
					"\t{\"type\":\"regression\", \"neuron_count\":784},\n"
					"\t{\"type\":\"adadelta\", \"learning_rate\":1, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";*/
			t =		"[\n"
					"\t{\"type\":\"input\", \"input_width\":28, \"input_height\":28, \"input_depth\":1},\n"
					"\t{\"type\":\"conv\", \"width\":8, \"height\":8, \"filter_count\":1, \"stride\":2, \"pad\":5, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"conv\", \"width\":8, \"height\":8, \"filter_count\":1, \"stride\":2, \"pad\":5, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"deconv\", \"width\":8, \"height\":8, \"filter_count\":1, \"stride\":2, \"pad\":5, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"deconv\", \"width\":8, \"height\":8, \"filter_count\":1, \"stride\":2, \"pad\":5},\n"
					"\t{\"type\":\"regression\", \"neuron_count\":784},\n"
					"\t{\"type\":\"adadelta\", \"learning_rate\":1, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
			augmentation = 0;
		}
		else Panic("Invalid type");
		
		img_sz = Size(28,28);
		do_flip = false;
		has_colors = false;
	}
	
	else if (loader == LOADER_CIFAR10) {
		if (type == TYPE_LEARNER) {
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
			augmentation = 30;
			do_flip = true;
		}
		else if (type == TYPE_AUTOENCODER) {
			Title("CIFAR-10 autoencoder");
			t =		"[\n"
					"\t{\"type\":\"input\", \"input_width\":32, \"input_height\":32, \"input_depth\":3},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":2},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"regression\", \"neuron_count\":3072},\n" // 3*32*32=3072
					"\t{\"type\":\"adadelta\", \"learning_rate\":1, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
			augmentation = 0;
			do_flip = false;
		}
		else if (type == TYPE_CONV) {
			Title("CIFAR-10 convolutive autoencoder");
			/*t =		"[\n"
					"\t{\"type\":\"input\", \"input_width\":32, \"input_height\":32, \"input_depth\":3},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":3, \"stride\":1, \"pad\":2, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":3, \"stride\":1, \"pad\":2, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"unpool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"deconv\", \"width\":5, \"height\":5, \"filter_count\":3, \"stride\":1, \"pad\":2, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"unpool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"deconv\", \"width\":5, \"height\":5, \"filter_count\":3, \"stride\":1, \"pad\":2},\n"
					"\t{\"type\":\"regression\", \"neuron_count\":3072},\n" // 3*32*32=3072
					"\t{\"type\":\"adadelta\", \"learning_rate\":1, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";*/
			t =		"[\n"
					"\t{\"type\":\"input\", \"input_width\":32, \"input_height\":32, \"input_depth\":3},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":3, \"stride\":2, \"pad\":2, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":3, \"stride\":2, \"pad\":2, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"deconv\", \"width\":5, \"height\":5, \"filter_count\":3, \"stride\":2, \"pad\":2, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"deconv\", \"width\":8, \"height\":8, \"filter_count\":3, \"stride\":2, \"pad\":2},\n"
					"\t{\"type\":\"regression\", \"neuron_count\":3072},\n" // 3*32*32=3072
					"\t{\"type\":\"adadelta\", \"learning_rate\":1, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
			augmentation = 0;
			do_flip = false;
		}
		else Panic("Invalid type");
		
		img_sz = Size(32,32);
		has_colors = true;
		layer_view.SetColor();
	}
	else {
		Panic("Invalid loader");
	}
	
	net_edit.SetData(t);
	
	average_size = 10;
	max_diff_imgs = 100000; // not limiting currently
	
	UpdateNetParamDisplay();
	
	dispatcher.RegisterStandardCommands();
	viewport.SetGraph(graph_node);
	viewport.SetEditor(editor);
	viewport.SetHistory(history);
	viewport.SetDispatcher(dispatcher);

	Add(tabs.SizePos());
	tabs.Add(v_split.SizePos(), "Network");
	tabs.Add(viewport.SizePos(), "Graph Editor");

	v_split.Vert();
	
	if (type == TYPE_LEARNER) {
		v_split << layer_view << pred_view;
		v_split.SetPos(6400);

		ses.SetTestPredict(true);
		
		pred_view.SetSession(ses);
		pred_view.SetAugmentation(augmentation, do_flip);
	}
	else if (type == TYPE_AUTOENCODER) {
		v_split << aenc_view << layer_view;
		v_split.SetPos(6400);
		
		aenc_view.SetSession(ses);
		layer_view.HideGradients();
	}
	else {
		v_split << layer_view;
	}
	
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
	pause.SetLabel("Pause");
	apply <<= THISBACK(ApplySettings);
	save_net <<= THISBACK(SaveFile);
	load_net <<= THISBACK(OpenFile);
	pause <<= THISBACK(Pause);
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
	settings.Add(pause.HSizePos(4,4).TopPos(11*row,row));
	rate.SetData(0.01);
	mom.SetData(0.9);
	batch.SetData(20);
	decay.SetData(0.001);
	
	
	ses.SetAugmentation(augmentation, do_flip);
	
	layer_view.SetSession(ses);
	
	loss_graph.SetSession(ses);
	loss_graph.SetModeLoss();
	
	SyncGraph();
	PostCallback(THISBACK(Refresher));
}

ClassifyImages::~ClassifyImages() {
	
	ses.StopTraining();
	
}

void ClassifyImages::DockInit() {
	DockLeft(Dockable(settings, "Settings").SizeHint(Size(320, 11*20)));
	DockLeft(Dockable(loss_graph, "Loss").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void ClassifyImages::UpdateNetParamDisplay() {
	TrainerBase& trainer = ses.GetTrainer();
	rate.SetData(trainer.GetLearningRate());
	mom.SetData(trainer.GetMomentum());
	batch.SetData(trainer.GetBatchSize());
	decay.SetData(trainer.GetL2Decay());
}

void ClassifyImages::ApplySettings() {
	TrainerBase& trainer = ses.GetTrainer();
	trainer.SetLearningRate(rate.GetData());
	trainer.SetMomentum(mom.GetData());
	trainer.SetBatchSize(batch.GetData());
	trainer.SetL2Decay(decay.GetData());
}

void ClassifyImages::Pause() {
	if (ses.IsTraining())
		ses.StopTraining();
	else
		ses.StartTraining();
}

void ClassifyImages::OpenFile() {
	String file = SelectFileOpen("BIN files\t*.bin\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	if (!FileExists(file)) {
		PromptOK("File does not exists");
		return;
	}
	
	ses.StopTraining();
	
	ticking_lock.Enter();
	FileIn fin(file);
	fin % ses;
	ticking_lock.Leave();
	
	ResetAll();
	ses.StartTraining();
	SyncGraph();
}

void ClassifyImages::SaveFile() {
	String file = SelectFileSaveAs("BIN files\t*.bin\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	FileOut fout(file);
	if (!fout.IsOpen()) {
		PromptOK("Error: could not open file " + file);
		return;
	}
	
	ses.StopTraining();
	
	fout % ses;
	
	ses.StartTraining();
}

void ClassifyImages::Reload() {
	ses.StopTraining();
	
	String net_str = net_edit.GetData();
	
	ticking_lock.Enter();
	
	bool success = ses.MakeLayers(net_str);
	
	ticking_lock.Leave();
	
	ResetAll();
	layer_view.Layout();
	
	if (success) {
		ses.StartTraining();
		SyncGraph();
	}
}

void ClassifyImages::RefreshStatus() {
	String s;
	if (type == TYPE_LEARNER) {
		s << "   Forward time per example: " << ses.GetForwardTime() << "\n";
		s << "   Backprop time per example: " << ses.GetBackwardTime() << "\n";
		s << "   Classification loss: " << ses.GetLossAverage() << "\n";
		s << "   L2 Weight decay loss: " << ses.GetL2DecayLossAverage() << "\n";
		s << "   Training accuracy: " << ses.GetTrainingAccuracyAverage() << "\n";
		s << "   Validation accuracy: " << ses.GetValidationAccuracyAverage() << "\n";
		s << "   Examples seen: " << ses.GetStepCount();
	}
	else if (type == TYPE_AUTOENCODER || type == TYPE_CONV) {
		s << "   Forward time per example: " << ses.GetForwardTime() << "\n";
		s << "   Backprop time per example: " << ses.GetBackwardTime() << "\n";
		s << "   Regression loss: " << ses.GetLossAverage() << "\n";
		s << "   L2 Weight decay loss: " << ses.GetL2DecayLossAverage() << "\n";
		s << "   L1 Weight decay loss: " << ses.GetL1DecayLossAverage() << "\n";
		s << "   Examples seen: " << ses.GetStepCount();
	}
	status.SetLabel(s);
}

void ClassifyImages::Refresher() {
	layer_view.Refresh();
	 
	if (type == TYPE_AUTOENCODER)
		aenc_view.Refresh();
		
	loss_graph.RefreshData();
	RefreshStatus();
	
	PostCallback(THISBACK(Refresher));
}

void ClassifyImages::ResetAll() {
	UpdateNetParamDisplay();
	loss_graph.Clear();
}




void ClassifyImages::SyncGraph() {
	ses.Enter();
	graph_node.Clear();
	
	Net& net = ses.GetNetwork();
	const Array<LayerBase>& layers = net.GetLayers();
	
	Vector<String> node_ids;
	
	for(int i = 0; i < layers.GetCount(); i++) {
		const LayerBase& l = layers[i];
		String id = "layer_" + IntStr(i);
		
		Upp::Node::NodeDoc& n = graph_node.AddNode(id);
		node_ids.Add(id);
		
		n.label = l.GetKey();
		if (n.label.IsEmpty()) {
			n.label = "Layer " + IntStr(i);
		}
		
		// Add input slot (except for first layer)
		if (i > 0) {
			n.slots.Add().id = "in";
		}
		
		// Add output pin (except for last layer)
		if (i < layers.GetCount() - 1) {
			n.pins.Add().id = "out";
		}
		
		// Add some metadata/labels based on layer type
		if (l.IsInputLayer()) {
			if (l.input_width > 0 && l.input_height > 0 && l.input_depth > 0)
				n.label << "\n(" << l.input_width << "x" << l.input_height << "x" << l.input_depth << ")";
		}
		else if (l.layer_type == CONV_LAYER || l.layer_type == DECONV_LAYER) {
			if (l.width > 0 && l.height > 0 && l.filter_count > 0)
				n.label << "\n(" << l.filter_count << " filters, " << l.width << "x" << l.height << ")";
		}
		else if (l.layer_type == FULLYCONN_LAYER) {
			if (l.neuron_count > 0)
				n.label << "\n(" << l.neuron_count << " neurons)";
		}
	}
	
	// Connect layers
	for(int i = 0; i < (int)layers.GetCount() - 1; i++) {
		graph_node.AddEdge("edge_" + IntStr(i), node_ids[i], "out", node_ids[i+1], "in");
	}
	
	// Apply layout
	Vector<Upp::Node::NodeState> states;
	Upp::Node::SpringLayout spring;
	spring.Iterations(100).Seed(42);
	spring.Run(graph_node, states);
	graph_node.ApplyLayout(states);
	
	viewport.SetGraph(graph_node);
	viewport.ZoomToFit();
	viewport.Update();
	ses.Leave();
}

bool ClassifyImages::IsViewportWhite() {
	Size sz = viewport.GetSize();
	if(sz.cx <= 0 || sz.cy <= 0) sz = Size(1024, 768);
	viewport.SetRect(sz);
	ImageDraw id(sz);
	viewport.Paint(id);
	Image img = id;
	
	for(int y = 0; y < img.GetHeight(); y++) {
		const RGBA* p = img[y];
		for(int x = 0; x < img.GetWidth(); x++) {
			if(p[x].r != 255 || p[x].g != 255 || p[x].b != 255)
				return false;
		}
	}
	return true;
}

void ClassifyImages::SaveViewportImage(const String& path) {
	Size sz = viewport.GetSize();
	if(sz.cx <= 0 || sz.cy <= 0) sz = Size(1024, 768);
	viewport.SetRect(sz);
	ImageDraw id(sz);
	viewport.Paint(id);
	PNGEncoder().SaveFile(path, id);
}
