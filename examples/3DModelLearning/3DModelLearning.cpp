#include "3DModelLearning.h"

Model3DApp::Model3DApp()
{
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("3D Model Learning and Generation");

    CtrlLayout(*this);

	// Define a default 3D model architecture (Voxel-based AE as starting point)
	t = BuildVoxelAEConfig();

	net_edit.SetData(t);

	average_size = 10;
	max_diff_imgs = 100000; // not limiting currently

	UpdateNetParamDisplay();

	Add(v_split.SizePos());
	v_split.Vert();

	v_split << layer_view;
	v_split.SetPos(6400);

	ses.SetTestPredict(true);

	layer_view.SetSession(ses);
	layer_view.SetColor();

	graph.SetSession(ses);
	graph.SetModeLoss();

	// Setup 3D-specific UI controls manually
	m_ctrl.Add(model_display.SizePos());

	// Setup model type selector
	model_type.Add(0, "Voxel Autoencoder");
	model_type.Add(1, "PointNet");
	model_type.Add(2, "Occupancy Network");
	model_type.Add(3, "DeepSDF");
	model_type.SetIndex(0);
	model_type <<= THISBACK(OnModelTypeChanged);

	// Setup buttons
	generate_btn.SetLabel("Generate 3D Model");
	train_vae_btn.SetLabel("Train VAE");
	train_gan_btn.SetLabel("Train GAN");
	load_model_btn.SetLabel("Load 3D Model");
	save_model_btn.SetLabel("Save 3D Model");
	render_3d_btn.SetLabel("Render 3D");

	generate_btn <<= THISBACK(OnGenerate);
	train_vae_btn <<= THISBACK(OnTrainVAE);
	train_gan_btn <<= THISBACK(OnTrainGAN);
	load_model_btn <<= THISBACK(OnLoad3DModel);
	save_model_btn <<= THISBACK(OnSave3DModel);
	render_3d_btn <<= THISBACK(OnRender3D);

	// Add controls manually
	m_ctrl.Add(model_type.HSizePos().TopPos(0, 24));
	m_ctrl.Add(generate_btn.HSizePos().TopPos(24, 24));
	m_ctrl.Add(train_vae_btn.HSizePos().TopPos(48, 24));
	m_ctrl.Add(train_gan_btn.HSizePos().TopPos(72, 24));
	m_ctrl.Add(load_model_btn.HSizePos().TopPos(96, 24));
	m_ctrl.Add(save_model_btn.HSizePos().TopPos(120, 24));
	m_ctrl.Add(render_3d_btn.HSizePos().TopPos(144, 24));
	m_ctrl.Add(model_display.HSizePos().VSizePos(168, 0));

	PostCallback(THISBACK(Refresher));
}

Model3DApp::~Model3DApp() {
	ses.StopTraining();
}

void Model3DApp::DockInit() {
	DockLeft(Dockable(settings, "Settings").SizeHint(Size(320, 11*20)));
	DockLeft(Dockable(graph, "Loss").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void Model3DApp::UpdateNetParamDisplay() {
	TrainerBase& trainer = ses.GetTrainer();
	rate.SetData(trainer.GetLearningRate());
	mom.SetData(trainer.GetMomentum());
	batch.SetData(trainer.GetBatchSize());
	decay.SetData(trainer.GetL2Decay());
}

void Model3DApp::ApplySettings() {
	TrainerBase& trainer = ses.GetTrainer();
	trainer.SetLearningRate(rate.GetData());
	trainer.SetMomentum(mom.GetData());
	trainer.SetBatchSize(batch.GetData());
	trainer.SetL2Decay(decay.GetData());
}

void Model3DApp::Pause() {
	if (ses.IsTraining())
		ses.StopTraining();
	else
		ses.StartTraining();
}

void Model3DApp::OpenFile() {
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
}

void Model3DApp::SaveFile() {
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

void Model3DApp::Reload() {
	ses.StopTraining();

	String net_str = net_edit.GetData().ToString();

	ticking_lock.Enter();

	bool success = ses.MakeLayers(net_str);

	ticking_lock.Leave();

	ResetAll();
	layer_view.Layout();

	if (success) {
		ses.StartTraining();
	}
}

void Model3DApp::RefreshStatus() {
	String s;
	s << "   Forward time per example: " << ses.GetForwardTime() << "\n";
	s << "   Backprop time per example: " << ses.GetBackwardTime() << "\n";
	s << "   Reconstruction loss: " << ses.GetLossAverage() << "\n";
	s << "   L2 Weight decay loss: " << ses.GetL2DecayLossAverage() << "\n";
	s << "   Examples seen: " << ses.GetStepCount();
	status.SetLabel(s);
}

void Model3DApp::Refresher() {
	layer_view.Refresh();
	graph.RefreshData();
	RefreshStatus();
	model_display.Refresh();

	PostCallback(THISBACK(Refresher));
}

void Model3DApp::ResetAll() {
	UpdateNetParamDisplay();
	graph.Clear();
}

void Model3DApp::OnGenerate() {
	Vector<Point> pts;
	for(int i = 0; i < 100; i++) {
		pts.Add(Point(50 + Random(200), 50 + Random(200)));
	}
	model_display.SetPointCloud(pts);
	model_display.Refresh();
	
	PromptOK("Generated 3D model (simulated)");
}

void Model3DApp::OnTrainVAE() {
	PromptOK("Training 3D VAE would start here");
}

void Model3DApp::OnTrainGAN() {
	PromptOK("Training 3D GAN would start here");
}

void Model3DApp::OnLoad3DModel() {
	String file = SelectFileOpen("3D Model files\t*.obj;*.ply;*.stl\nAll files\t*.*");
	if (file.IsEmpty()) return;

	if (!FileExists(file)) {
		PromptOK("File does not exist");
		return;
	}

	LOG("3D model loaded from: " + file);
	PromptOK("3D model loaded (simulated)");
}

void Model3DApp::OnSave3DModel() {
	String file = SelectFileSaveAs("3D Model files\t*.obj;*.ply;*.stl\nAll files\t*.*");
	if (file.IsEmpty()) return;

	LOG("3D model saved to: " + file);
	PromptOK("3D model saved (simulated)");
}

void Model3DApp::OnRender3D() {
	model_display.Refresh();
	LOG("3D model rendered");
}

void Model3DApp::OnModelTypeChanged() {
	int idx = model_type.GetIndex();

	switch(idx) {
		case 0: t = BuildVoxelAEConfig(); break;
		case 1: t = BuildPointNetConfig(); break;
		case 2: t = BuildOccupancyNetConfig(); break;
		case 3: t = BuildDeepSDFConfig(); break;
		default: t = BuildVoxelAEConfig(); break;
	}

	net_edit.SetData(t);
	Reload();
}

String Model3DApp::BuildVoxelAEConfig() {
	return 
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":32, \"input_height\":32, \"input_depth\":32},\n"
		"  { \"type\" : \"conv3d\", \"filter_width\":4, \"filter_height\":4, \"filter_depth\":4, \"filter_count\":32, \"stride\":2, \"padding\":1, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"conv3d\", \"filter_width\":4, \"filter_height\":4, \"filter_depth\":4, \"filter_count\":64, \"stride\":2, \"padding\":1, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"conv3d\", \"filter_width\":4, \"filter_height\":4, \"filter_depth\":4, \"filter_count\":128, \"stride\":2, \"padding\":1, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"conv3d\", \"filter_width\":4, \"filter_height\":4, \"filter_depth\":4, \"filter_count\":256, \"stride\":2, \"padding\":1, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"flatten\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":256, \"activation\":\"linear\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":2048, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"reshape\", \"dim1\":8, \"dim2\":8, \"dim3\":32},\n"
		"  { \"type\" : \"conv3d_t\", \"filter_width\":4, \"filter_height\":4, \"filter_depth\":4, \"filter_count\":128, \"stride\":2, \"padding\":1, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"conv3d_t\", \"filter_width\":4, \"filter_height\":4, \"filter_depth\":4, \"filter_count\":64, \"stride\":2, \"padding\":1, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"conv3d_t\", \"filter_width\":4, \"filter_height\":4, \"filter_depth\":4, \"filter_count\":32, \"stride\":2, \"padding\":1, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"conv3d_t\", \"filter_width\":4, \"filter_height\":4, \"filter_depth\":4, \"filter_count\":1, \"stride\":2, \"padding\":1, \"activation\":\"sigmoid\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":16, \"l2_decay\":0.0001}\n"
		"]\n";
}

String Model3DApp::BuildPointNetConfig() {
	return
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":2048, \"input_height\":3, \"input_depth\":1},\n"
		"  { \"type\" : \"conv\", \"filter_width\":1, \"filter_height\":1, \"filter_count\":64, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"conv\", \"filter_width\":1, \"filter_height\":1, \"filter_count\":128, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"conv\", \"filter_width\":1, \"filter_height\":1, \"filter_count\":1024, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"max_pool\", \"pool_width\":1, \"pool_height\":2048, \"stride\":1},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"dropout\", \"dropout_rate\":0.3},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":256, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"dropout\", \"dropout_rate\":0.3},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":40, \"activation\":\"softmax\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":32, \"l2_decay\":0.0001}\n"
		"]\n";
}

String Model3DApp::BuildOccupancyNetConfig() {
	return
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":4, \"input_height\":1, \"input_depth\":1},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"leaky_relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"leaky_relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"leaky_relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"leaky_relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"leaky_relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"leaky_relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":64, \"l2_decay\":0.0001}\n"
		"]\n";
}

String Model3DApp::BuildDeepSDFConfig() {
	return
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":512, \"input_height\":1, \"input_depth\":1},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"add\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":1, \"activation\":\"linear\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":32, \"l2_decay\":0.0001}\n"
		"]\n";
}

Model3DControl::Model3DControl() { modelType = "voxel"; }

void Model3DControl::Paint(Draw& draw) {
	Size sz = GetSize();
	draw.DrawRect(sz, SColorFace());
	draw.DrawText(10, 10, "3D Visualization", StdFont(), Black());

	if (modelType == "voxel") {
		int gridSize = 8;
		int cellSize = min(sz.cx, sz.cy) / (gridSize + 2);
		int offsetX = (sz.cx - gridSize * cellSize) / 2;
		int offsetY = (sz.cy - gridSize * cellSize) / 2;

		for (int i = 0; i < gridSize; i++) {
			for (int j = 0; j < gridSize; j++) {
				if (Random(100) < 30) {
					draw.DrawRect(offsetX + i * cellSize, offsetY + j * cellSize, cellSize, cellSize, LtBlue());
				}
				draw.DrawRect(offsetX + i * cellSize, offsetY + j * cellSize, cellSize, cellSize, Gray());
			}
		}
	}
	else if (modelType == "pointnet") {
		for(int i = 0; i < points.GetCount(); i++) {
			Point p = points[i];
			int x = (p.x * (sz.cx - 20)) / 200 + 10;
			int y = (p.y * (sz.cy - 20)) / 200 + 10;
			draw.DrawRect(x, y, 3, 3, Blue());
		}
	}
}

void Model3DControl::SetModelType(const String& type) { modelType = type; Refresh(); }
void Model3DControl::SetPointCloud(const Vector<Point>& pts) { points = clone(pts); Refresh(); }
void Model3DControl::SetMesh(const Vector<Pointf>& verts, const Vector<int>& idx) { vertices = clone(verts); indices = clone(idx); Refresh(); }
