#include "AnimationLearning.h"

AnimationApp::AnimationApp()
{
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("Animation Learning and Generation with Skeleton Movement");

	// Define a default animation architecture (LSTM-based as starting point)
	t = BuildLSTMSkeletonConfig();

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

	// Setup animation-specific UI controls manually
	Add(anim_ctrl.HSizePos());
	anim_ctrl << anim_display;

	// Setup animation type selector
	anim_type.Add(0, "LSTM Skeleton");
	anim_type.Add(1, "Transformer Skeleton");
	anim_type.Add(2, "VAE Skeleton");
	anim_type.SetIndex(0);
	anim_type <<= THISBACK(OnAnimTypeChanged);

	// Setup buttons
	generate_anim_btn.SetLabel("Generate Animation");
	train_rnn_btn.SetLabel("Train RNN/LSTM");
	train_transformer_btn.SetLabel("Train Transformer");
	load_motion_btn.SetLabel("Load Motion");
	save_motion_btn.SetLabel("Save Motion");
	play_btn.SetLabel("Play");
	pause_btn.SetLabel("Pause");
	stop_btn.SetLabel("Stop");

	generate_anim_btn <<= THISBACK(OnGenerateAnim);
	train_rnn_btn <<= THISBACK(OnTrainRNN);
	train_transformer_btn <<= THISBACK(OnTrainTransformer);
	load_motion_btn <<= THISBACK(OnLoadMotion);
	save_motion_btn <<= THISBACK(OnSaveMotion);
	play_btn <<= THISBACK(OnPlay);
	pause_btn <<= THISBACK(OnPause);
	stop_btn <<= THISBACK(OnStop);

	// Add controls manually
	anim_ctrl.Add(anim_type.HSizePos().TopPos(0, 24));
	anim_ctrl.Add(generate_anim_btn.HSizePos().TopPos(24, 24));
	anim_ctrl.Add(train_rnn_btn.HSizePos().TopPos(48, 24));
	anim_ctrl.Add(train_transformer_btn.HSizePos().TopPos(72, 24));
	anim_ctrl.Add(load_motion_btn.HSizePos().TopPos(96, 24));
	anim_ctrl.Add(save_motion_btn.HSizePos().TopPos(120, 24));
	anim_ctrl.Add(play_btn.HSizePos().TopPos(144, 24));
	anim_ctrl.Add(pause_btn.HSizePos().TopPos(168, 24));
	anim_ctrl.Add(stop_btn.HSizePos().TopPos(192, 24));
	anim_ctrl.Add(anim_display.HSizePos().VSizePos(216, 0));

	PostCallback(THISBACK(Refresher));
}

AnimationApp::~AnimationApp() {
	ses.StopTraining();
}

void AnimationApp::DockInit() {
	DockLeft(Dockable(settings, "Settings").SizeHint(Size(320, 11*20)));
	DockLeft(Dockable(graph, "Loss").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void AnimationApp::UpdateNetParamDisplay() {
	TrainerBase& trainer = ses.GetTrainer();
	rate.SetData(trainer.GetLearningRate());
	mom.SetData(trainer.GetMomentum());
	batch.SetData(trainer.GetBatchSize());
	decay.SetData(trainer.GetL2Decay());
}

void AnimationApp::ApplySettings() {
	TrainerBase& trainer = ses.GetTrainer();
	trainer.SetLearningRate(rate.GetData());
	trainer.SetMomentum(mom.GetData());
	trainer.SetBatchSize(batch.GetData());
	trainer.SetL2Decay(decay.GetData());
}

void AnimationApp::Pause() {
	if (ses.IsTraining())
		ses.StopTraining();
	else
		ses.StartTraining();
}

void AnimationApp::OpenFile() {
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

void AnimationApp::SaveFile() {
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

void AnimationApp::Reload() {
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

void AnimationApp::RefreshStatus() {
	String s;
	s << "   Forward time per example: " << ses.GetForwardTime() << "\n";
	s << "   Backprop time per example: " << ses.GetBackwardTime() << "\n";
	s << "   Sequence loss: " << ses.GetLossAverage() << "\n";
	s << "   L2 Weight decay loss: " << ses.GetL2DecayLossAverage() << "\n";
	s << "   Examples seen: " << ses.GetStepCount();
	status.SetLabel(s);
}

void AnimationApp::Refresher() {
	layer_view.Refresh();
	graph.RefreshData();
	RefreshStatus();
	anim_display.Refresh();

	PostCallback(THISBACK(Refresher));
}

void AnimationApp::ResetAll() {
	UpdateNetParamDisplay();
	graph.Clear();
}

void AnimationApp::OnGenerateAnim() {
	Vector<Pointf> joints;
	joints.Add(Pointf(100, 50));
	joints.Add(Pointf(100, 100));
	
	Vector<Vector<int>> connections;
	Vector<int> conn;
	conn.Add(0); conn.Add(1);
	connections.Add(clone(conn));
	
	anim_display.SetJoints(joints);
	anim_display.SetConnections(connections);
	anim_display.SetTotalFrames(30);
	anim_display.Refresh();
	
	PromptOK("Generated animation sequence (simulated)");
}

void AnimationApp::OnTrainRNN() { PromptOK("Training RNN/LSTM for animation would start here"); }
void AnimationApp::OnTrainTransformer() { PromptOK("Training Transformer for animation would start here"); }

void AnimationApp::OnLoadMotion() {
	String file = SelectFileOpen("Motion files\t*.bvh;*.trc;*.c3d\nAll files\t*.*");
	if (file.IsEmpty()) return;
	LOG("Motion data loaded from: " + file);
	PromptOK("Motion data loaded (simulated)");
}

void AnimationApp::OnSaveMotion() {
	String file = SelectFileSaveAs("Motion files\t*.bvh;*.trc;*.c3d\nAll files\t*.*");
	if (file.IsEmpty()) return;
	LOG("Motion data saved to: " + file);
	PromptOK("Motion data saved (simulated)");
}

void AnimationApp::OnPlay() { anim_display.SetFrame(0); anim_display.Refresh(); }
void AnimationApp::OnPause() {}
void AnimationApp::OnStop() { anim_display.SetFrame(0); anim_display.Refresh(); }

void AnimationApp::OnAnimTypeChanged() {
	int idx = anim_type.GetIndex();
	switch(idx) {
		case 0: t = BuildLSTMSkeletonConfig(); break;
		case 1: t = BuildTransformerSkeletonConfig(); break;
		case 2: t = BuildVAESkeletonConfig(); break;
		default: t = BuildLSTMSkeletonConfig(); break;
	}
	net_edit.SetData(t);
	Reload();
}

String AnimationApp::BuildLSTMSkeletonConfig() {
	return 
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":64, \"input_height\":21, \"input_depth\":1},\n"
		"  { \"type\" : \"lstm\", \"hidden_size\":512, \"num_layers\":2, \"batch_first\":true, \"dropout\":0.1},\n"
		"  { \"type\" : \"lstm\", \"hidden_size\":512, \"num_layers\":1, \"batch_first\":true, \"dropout\":0.1},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":1024, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"dropout\", \"dropout_rate\":0.3},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"dropout\", \"dropout_rate\":0.3},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":21, \"activation\":\"linear\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.0005, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":32, \"l2_decay\":0.0001}\n"
		"]\n";
}

String AnimationApp::BuildTransformerSkeletonConfig() {
	return
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":64, \"input_height\":21, \"input_depth\":1},\n"
		"  { \"type\" : \"positional_encoding\", \"d_model\":256},\n"
		"  { \"type\" : \"transformer_encoder\", \"d_model\":256, \"nhead\":8, \"num_layers\":4, \"dim_feedforward\":512, \"dropout\":0.1},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"dropout\", \"dropout_rate\":0.2},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":256, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"dropout\", \"dropout_rate\":0.2},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":21, \"activation\":\"linear\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":16, \"l2_decay\":0.0001}\n"
		"]\n";
}

String AnimationApp::BuildVAESkeletonConfig() {
	return
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":64, \"input_height\":21, \"input_depth\":1},\n"
		"  { \"type\" : \"lstm\", \"hidden_size\":256, \"num_layers\":2, \"batch_first\":true, \"dropout\":0.1},\n"
		"  { \"type\" : \"flatten\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":128, \"activation\":\"linear\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":128, \"activation\":\"linear\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":256, \"activation\":\"relu\"},\n"
		"  { \"type\" : \"reshape\", \"dim1\":64, \"dim2\":4},\n"
		"  { \"type\" : \"lstm_t\", \"hidden_size\":256, \"num_layers\":2, \"batch_first\":true, \"dropout\":0.1},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":21, \"activation\":\"linear\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.0005, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":32, \"l2_decay\":0.0001}\n"
		"]\n";
}

AnimationControl::AnimationControl() { currentFrame = 0; totalFrames = 0; }

void AnimationControl::Paint(Draw& draw) {
	Size sz = GetSize();
	draw.DrawRect(sz, SColorFace());
	draw.DrawText(10, 10, "Skeleton Animation", StdFont(), Black());

	if (joints.GetCount() > 0) {
		for (int i = 0; i < connections.GetCount(); i++) {
			const Vector<int>& conn = connections[i];
			if (conn.GetCount() >= 2) {
				Pointf start = joints[conn[0]];
				Pointf end = joints[conn[1]];
				Pointf scaledStart((start.x / 200) * sz.cx, (start.y / 200) * sz.cy);
				Pointf scaledEnd((end.x / 200) * sz.cx, (end.y / 200) * sz.cy);
				draw.DrawLine(scaledStart, scaledEnd, 2, Blue());
			}
		}
		for (int i = 0; i < joints.GetCount(); i++) {
			Pointf joint = joints[i];
			Point scaledJoint((int)((joint.x / 200) * sz.cx), (int)((joint.y / 200) * sz.cy));
			draw.DrawEllipse(scaledJoint.x - 4, scaledJoint.y - 4, 8, 8, Red(), 2, Red());
		}
	}
}

void AnimationControl::SetJoints(const Vector<Pointf>& jnts) { joints = clone(jnts); Refresh(); }
void AnimationControl::SetConnections(const Vector<Vector<int>>& conns) { connections = clone(conns); Refresh(); }
void AnimationControl::SetFrame(int frame) { currentFrame = frame; Refresh(); }
void AnimationControl::SetTotalFrames(int total) { totalFrames = total; Refresh(); }
