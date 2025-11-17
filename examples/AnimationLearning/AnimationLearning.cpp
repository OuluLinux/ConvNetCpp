#include "AnimationLearning.h"

#define IMAGECLASS AnimImg
#define IMAGEFILE <AnimationLearning/AnimationLearning.iml>
#include <Draw/iml_source.h>

AnimationApp::AnimationApp()
{
	Icon(AnimImg::icon());
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

	// Setup animation-specific UI controls
	Add(anim_ctrl.SizeHorz());
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

	// Add controls to the layout
	CtrlLayout(anim_ctrl, "Animation Options");
	anim_ctrl.Add(anim_type.HSizePos().TopPos(0, 24));
	anim_ctrl.Add(generate_anim_btn.HSizePos().TopPos(24, 48));
	anim_ctrl.Add(train_rnn_btn.HSizePos().TopPos(48, 72));
	anim_ctrl.Add(train_transformer_btn.HSizePos().TopPos(72, 96));
	anim_ctrl.Add(load_motion_btn.HSizePos().TopPos(96, 120));
	anim_ctrl.Add(save_motion_btn.HSizePos().TopPos(120, 144));
	anim_ctrl.Add(play_btn.HSizePos().TopPos(144, 168));
	anim_ctrl.Add(pause_btn.HSizePos().TopPos(168, 192));
	anim_ctrl.Add(stop_btn.HSizePos().TopPos(192, 216));
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

	String net_str = net_edit.GetData();

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
	// In a real implementation, this would generate an animation sequence using the trained network
	// For demo, we'll just create a simple stick figure animation
	
	Vector<Pointf> joints;
	// Simple stick figure (2 joints: head and body)
	joints.Add(Pointf(100, 50));   // Head
	joints.Add(Pointf(100, 100));  // Body/torso
	
	Vector<Vector<int>> connections;
	Vector<int> conn;
	conn.Add(0);  // Head
	conn.Add(1);  // Body
	connections.Add(conn);
	
	anim_display.SetJoints(joints);
	anim_display.SetConnections(connections);
	anim_display.SetTotalFrames(30); // 30 frames for demo
	anim_display.Refresh();
	
	PromptOK("Generated animation sequence (simulated)");
}

void AnimationApp::OnTrainRNN() {
	PromptOK("Training RNN/LSTM for animation would start here");
	
	// In a real implementation:
	// 1. Load skeleton movement data (joint positions over time)
	// 2. Set up RNN/LSTM architecture for sequence modeling
	// 3. Train to predict next joint positions given history
}

void AnimationApp::OnTrainTransformer() {
	PromptOK("Training Transformer for animation would start here");
	
	// In a real implementation:
	// 1. Load skeleton movement data (joint positions over time)
	// 2. Set up Transformer architecture for sequence modeling
	// 3. Train with attention mechanisms to capture long-term dependencies
}

void AnimationApp::OnLoadMotion() {
	// Allow user to load a motion capture file (BVH, etc.)
	String file = SelectFileOpen("Motion files\t*.bvh;*.trc;*.c3d\nAll files\t*.*");
	if (file.IsEmpty()) return;

	if (!FileExists(file)) {
		PromptOK("File does not exist");
		return;
	}

	// In a real implementation, this would load the motion capture data
	// For now, we'll just show a message
	LOG("Motion data loaded from: " + file);
	PromptOK("Motion data loaded (simulated)");
}

void AnimationApp::OnSaveMotion() {
	// Allow user to save the current animation
	String file = SelectFileSaveAs("Motion files\t*.bvh;*.trc;*.c3d\nAll files\t*.*");
	if (file.IsEmpty()) return;

	// In a real implementation, this would save the animation data
	// For now, we'll just show a message
	LOG("Motion data saved to: " + file);
	PromptOK("Motion data saved (simulated)");
}

void AnimationApp::OnPlay() {
	// In a real implementation, this would start the animation playback
	// For now, we'll just simulate it with a message
	LOG("Animation playing...");
	anim_display.SetFrame(0);
	anim_display.Refresh();
}

void AnimationApp::OnPause() {
	// In a real implementation, this would pause the animation playback
	LOG("Animation paused");
}

void AnimationApp::OnStop() {
	// In a real implementation, this would stop the animation playback
	LOG("Animation stopped");
	anim_display.SetFrame(0);
	anim_display.Refresh();
}

void AnimationApp::OnAnimTypeChanged() {
	int idx = anim_type.GetIndex();

	switch(idx) {
		case 0:
			t = BuildLSTMSkeletonConfig();
			break;
		case 1:
			t = BuildTransformerSkeletonConfig();
			break;
		case 2:
			t = BuildVAESkeletonConfig();
			break;
		default:
			t = BuildLSTMSkeletonConfig();
			break;
	}

	// Update network with new configuration
	net_edit.SetData(t);
	Reload();
}

// Animation-specific architectures
String AnimationApp::BuildLSTMSkeletonConfig() {
	// LSTM-based architecture for skeleton animation
	// Input: sequence of joint positions over time
	return 
		"[\n"
		"\t{\"type\":\"input\", \"input_width\":64, \"input_height\":21, \"input_depth\":1},\n"  // 64 timesteps, 21 joints (simplified skeleton)
		
		// LSTM layers to model temporal dependencies
		"\t{\"type\":\"lstm\", \"hidden_size\":512, \"num_layers\":2, \"batch_first\":true, \"dropout\":0.1},\n"
		"\t{\"type\":\"lstm\", \"hidden_size\":512, \"num_layers\":1, \"batch_first\":true, \"dropout\":0.1},\n"
		
		// Dense layers for output
		"\t{\"type\":\"fc\", \"neuron_count\":1024, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"dropout\", \"dropout_rate\":0.3},\n"
		"\t{\"type\":\"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"dropout\", \"dropout_rate\":0.3},\n"
		
		// Output layer - next frame joint positions
		"\t{\"type\":\"fc\", \"neuron_count\":21, \"activation\":\"linear\"},\n"  // 21 joint positions (x, y for 10+ joints + root)
		
		// Optimizer settings
		"\t{\"type\":\"adam\", \"learning_rate\":0.0005, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":32, \"l2_decay\":0.0001}\n"
		"]\n";
}

String AnimationApp::BuildTransformerSkeletonConfig() {
	// Transformer-based architecture for skeleton animation
	// Uses self-attention to model temporal and spatial relationships
	return
		"[\n"
		"\t{\"type\":\"input\", \"input_width\":64, \"input_height\":21, \"input_depth\":1},\n"  // 64 timesteps, 21 joints
		
		// Positional encoding for temporal information
		"\t{\"type\":\"positional_encoding\", \"d_model\":256},\n"
		
		// Transformer encoder layers
		"\t{\"type\":\"transformer_encoder\", \"d_model\":256, \"nhead\":8, \"num_layers\":4, \"dim_feedforward\":512, \"dropout\":0.1},\n"
		
		// Processing layers
		"\t{\"type\":\"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"dropout\", \"dropout_rate\":0.2},\n"
		"\t{\"type\":\"fc\", \"neuron_count\":256, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"dropout\", \"dropout_rate\":0.2},\n"
		
		// Output layer - next frame joint positions
		"\t{\"type\":\"fc\", \"neuron_count\":21, \"activation\":\"linear\"},\n"
		
		// Optimizer settings
		"\t{\"type\":\"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":16, \"l2_decay\":0.0001}\n"
		"]\n";
}

String AnimationApp::BuildVAESkeletonConfig() {
	// VAE-based architecture for skeleton animation
	// Learns a latent space of movement patterns
	return
		"[\n"
		"\t{\"type\":\"input\", \"input_width\":64, \"input_height\":21, \"input_depth\":1},\n"  // 64 timesteps, 21 joints
		
		// Encoder: compress sequence to latent space
		"\t{\"type\":\"lstm\", \"hidden_size\":256, \"num_layers\":2, \"batch_first\":true, \"dropout\":0.1},\n"
		"\t{\"type\":\"flatten\"},\n"
		"\t{\"type\":\"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		
		// VAE bottleneck (mean and variance)
		"\t{\"type\":\"fc\", \"neuron_count\":128, \"activation\":\"linear\"},\n"  // Mean
		"\t{\"type\":\"fc\", \"neuron_count\":128, \"activation\":\"linear\"},\n"  // Variance
		
		// Decoder: reconstruct from latent space
		"\t{\"type\":\"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"fc\", \"neuron_count\":256, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"reshape\", \"dim1\":64, \"dim2\":4},\n"  // Reshape for LSTM
		"\t{\"type\":\"lstm_t\", \"hidden_size\":256, \"num_layers\":2, \"batch_first\":true, \"dropout\":0.1},\n"
		
		// Output layer - reconstructed joint positions
		"\t{\"type\":\"fc\", \"neuron_count\":21, \"activation\":\"linear\"},\n"
		
		// Optimizer settings
		"\t{\"type\":\"adam\", \"learning_rate\":0.0005, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":32, \"l2_decay\":0.0001}\n"
		"]\n";
}

// AnimationControl implementation
AnimationControl::AnimationControl() {
	currentFrame = 0;
	totalFrames = 0;
}

void AnimationControl::Paint(Draw& draw) {
	Size sz = GetSize();
	draw.DrawRect(sz, SColorFace());

	// Draw animation visualization
	draw.DrawText(10, 10, "Skeleton Animation Visualization", StdFont(), Black);

	// Draw frame counter if we have frames
	if (totalFrames > 0) {
		String frameText = Format("Frame %d/%d", currentFrame + 1, totalFrames);
		draw.DrawText(10, 30, frameText, StdFont(), Black);
	}

	// Draw joints and connections if available
	if (joints.GetCount() > 0) {
		// Scale joints to fit in the control
		for (int i = 0; i < connections.GetCount(); i++) {
			Vector<int> conn = connections[i];
			if (conn.GetCount() >= 2) {
				Pointf start = joints[conn[0]];
				Pointf end = joints[conn[1]];

				// Scale to control size
				Pointf scaledStart((start.x / 200) * sz.cx, (start.y / 200) * sz.cy);
				Pointf scaledEnd((end.x / 200) * sz.cx, (end.y / 200) * sz.cy);

				// Draw bone/connection
				draw.DrawLine(scaledStart, scaledEnd, 2, Blue);
			}
		}

		// Draw joints
		for (int i = 0; i < joints.GetCount(); i++) {
			Pointf joint = joints[i];
			Point scaledJoint((joint.x / 200) * sz.cx, (joint.y / 200) * sz.cy);

			// Draw joint as a circle
			draw.DrawEllipse(scaledJoint.x - 4, scaledJoint.y - 4, 8, 8, Red, 2, LtRed);
		}
	} else {
		// Draw a more detailed stick figure as default
		int centerX = sz.cx / 2;
		int centerY = sz.cy / 2;

		// Draw head
		draw.DrawEllipse(centerX - 10, centerY - 50, 20, 20, Black, 1, White);

		// Draw body
		draw.DrawLine(Point(centerX, centerY - 30), Point(centerX, centerY + 20), 2, Black);

		// Draw arms
		draw.DrawLine(Point(centerX, centerY - 10), Point(centerX - 25, centerY), 2, Black);  // Left arm
		draw.DrawLine(Point(centerX, centerY - 10), Point(centerX + 25, centerY), 2, Black);  // Right arm

		// Draw legs
		draw.DrawLine(Point(centerX, centerY + 20), Point(centerX - 20, centerY + 50), 2, Black);  // Left leg
		draw.DrawLine(Point(centerX, centerY + 20), Point(centerX + 20, centerY + 50), 2, Black);  // Right leg

		// Draw a timeline if we have frames
		if (totalFrames > 0) {
			int timelineX = 50;
			int timelineY = sz.cy - 30;
			int timelineWidth = sz.cx - 100;
			int timelineHeight = 10;

			// Draw timeline background
			draw.DrawRect(timelineX, timelineY, timelineWidth, timelineHeight, Gray);

			// Draw current frame position
			int framePos = timelineX + (currentFrame * timelineWidth) / totalFrames;
			draw.DrawRect(framePos - 2, timelineY - 5, 4, timelineHeight + 10, Red);
		}
	}
}

void AnimationControl::SetJoints(const Vector<Pointf>& jnts) {
	joints = jnts;
	Refresh();
}

void AnimationControl::SetConnections(const Vector<Vector<int>>& conns) {
	connections = conns;
	Refresh();
}

void AnimationControl::SetFrame(int frame) {
	currentFrame = frame;
	Refresh();
}

void AnimationControl::SetTotalFrames(int total) {
	totalFrames = total;
	Refresh();
}