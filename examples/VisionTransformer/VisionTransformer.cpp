#include "VisionTransformer.h"

// ViT implementation
VisionTransformer::VisionTransformer()
    : loader(ses)
{
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("Vision Transformer on CIFAR-10");

	// Define a Vision Transformer architecture
	t = BuildViTConfig();

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

	PostCallback(THISBACK(Refresher));
}

VisionTransformer::~VisionTransformer() {
	ses.StopTraining();
}

void VisionTransformer::DockInit() {
	DockLeft(Dockable(settings, "Settings").SizeHint(Size(320, 11*20)));
	DockLeft(Dockable(graph, "Loss").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void VisionTransformer::UpdateNetParamDisplay() {
	TrainerBase& trainer = ses.GetTrainer();
	rate.SetData(trainer.GetLearningRate());
	mom.SetData(trainer.GetMomentum());
	batch.SetData(trainer.GetBatchSize());
	decay.SetData(trainer.GetL2Decay());
}

void VisionTransformer::ApplySettings() {
	TrainerBase& trainer = ses.GetTrainer();
	trainer.SetLearningRate(rate.GetData());
	trainer.SetMomentum(mom.GetData());
	trainer.SetBatchSize(batch.GetData());
	trainer.SetL2Decay(decay.GetData());
}

void VisionTransformer::Pause() {
	if (ses.IsTraining())
		ses.StopTraining();
	else
		ses.StartTraining();
}

void VisionTransformer::OpenFile() {
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

void VisionTransformer::SaveFile() {
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

void VisionTransformer::Reload() {
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

void VisionTransformer::RefreshStatus() {
	String s;
	s << "   Forward time per example: " << ses.GetForwardTime() << "\n";
	s << "   Backprop time per example: " << ses.GetBackwardTime() << "\n";
	s << "   Classification loss: " << ses.GetLossAverage() << "\n";
	s << "   L2 Weight decay loss: " << ses.GetL2DecayLossAverage() << "\n";
	s << "   Training accuracy: " << ses.GetTrainingAccuracyAverage() << "\n";
	s << "   Validation accuracy: " << ses.GetValidationAccuracyAverage() << "\n";
	s << "   Examples seen: " << ses.GetStepCount();
	status.SetLabel(s);
}

void VisionTransformer::Refresher() {
	layer_view.Refresh();
	graph.RefreshData();
	RefreshStatus();

	PostCallback(THISBACK(Refresher));
}

void VisionTransformer::ResetAll() {
	UpdateNetParamDisplay();
	graph.Clear();
}

String VisionTransformer::BuildViTConfig() {
	// Vision Transformer architecture for CIFAR-10
	return
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":32, \"input_height\":32, \"input_depth\":3},\n"
		"  { \"type\" : \"patch_embed\", \"patch_size\":4, \"embed_dim\":128},\n"
		"  { \"type\" : \"vit_block\", \"embed_dim\":128, \"num_heads\":8, \"mlp_ratio\":4.0, \"dropout\":0.1},\n"
		"  { \"type\" : \"vit_block\", \"embed_dim\":128, \"num_heads\":8, \"mlp_ratio\":4.0, \"dropout\":0.1},\n"
		"  { \"type\" : \"vit_block\", \"embed_dim\":128, \"num_heads\":8, \"mlp_ratio\":4.0, \"dropout\":0.1},\n"
		"  { \"type\" : \"layer_norm\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":10, \"activation\":\"softmax\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":64, \"l2_decay\":0.0001}\n"
		"]\n";
}
