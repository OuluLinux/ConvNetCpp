#include "SwinTransformer.h"

// Swin Transformer implementation
SwinTransformer::SwinTransformer()
    : loader(ses)
{
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("Swin Transformer on CIFAR-10");

	// Define a Swin Transformer architecture
	t = BuildSwinConfig();

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

SwinTransformer::~SwinTransformer() {
	ses.StopTraining();
}

void SwinTransformer::DockInit() {
	DockLeft(Dockable(settings, "Settings").SizeHint(Size(320, 11*20)));
	DockLeft(Dockable(graph, "Loss").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void SwinTransformer::UpdateNetParamDisplay() {
	TrainerBase& trainer = ses.GetTrainer();
	rate.SetData(trainer.GetLearningRate());
	mom.SetData(trainer.GetMomentum());
	batch.SetData(trainer.GetBatchSize());
	decay.SetData(trainer.GetL2Decay());
}

void SwinTransformer::ApplySettings() {
	TrainerBase& trainer = ses.GetTrainer();
	trainer.SetLearningRate(rate.GetData());
	trainer.SetMomentum(mom.GetData());
	trainer.SetBatchSize(batch.GetData());
	trainer.SetL2Decay(decay.GetData());
}

void SwinTransformer::Pause() {
	if (ses.IsTraining())
		ses.StopTraining();
	else
		ses.StartTraining();
}

void SwinTransformer::OpenFile() {
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

void SwinTransformer::SaveFile() {
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

void SwinTransformer::Reload() {
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

void SwinTransformer::RefreshStatus() {
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

void SwinTransformer::Refresher() {
	layer_view.Refresh();
	graph.RefreshData();
	RefreshStatus();

	PostCallback(THISBACK(Refresher));
}

void SwinTransformer::ResetAll() {
	UpdateNetParamDisplay();
	graph.Clear();
}

String SwinTransformer::BuildSwinConfig() {
	// Hierarchical Swin Transformer architecture for CIFAR-10
	return
		"[\n"
		"  { \"type\" : \"input\", \"input_width\":32, \"input_height\":32, \"input_depth\":3},\n"
		"  { \"type\" : \"patch_embed\", \"patch_size\":2, \"embed_dim\":96},\n"
		"  { \"type\" : \"swin_block\", \"embed_dim\":96, \"num_heads\":3, \"window_size\":7, \"shift_size\":0},\n"
		"  { \"type\" : \"swin_block\", \"embed_dim\":96, \"num_heads\":3, \"window_size\":7, \"shift_size\":3},\n"
		"  { \"type\" : \"patch_merging\", \"input_dim\":96},\n"
		"  { \"type\" : \"swin_block\", \"embed_dim\":192, \"num_heads\":6, \"window_size\":7, \"shift_size\":0},\n"
		"  { \"type\" : \"layer_norm\"},\n"
		"  { \"type\" : \"fc\", \"neuron_count\":10, \"activation\":\"softmax\"},\n"
		"  { \"type\" : \"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":32, \"l2_decay\":0.0001}\n"
		"]\n";
}
