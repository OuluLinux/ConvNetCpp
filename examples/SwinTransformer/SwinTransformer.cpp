#include "SwinTransformer.h"

#define IMAGECLASS SwinTransformerImg
#define IMAGEFILE <SwinTransformer/SwinTransformer.iml>
#include <Draw/iml_source.h>

// Swin Transformer implementation
SwinTransformer::SwinTransformer()
{
	Icon(SwinTransformerImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("Swin Transformer on CIFAR-10");

	// Define a Swin Transformer architecture using the existing transformer layers
	t =		"[\n"
			"\t{\"type\":\"input\", \"input_width\":32, \"input_height\":32, \"input_depth\":3},\n"
			"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":96, \"stride\":4, \"pad\":0, \"activation\":\"gelu\"},\n"  // Patch embedding using conv layer
			"\t{\"type\":\"swin_block\", \"dim\":96, \"input_resolution\":[8,8], \"num_heads\":3, \"window_size\":7, \"shift_size\":0},\n"  // First Swin block
			"\t{\"type\":\"swin_block\", \"dim\":96, \"input_resolution\":[8,8], \"num_heads\":3, \"window_size\":7, \"shift_size\":3},\n"  // Shifted window Swin block
			"\t{\"type\":\"swin_patch_merge\", \"dim\":96, \"out_dim\":192},\n"  // Patch merging - downsample
			"\t{\"type\":\"swin_block\", \"dim\":192, \"input_resolution\":[4,4], \"num_heads\":6, \"window_size\":7, \"shift_size\":0},\n"  // Second stage Swin block
			"\t{\"type\":\"swin_block\", \"dim\":192, \"input_resolution\":[4,4], \"num_heads\":6, \"window_size\":7, \"shift_size\":3},\n"  // Shifted window Swin block
			"\t{\"type\":\"swin_patch_merge\", \"dim\":192, \"out_dim\":384},\n"  // Patch merging - downsample
			"\t{\"type\":\"swin_block\", \"dim\":384, \"input_resolution\":[2,2], \"num_heads\":12, \"window_size\":2, \"shift_size\":0},\n"  // Third stage Swin block
			"\t{\"type\":\"swin_block\", \"dim\":384, \"input_resolution\":[2,2], \"num_heads\":12, \"window_size\":2, \"shift_size\":1},\n"  // Shifted window Swin block
			"\t{\"type\":\"swin_patch_merge\", \"dim\":384, \"out_dim\":768},\n"  // Patch merging - downsample
			"\t{\"type\":\"swin_block\", \"dim\":768, \"input_resolution\":[1,1], \"num_heads\":24, \"window_size\":1, \"shift_size\":0},\n"  // Final stage Swin block
			"\t{\"type\":\"fc\", \"neuron_count\":10, \"activation\":\"softmax\"},\n"  // Classifier
			"\t{\"type\":\"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":32, \"l2_decay\":0.0001}\n"
			"]\n";

	// Image settings for CIFAR-10
	img_sz = Size(32,32);
	augmentation = 32;
	do_flip = true;
	has_colors = true;

	net_edit.SetData(t);

	average_size = 10;
	max_diff_imgs = 100000; // not limiting currently

	UpdateNetParamDisplay();

	Add(v_split.SizePos());
	v_split.Vert();

	v_split << layer_view << pred_view;
	v_split.SetPos(6400);

	ses.SetTestPredict(true);
	ses.SetAugmentation(augmentation, do_flip);

	pred_view.SetSession(ses);
	pred_view.SetAugmentation(augmentation, do_flip);

	net_ctrl.Add(net_edit.HSizePos().VSizePos(0,30));
	net_ctrl.Add(reload_btn.HSizePos().BottomPos(0,30));
	reload_btn.SetLabel("Reload Network");
	reload_btn <<= THISBACK(Reload);

	// Settings panel
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
	rate.SetData(0.0001);
	mom.SetData(0.9);
	batch.SetData(32);
	decay.SetData(0.0001);

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