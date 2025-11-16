#include "VisionTransformer.h"

#define IMAGECLASS VisionTransformerImg
#define IMAGEFILE <VisionTransformer/VisionTransformer.iml>
#include <Draw/iml_source.h>

// Vision Transformer implementation
VisionTransformer::VisionTransformer()
{
	Icon(VisionTransformerImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("Vision Transformer on CIFAR-10");

	// Define a Vision Transformer architecture using the existing transformer layers
	// This is a simplified version that demonstrates the concept
	t =		"[\n"
			"\t{\"type\":\"input\", \"input_width\":32, \"input_height\":32, \"input_depth\":3},\n"
			"\t{\"type\":\"vit_patch_embed\", \"patch_size\":4, \"embed_dim\":256, \"num_patches\":64},\n"  // Patch embedding: 32x32 image -> patches of 4x4 -> 64 patches of 48 dim (3x4x4)
			"\t{\"type\":\"vit_encoder\", \"embed_dim\":256, \"num_heads\":8, \"ff_dim\":512, \"num_layers\":6},\n"
			"\t{\"type\":\"vit_classifier\", \"num_classes\":10, \"embed_dim\":256},\n"
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