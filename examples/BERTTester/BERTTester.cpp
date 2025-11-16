#include "BERTTester.h"

#define IMAGECLASS BERTTesterImg
#define IMAGEFILE <BERTTester/BERTTester.iml>
#include <Draw/iml_source.h>

// BERT implementation
BERTTester::BERTTester()
{
	Icon(BERTTesterImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("BERT on Text Data");

	// Define a BERT architecture using the existing transformer layers
	t =		"[\n"
			"\t{\"type\":\"input\", \"input_width\":512, \"input_height\":1, \"input_depth\":768},\n"  // Input sequence of max length 512, embedding dim 768
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"  // First attention layer with masking
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"  // More attention layers
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":30522, \"activation\":\"softmax\"},\n"  // Output for vocab size ~30K
			"\t{\"type\":\"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":16, \"l2_decay\":0.01}\n"
			"]\n";

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

BERTTester::~BERTTester() {
	ses.StopTraining();
}

void BERTTester::DockInit() {
	DockLeft(Dockable(settings, "Settings").SizeHint(Size(320, 11*20)));
	DockLeft(Dockable(graph, "Loss").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void BERTTester::UpdateNetParamDisplay() {
	TrainerBase& trainer = ses.GetTrainer();
	rate.SetData(trainer.GetLearningRate());
	mom.SetData(trainer.GetMomentum());
	batch.SetData(trainer.GetBatchSize());
	decay.SetData(trainer.GetL2Decay());
}

void BERTTester::ApplySettings() {
	TrainerBase& trainer = ses.GetTrainer();
	trainer.SetLearningRate(rate.GetData());
	trainer.SetMomentum(mom.GetData());
	trainer.SetBatchSize(batch.GetData());
	trainer.SetL2Decay(decay.GetData());
}

void BERTTester::Pause() {
	if (ses.IsTraining())
		ses.StopTraining();
	else
		ses.StartTraining();
}

void BERTTester::OpenFile() {
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

void BERTTester::SaveFile() {
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

void BERTTester::Reload() {
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

void BERTTester::RefreshStatus() {
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

void BERTTester::Refresher() {
	layer_view.Refresh();
	graph.RefreshData();
	RefreshStatus();

	PostCallback(THISBACK(Refresher));
}

void BERTTester::ResetAll() {
	UpdateNetParamDisplay();
	graph.Clear();
}