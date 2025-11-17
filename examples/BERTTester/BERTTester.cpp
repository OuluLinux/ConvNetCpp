#include "BERTTester.h"

#define IMAGECLASS BERTTesterImg
#define IMAGEFILE <BERTTester/BERTTester.iml>
#include <Draw/iml_source.h>

// BERT implementation with MLM and NSP tasks
BERTTester::BERTTester()
{
	Icon(BERTTesterImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("BERT on Text Data - MLM & NSP");

	// Initialize BERT task state
	current_task = 0; // Start with MLM

	// Define a BERT architecture using the existing transformer layers
	t = BuildBERTConfig();

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

	// Add BERT-specific controls
	Add(bert_ctrl.SizeHorz());
	CtrlLayout(bert_ctrl, "BERT Operations");
	bert_ctrl.Add(input_text1.VSizePos(0, 40).HSizePos());
	bert_ctrl.Add(input_text2.VSizePos(40, 80).HSizePos());
	bert_ctrl.Add(output_text.VSizePos(80, 120).HSizePos());

	// Setup buttons
	tokenize_btn.SetLabel("Tokenize");
	mask_btn.SetLabel("Mask Tokens");
	predict_btn.SetLabel("Predict");
	task_select.SetLabel("Task: MLM");

	// Add buttons to the control
	bert_ctrl.Add(tokenize_btn.VSizePos(120, 144).HSizePos(0, 200));
	bert_ctrl.Add(mask_btn.VSizePos(120, 144).HSizePos(210, 400));
	bert_ctrl.Add(predict_btn.VSizePos(120, 144).HSizePos(410, 600));
	bert_ctrl.Add(task_select.VSizePos(120, 144).HSizePos(610, 800));

	// Setup button callbacks
	tokenize_btn <<= THISBACK(OnTokenize);
	mask_btn <<= THISBACK(OnMask);
	predict_btn <<= THISBACK(OnPredict);
	task_select <<= THISBACK(OnTaskToggle);

	PostCallback(THISBACK(Refresher));
}

// BERT-specific functionality
void BERTTester::OnTaskToggle() {
	// Toggle between MLM (0) and NSP (1)
	current_task = 1 - current_task;  // Switch between 0 and 1

	if (current_task == 0) {
		task_select.SetLabel("Task: MLM");
	} else {
		task_select.SetLabel("Task: NSP");
	}

	// Update the BERT configuration based on the selected task
	t = BuildBERTConfig();
	net_edit.SetData(t);
	Reload();
}

void BERTTester::OnTokenize() {
	// In a real implementation, this would tokenize the input text
	// For now, just show a message
	String text1 = input_text1.GetText();
	String text2 = input_text2.GetText();

	output_text.SetText("Tokenized: [" + text1 + "] and [" + text2 + "]");
}

void BERTTester::OnMask() {
	// In a real implementation, this would mask tokens for MLM task
	// For now, just show a message
	String text1 = input_text1.GetText();

	// Simple demo: mask random words in the text
	Vector<String> words = Split(text1, ' ');
	String masked_text = "";
	for(int i = 0; i < words.GetCount(); i++) {
		if (Random(0, 10) < 2) {  // Mask ~20% of tokens
			masked_text += "[MASK] ";
		} else {
			masked_text += words[i] + " ";
		}
	}

	output_text.SetText("Masked: " + masked_text);
}

void BERTTester::OnPredict() {
	// In a real implementation, this would run the BERT model to make predictions
	// For now, just show a message indicating the task being performed
	if (current_task == 0) {
		// MLM task - predict masked tokens
		output_text.SetText("Running MLM prediction (would predict masked tokens with trained BERT model)");
	} else {
		// NSP task - predict if sentences are consecutive
		output_text.SetText("Running NSP prediction (would predict if sentences are consecutive with trained BERT model)");
	}
}

String BERTTester::BuildBERTConfig() {
	// Return BERT configuration based on the current task (MLM or NSP)
	if (current_task == 0) {
		// Configuration for MLM task
		return
			"[\n"
			"\t{\"type\":\"input\", \"input_width\":512, \"input_height\":1, \"input_depth\":768},\n"  // Input sequence of max length 512, embedding dim 768
			"\t{\"type\":\"embedding\", \"vocab_size\":30522, \"embed_dim\":768, \"max_pos_embeddings\":512, \"type_vocab_size\":2},\n"  // Token, position, and segment embeddings
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
			"\t{\"type\":\"masked_attention\", \"embed_dim\":768, \"num_heads\":12},\n"
			"\t{\"type\":\"layer_norm\"},\n"  // Layer normalization
			"\t{\"type\":\"fc\", \"neuron_count\":768, \"activation\":\"gelu\"},\n"  // Feed-forward layer
			"\t{\"type\":\"fc\", \"neuron_count\":30522, \"activation\":\"softmax\"},\n"  // Output for vocab size ~30K (MLM head)
			"\t{\"type\":\"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":16, \"l2_decay\":0.01}\n"
			"]\n";
	} else {
		// Configuration for NSP task
		return
			"[\n"
			"\t{\"type\":\"input\", \"input_width\":512, \"input_height\":1, \"input_depth\":768},\n"  // Input sequence of max length 512, embedding dim 768
			"\t{\"type\":\"embedding\", \"vocab_size\":30522, \"embed_dim\":768, \"max_pos_embeddings\":512, \"type_vocab_size\":2},\n"  // Token, position, and segment embeddings
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
			"\t{\"type\":\"layer_norm\"},\n"  // Layer normalization
			"\t{\"type\":\"fc\", \"neuron_count\":768, \"activation\":\"gelu\"},\n"  // Feed-forward layer
			"\t{\"type\":\"fc\", \"neuron_count\":2, \"activation\":\"softmax\"},\n"  // Output for binary classification (NSP head)
			"\t{\"type\":\"adam\", \"learning_rate\":0.0001, \"beta1\":0.9, \"beta2\":0.999, \"batch_size\":16, \"l2_decay\":0.01}\n"
			"]\n";
	}
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
	if (net_str != t) {  // Only reload if config has changed
		t = net_str;
	}

	ticking_lock.Enter();

	bool success = ses.MakeLayers(t);

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