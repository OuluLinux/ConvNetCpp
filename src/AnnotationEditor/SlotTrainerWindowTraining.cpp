#include "SlotTrainerWindow.h"
#include "AnnotationEditorCommon.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include <AnnLayCore/GroupRegistry.h>
#include <AnnLayCore/AiTrainState.h>

NAMESPACE_UPP

void SlotTrainerWindow::StopTrain(bool wait_thread) {
	if(train_running.load()) {
		stop_requested.store(true);
		ses.StopTraining();
		if(wait_thread) {
			train_thread.Wait();
			train_running.store(false);
			if(!closing.load() && !train_done)
				OnTrainingStoppedPrompt();
		}
	}
	SetControlsEnabled();
}

void SlotTrainerWindow::TrainingMain() {
	ASSERT(!IsMainThread());
	int epochs;
	String crops_dir;
	String annprj_path;
	String images_dir;
	String slot_key;
	bool is_bool = false;
	bool auto_stop = true;
	{
		Mutex::Lock __(lock);
		epochs = target_epochs;
		crops_dir = train_crops_dir;
		annprj_path = train_annprj_path;
		images_dir = train_images_dir;
		slot_key = trained_slot_key;
		is_bool = train_is_bool;
		auto_stop = train_auto_stop;
	}
	Cout() << "[SlotTrainer] TrainingMain: begin is_bool=" << (is_bool ? "true" : "false")
	       << " slot_key='" << slot_key
	       << "' epochs=" << epochs
	       << " crops='" << crops_dir
	       << "' annprj='" << annprj_path
	       << "' images='" << images_dir
	       << "' auto_stop=" << (auto_stop ? "true" : "false") << "\n";

	auto_stop_triggered.store(false);

	bool ok = false;
	bool ignore_auto = !auto_stop;

	while(true) {
		TrainingJobRequest req;
		req.source_mode = is_bool ? TRAIN_SOURCE_ANNPRJ_BOOL : TRAIN_SOURCE_CROPS;
		req.group_keys.Add(slot_key);
		req.crops_dir = crops_dir;
		req.annprj_path = annprj_path;
		req.images_dir = images_dir;
		req.max_epochs = epochs;
		req.balance_by_slot = true;
		req.slot_cap = 0;
		req.ignore_auto_stop = ignore_auto;
		req.debug = !is_bool; // preserve previous verbose debug for crops path

		AnnMdl mdl;
		mdl.Load(annlay_path);

		TrainingJobResult result;
			ok = AnchoredSlotClassifier::RunTrainingJob(
				lay, req, result,
				[this, is_bool](String, int epoch, double loss, double acc) {
				if(stop_requested.load()) {
					return;
				}
				Mutex::Lock __(lock);
				trained_epochs = epoch;
				last_loss = loss;
				last_val_acc = acc;
				int idx = max(0, epoch - 1);
				while(loss_history.GetCount() <= idx) {
					loss_history.Add(loss);
					val_acc_history.Add(acc);
				}
				loss_history[idx] = loss;
				val_acc_history[idx] = acc;
				if(is_bool || IsFin(loss))
					graph.PostAddValue(loss);
					if(train_is_bool)
						ComputeBoolMetrics();
				},
				&ses, &mdl);
			(void)result;

		{
			Mutex::Lock __(lock);
			train_mdl = pick(mdl);
		}

		bool stopped = stop_requested.load();
		int cur_trained_epochs = 0;
		{
			Mutex::Lock __(lock);
			cur_trained_epochs = trained_epochs;
		}

		if(ok && !stopped && cur_trained_epochs > 0 && cur_trained_epochs < epochs && !ignore_auto) {
			// Auto-stop occurred. Prompt user.
			continue_decision.store(0);
			PostCallback([this, cur_trained_epochs] {
				if(PromptYesNo(Format("Training seems done at epoch %d. Keep training regardless?", cur_trained_epochs)))
					continue_decision.store(1);
				else
					continue_decision.store(2);
			});

			while(continue_decision.load() == 0 && !closing.load()) {
				Sleep(50);
			}

			if(continue_decision.load() == 1) {
				ignore_auto = true;
				continue; // Resume
			}
		}
		break;
	}

	Cout() << "[SlotTrainer] TrainingMain: train loop finished ok=" << (ok ? "true" : "false") << "\n";

	bool stopped = stop_requested.load();
	bool completed = false;
	bool early_stopped = false;
	int epochs_done = 0;
	{
		Mutex::Lock __(lock);
		train_done = ok;
		epochs_done = trained_epochs;
		completed = ok && trained_epochs >= target_epochs && !stopped;
		early_stopped = ok && !stopped && trained_epochs > 0 && trained_epochs < target_epochs;

		if(completed)
			last_message = Format("Training complete. epochs=%d loss=%.6f val_acc=%.4f", trained_epochs, last_loss, last_val_acc);
		else if(early_stopped)
			last_message = Format("Training early-stopped (success). epochs=%d loss=%.6f val_acc=%.4f", trained_epochs, last_loss, last_val_acc);
		else if(stopped)
			last_message = Format("Training stopped at epoch %d/%d.", trained_epochs, target_epochs);
		else
			last_message = "Training failed or no crops found.";
	}
	train_running.store(false);
	Cout() << "[SlotTrainer] TrainingMain: done stopped=" << (stopped ? "true" : "false")
	       << " completed=" << (completed ? "true" : "false")
	       << " early_stopped=" << (early_stopped ? "true" : "false")
	       << " epochs_done=" << epochs_done << "\n";
	if(closing.load())
		return;
	if(completed || early_stopped) {
		PostCallback(THISBACK(OnTrainingCompletedPrompt));
	}
	else if(stopped && epochs_done > 0) {
		PostCallback(THISBACK(OnTrainingStoppedPrompt));
	}
}

void SlotTrainerWindow::ComputeBoolMetrics() {
	if(!train_is_bool) return;
	
	ConvNet::SessionData& sd = ses.Data();
	int n = sd.GetTestCount();
	if(n == 0) return;

	BoolGateMetrics m;
	m.samples = n;
	
	ses.SetTestPredict(true);
	for(int i = 0; i < n; i++) {
		Vector<double> res = ses.Predict(sd.GetTest(i));
		int pred = 0;
		if(res.GetCount() >= 2) {
			// Support both softmax (probs) and SVM-like outputs
			pred = (res[1] > res[0]) ? 1 : 0;
		}
		int gt = sd.GetTestLabel(i);
		
		if(gt == 1) {
			m.gt_pos++;
			if(pred == 1) m.tp++;
			else m.fn++;
		}
		else {
			if(pred == 1) m.fp++;
			else m.tn++;
		}
	}
	m.Calc();
	
	{
		Mutex::Lock __(lock);
		metrics = m;
	}
}

void SlotTrainerWindow::DoSave() {
	if(annlay_path.IsEmpty()) {
		SetStatus("No annlay loaded.");
		return;
	}

	AnnMdl mdl;
	String group_key;
	Vector<String> slot_ids;
	int epochs_done = 0;
	int target_ep = 0;
	double loss = 0, val_acc = 0;
	Vector<double> loss_hist, val_acc_hist;
	BoolGateMetrics m;
	{
		Mutex::Lock __(lock);
		mdl         = pick(train_mdl);
		group_key   = train_group_key;
		slot_ids    = clone(trained_slot_ids);
		epochs_done = trained_epochs;
		target_ep   = target_epochs;
		loss        = last_loss;
		val_acc     = last_val_acc;
		loss_hist   = clone(loss_history);
		val_acc_hist = clone(val_acc_history);
		m           = metrics;
	}

	if(!lay.Save(annlay_path)) {
		SetStatus("Failed to save annlay.");
		return;
	}
	if(!mdl.Save(annlay_path)) {
		SetStatus("Failed to save annmdl.");
		return;
	}

	AiTrainState state;
	state.group_key        = group_key;
	state.slot_ids         = pick(slot_ids);
	state.epochs_completed = epochs_done;
	state.target_epochs    = target_ep;
	state.timestamp        = MakeIsoTimestamp();
	state.loss_history     = pick(loss_hist);
	state.val_acc_history  = pick(val_acc_hist);
	state.last_precision   = m.precision;
	state.last_recall      = m.recall;
	state.last_f1          = m.f1;
	bool sidecar_ok = state.Save(annlay_path);

	SetStatus(Format("Saved. epochs=%d loss=%.4f val_acc=%.4f%s",
	                 epochs_done, loss, val_acc,
	                 sidecar_ok ? "" : " (sidecar save failed)"));
}

void SlotTrainerWindow::OnTrainingCompletedPrompt() {
	if(closing.load())
		return;
	RehookSession();
	int epochs = 0;
	double loss = 0;
	double acc = 0;
	{
		Mutex::Lock __(lock);
		epochs = trained_epochs;
		loss = last_loss;
		acc = last_val_acc;
	}
	String msg = Format("Training complete (%d epochs, loss=%.6f, val_acc=%.4f).\nSave model to annlay+annmdl?",
	                    epochs, loss, acc);
	if(PromptYesNo(msg))
		DoSave();
	else
		SetStatus("Training complete. Not saved.");
}

void SlotTrainerWindow::OnTrainingStoppedPrompt() {
	if(closing.load())
		return;
	RehookSession();
	int epochs = 0;
	int target = 0;
	{
		Mutex::Lock __(lock);
		epochs = trained_epochs;
		target = target_epochs;
	}
	String msg = Format("Training stopped at epoch %d/%d.\nSave partial model?", epochs, target);
	if(PromptYesNo(msg))
		DoSave();
	else
		SetStatus("Training stopped. Partial weights not saved.");
}

void SlotTrainerWindow::SetControlsEnabled() {
	bool running = train_running.load();
	btn_start.Enable(!running);
	btn_stop.Enable(running);
}

void SlotTrainerWindow::Refresher() {
	ASSERT(IsMainThread());
	graph.RefreshData();
	layer_view.Refresh();
	pred_view.Refresh();

	String msg;
	int epochs;
	double loss;
	double acc;
	bool running = train_running.load();
	{
		Mutex::Lock __(lock);
		msg = last_message;
		epochs = trained_epochs;
		loss = last_loss;
		acc = last_val_acc;
	}
	lbl_status.SetLabel(Format("%s\nEpoch %d | Loss %.6f | Val.Acc %.4f%s",
	                           msg, epochs, loss, acc, running ? " [running]" : ""));
	
	if(train_is_bool) {
		BoolGateMetrics m;
		{
			Mutex::Lock __(lock);
			m = metrics;
		}
		if(m.samples > 0) {
			lbl_metrics.SetLabel(Format("Validation Metrics:\nPrec: %.4f | Rec: %.4f | F1: %.4f",
			                            m.precision, m.recall, m.f1));
		}
		else {
			lbl_metrics.SetLabel("");
		}
	}
	else {
		lbl_metrics.SetLabel("");
	}
	SetControlsEnabled();
	SetTimeCallback(250, THISBACK(Refresher));
}

END_UPP_NAMESPACE
