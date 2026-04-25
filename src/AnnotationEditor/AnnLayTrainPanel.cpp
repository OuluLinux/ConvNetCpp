#include "AnnLayTrainPanel.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include <AnnLayCore/AnnMdl.h>
#include <AnnLayCore/GroupRegistry.h>
#include <AnnLayCore/AiTrainState.h>

#include <plugin/jpg/jpg.h>
#include <plugin/png/png.h>

NAMESPACE_UPP

namespace {

String TrimLowerPanel(const String& s) {
	return ToLower(TrimBoth(s));
}

void ShufflePairedPanel(Vector<Vector<double>>& samples, Vector<int>& labels) {
	ASSERT(samples.GetCount() == labels.GetCount());
	srand(42);
	for(int i = samples.GetCount() - 1; i > 0; i--) {
		int j = rand() % (i + 1);
		if(i != j) {
			Swap(samples[i], samples[j]);
			Swap(labels[i], labels[j]);
		}
	}
}

String MakeIsoTimestampPanel() {
	Time t = GetSysTime();
	return Format("%04d-%02d-%02dT%02d:%02d:%02d",
	              t.year, t.month, t.day, t.hour, t.minute, t.second);
}

}

AnnLayTrainPanel::AnnLayTrainPanel() {
	Add(edit_crops_dir.LeftPos(8, 520).TopPos(8, 24));
	Add(btn_browse_crops.LeftPos(536, 120).TopPos(8, 24));
	Add(edit_annlay_path.LeftPos(8, 520).TopPos(38, 24));
	Add(btn_browse_annlay.LeftPos(536, 120).TopPos(38, 24));

	Add(drop_slot.LeftPos(8, 220).TopPos(68, 24));
	Add(edit_epochs.LeftPos(236, 80).TopPos(68, 24));
	Add(btn_start.LeftPos(324, 90).TopPos(68, 24));
	Add(btn_stop.LeftPos(420, 90).TopPos(68, 24));
	Add(btn_save.LeftPos(516, 90).TopPos(68, 24));
	Add(lbl_status.LeftPos(8, 800).TopPos(98, 22));
	Add(h_split.HSizePos(8, 8).VSizePos(126, 8));
	h_split << graph << layer_view << pred_view;
	h_split.SetPos(4000);

	edit_crops_dir.SetText("");
	btn_browse_crops.SetLabel("Browse Crops");
	edit_annlay_path.SetText("");
	btn_browse_annlay.SetLabel("Browse AnnLay");
	drop_slot.Clear();
	edit_epochs.MinMax(1, 100000);
	edit_epochs.SetData(50);
	btn_start.SetLabel("Start");
	btn_stop.SetLabel("Stop");
	btn_save.SetLabel("Save");
	lbl_status.SetLabel("Ready.");

	btn_browse_crops << THISBACK(BrowseCropsDir);
	btn_browse_annlay << THISBACK(BrowseAnnLayPath);
	btn_start << THISBACK(StartTrain);
	btn_stop << [=] { StopTrain(true); };
	btn_save << THISBACK(SaveAnnLay);

	graph.SetSession(ses);
	graph.SetModeLoss();
	graph.SetInterval(1);
	graph.SetUpdateHz(5.0);
	graph.SetAverage(5);

	layer_view.SetSession(ses);
	layer_view.SetColor();
	pred_view.SetSession(ses);
	pred_view.SetAugmentation(0, false);

	PostCallback(THISBACK(Refresher));
}

AnnLayTrainPanel::~AnnLayTrainPanel() {
	StopTrain(true);
}

void AnnLayTrainPanel::ConfigureForStartup(const String& crops_dir, const String& annlay_file) {
	if(!crops_dir.IsEmpty())
		edit_crops_dir.SetData(crops_dir);
	if(!annlay_file.IsEmpty()) {
		edit_annlay_path.SetData(annlay_file);
		LoadAnnLay();
	}
}

void AnnLayTrainPanel::SetStatus(const String& s) {
	Mutex::Lock __(lock);
	last_message = s;
}

void AnnLayTrainPanel::BrowseCropsDir() {
	FileSel fs;
	if(fs.ExecuteSelectDir("Select crops root directory"))
		edit_crops_dir.SetData(fs.Get());
}

void AnnLayTrainPanel::BrowseAnnLayPath() {
	FileSel fs;
	fs.Type("AnnLay", "*.annlay");
	if(fs.ExecuteOpen("Select .annlay")) {
		edit_annlay_path.SetData(fs.Get());
		LoadAnnLay();
	}
}

Vector<String> AnnLayTrainPanel::ResolveTargetSlots(const String& slot_key) const {
	Vector<String> out;
	VectorMap<String, Vector<String>> groups = AnchoredSlotClassifier::GetSlotGroups(lay);
	int g = groups.Find(slot_key);
	if(g >= 0) {
		for(const String& id : groups[g]) {
			const AnnLaySlot* slot = lay.FindSlot(id);
			if(slot && (slot->method == ANNLAY_CLASSIFIER_LABEL || slot->method == ANNLAY_CLASSIFIER_BOOL))
				out.Add(id);
		}
	}
	else {
		for(const AnnLaySlot& slot : lay.slots) {
			if(slot.id == slot_key &&
			   (slot.method == ANNLAY_CLASSIFIER_LABEL || slot.method == ANNLAY_CLASSIFIER_BOOL)) {
				out.Add(slot.id);
				break;
			}
		}
	}
	return out;
}

bool AnnLayTrainPanel::LoadAnnLay() {
	String p = TrimBoth(edit_annlay_path.GetData());
	if(p.IsEmpty()) {
		SetStatus("Missing .annlay path.");
		return false;
	}
	AnnLay tmp;
	if(!tmp.Load(p)) {
		SetStatus("Failed to load .annlay: " + p);
		return false;
	}
	{
		Mutex::Lock __(lock);
		lay = pick(tmp);
		annlay_path = p;
		train_net_json.Clear();
	}
	ReloadSlots();
	SetStatus(Format("Loaded annlay: %s (%d slots)", GetFileName(p), lay.slots.GetCount()));
	return true;
}

void AnnLayTrainPanel::ReloadSlots() {
	String keep;
	int keep_i = drop_slot.GetIndex();
	if(keep_i >= 0)
		keep = drop_slot.GetKey(keep_i);
	drop_slot.Clear();
	VectorMap<String, Vector<String>> groups = AnchoredSlotClassifier::GetSlotGroups(lay);
	for(int g = 0; g < groups.GetCount(); g++) {
		Vector<String> ids;
		for(const String& id : groups[g]) {
			const AnnLaySlot* slot = lay.FindSlot(id);
			if(slot && (slot->method == ANNLAY_CLASSIFIER_LABEL || slot->method == ANNLAY_CLASSIFIER_BOOL))
				ids.Add(id);
		}
		if(ids.IsEmpty())
			continue;

		if(ids.GetCount() > 1) {
			String label = Format("Group (%d slots): ", ids.GetCount());
			int shown = min(3, ids.GetCount());
			for(int i = 0; i < shown; i++) {
				if(i) label << ", ";
				label << ids[i];
			}
			if(ids.GetCount() > shown)
				label << "...";
			drop_slot.Add(groups.GetKey(g), label);
		}
		else {
			drop_slot.Add(ids[0], ids[0]);
		}
	}
	int idx = drop_slot.FindKey(keep);
	if(idx >= 0)
		drop_slot.SetIndex(idx);
	else if(drop_slot.GetCount())
		drop_slot.SetIndex(0);
}

String AnnLayTrainPanel::BuildNetJson(int w, int h, int d, int cls_count) const {
	String s;
	s << "["
	  << "{\"type\":\"input\",\"input_width\":" << w
	  << ",\"input_height\":" << h
	  << ",\"input_depth\":" << d << "},"
	  << "{\"type\":\"conv\",\"width\":5,\"height\":5,\"filter_count\":8,\"stride\":1,\"pad\":2,\"activation\":\"relu\"},"
	  << "{\"type\":\"pool\",\"width\":2,\"height\":2,\"stride\":2},"
	  << "{\"type\":\"conv\",\"width\":5,\"height\":5,\"filter_count\":16,\"stride\":1,\"pad\":2,\"activation\":\"relu\"},"
	  << "{\"type\":\"pool\",\"width\":2,\"height\":2,\"stride\":2},"
	  << "{\"type\":\"softmax\",\"class_count\":" << cls_count << "},"
	  << "{\"type\":\"adadelta\",\"batch_size\":20,\"l2_decay\":0.001}"
	  << "]";
	return s;
}

Vector<double> AnnLayTrainPanel::ImageToSample(const Image& crop, Size crop_size) const {
	int len = crop_size.cx * crop_size.cy * 3;
	Vector<double> sample;
	sample.SetCount(len, 0.0);
	for(int y = 0; y < crop_size.cy; y++) {
		const RGBA* row = crop[y];
		for(int x = 0; x < crop_size.cx; x++) {
			int base = (y * crop_size.cx + x) * 3;
			sample[base + 0] = row[x].r / 255.0;
			sample[base + 1] = row[x].g / 255.0;
			sample[base + 2] = row[x].b / 255.0;
		}
	}
	return sample;
}

bool AnnLayTrainPanel::LoadCrops() {
	String crops_dir = TrimBoth(edit_crops_dir.GetData());
	if(crops_dir.IsEmpty()) {
		SetStatus("Missing crops directory.");
		return false;
	}
	if(!DirectoryExists(crops_dir)) {
		SetStatus("Crops directory does not exist: " + crops_dir);
		return false;
	}

	String slot_key;
	int slot_i = drop_slot.GetIndex();
	if(slot_i >= 0)
		slot_key = drop_slot.GetKey(slot_i);
	Vector<String> target_slots = ResolveTargetSlots(slot_key);
	if(target_slots.IsEmpty()) {
		SetStatus("No matching classifier slots to train.");
		return false;
	}

	const AnnLaySlot* slot = lay.FindSlot(target_slots[0]);
	if(!slot || slot->classes.IsEmpty()) {
		SetStatus("Selected slot has no class list.");
		return false;
	}

	VectorMap<String, int> class_map;
	for(int i = 0; i < slot->classes.GetCount(); i++)
		class_map.GetAdd(TrimLowerPanel(slot->classes[i])) = i;

	Vector<Vector<double>> samples;
	Vector<int> labels;
	int ignored_dirs = 0;

	FindFile ff;
	if(ff.Search(AppendFileName(crops_dir, "*"))) {
		do {
			if(!ff.IsFolder())
				continue;
			String dname = ff.GetName();
			if(dname == "." || dname == "..")
				continue;
			String cls_name = TrimLowerPanel(dname);
			int q = class_map.Find(cls_name);
			if(q < 0) {
				ignored_dirs++;
				continue;
			}
			int label = class_map[q];
			String subdir = AppendFileName(crops_dir, dname);
			FindFile imgf;
			if(imgf.Search(AppendFileName(subdir, "*"))) {
				do {
					if(!imgf.IsFile())
						continue;
					String ext = ToLower(GetFileExt(imgf.GetName()));
					if(ext != ".png" && ext != ".jpg" && ext != ".jpeg")
						continue;
					Image img = StreamRaster::LoadFileAny(imgf.GetPath());
					if(img.IsEmpty())
						continue;
					Image crop = Rescale(img, Size(32, 32));
					samples.Add(ImageToSample(crop, Size(32, 32)));
					labels.Add(label);
				}
				while(imgf.Next());
			}
		}
		while(ff.Next());
	}

	if(samples.IsEmpty()) {
		SetStatus("No valid crops found in: " + crops_dir);
		return false;
	}

	ShufflePairedPanel(samples, labels);

	{
		Mutex::Lock __(lock);
		loaded_samples = pick(samples);
		loaded_labels = pick(labels);
		trained_slot_ids = pick(target_slots);
		trained_slot_key = slot_key;
		train_done = false;
		trained_epochs = 0;
		last_loss = 0;
		last_val_acc = 0;
	}

	SetStatus(Format("Loaded %d samples from %s (%d ignored class dirs).",
	                 loaded_samples.GetCount(), crops_dir, ignored_dirs));
	return true;
}

void AnnLayTrainPanel::StartTrain() {
	{
		Mutex::Lock __(lock);
		if(train_running) {
			last_message = "Training already running.";
			return;
		}
	}

	if(!LoadAnnLay())
		return;

	String crops_dir = TrimBoth(edit_crops_dir.GetData());
	if(crops_dir.IsEmpty()) {
		SetStatus("Missing crops directory.");
		return;
	}
	if(!DirectoryExists(crops_dir)) {
		SetStatus("Crops directory does not exist: " + crops_dir);
		return;
	}

	String slot_key;
	int slot_i = drop_slot.GetIndex();
	if(slot_i >= 0)
		slot_key = drop_slot.GetKey(slot_i);
	Vector<String> target_slots = ResolveTargetSlots(slot_key);
	if(target_slots.IsEmpty()) {
		SetStatus("No matching classifier slots to train.");
		return;
	}
	String net_json;
	{
		const AnnLaySlot* ref_slot = lay.FindSlot(target_slots[0]);
		if(ref_slot) {
			int w = max(1, ref_slot->crop_size.cx);
			int h = max(1, ref_slot->crop_size.cy);
			int cls_count = max(2, ref_slot->classes.GetCount());
			net_json = BuildNetJson(w, h, 3, cls_count);
		}
	}

	int epochs = max(1, (int)edit_epochs.GetData());

	graph.Clear();

	{
		Mutex::Lock __(lock);
		train_crops_dir = crops_dir;
		trained_slot_key = slot_key;
		trained_slot_ids = pick(target_slots);
		train_net_json = net_json;
		target_epochs = epochs;
		stop_requested = false;
		train_running = true;
		train_done = false;
		trained_epochs = 0;
		last_loss = 0;
		last_val_acc = 0;
		loss_history.Clear();
		val_acc_history.Clear();
		last_message = Format("Training started from crops: %s", crops_dir);
	}

	train_thread.Run([=] { TrainingMain(); }, true);
	SetControlsEnabled();
}

void AnnLayTrainPanel::StopTrain(bool wait_thread) {
	bool running = false;
	{
		Mutex::Lock __(lock);
		if(train_running) {
			stop_requested = true;
			running = true;
		}
	}
	if(running && wait_thread) {
		train_thread.Wait();
		Mutex::Lock __(lock);
		train_running = false;
		if(last_message.IsEmpty())
			last_message = "Training stopped.";
	}
	SetControlsEnabled();
}

void AnnLayTrainPanel::TrainingMain() {
	int epochs;
	String crops_dir;
	String slot_key;
	{
		Mutex::Lock __(lock);
		epochs = target_epochs;
		crops_dir = train_crops_dir;
		slot_key = trained_slot_key;
	}

	TrainingJobRequest req;
	req.source_mode = TRAIN_SOURCE_CROPS;
	req.group_keys.Add(slot_key);
	req.crops_dir = crops_dir;
	req.max_epochs = epochs;
	req.balance_by_slot = true;
	req.slot_cap = 0;
	req.ignore_auto_stop = false;

	TrainingJobResult result;
	bool ok = AnchoredSlotClassifier::RunTrainingJob(
		lay, req, result,
		[this](String, int epoch, double loss, double acc) {
			Mutex::Lock __(lock);
			if(stop_requested)
				return;
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
		},
		&ses, nullptr);
	(void)result;

	{
		Mutex::Lock __(lock);
		train_running = false;
		train_done = ok;
		last_message = ok
		    ? Format("Training complete. epochs=%d loss=%.6f val_acc=%.4f", trained_epochs, last_loss, last_val_acc)
		    : String("Training failed or no crops found.");
	}
}

void AnnLayTrainPanel::SaveAnnLay() {
	StopTrain(true);

	bool ready = false;
	{
		Mutex::Lock __(lock);
		ready = train_done && !trained_slot_ids.IsEmpty();
	}
	if(!ready) {
		SetStatus("No completed training session to save.");
		return;
	}
	if(annlay_path.IsEmpty() || lay.slots.IsEmpty()) {
		SetStatus("No annlay loaded.");
		return;
	}

	if(!lay.Save(annlay_path)) {
		SetStatus("Failed to save annlay: " + annlay_path);
		return;
	}
	AnnMdl mdl;
	mdl.Load(annlay_path);
	String net_json;
	{
		Mutex::Lock __(lock);
		net_json = train_net_json;
	}
	if(net_json.IsEmpty()) {
		SetStatus("Saved annlay, but missing network json for annmdl.");
		return;
	}
	StringStream ss_weights;
	ses.SerializeWeights(ss_weights);
	String net_data = ss_weights.GetResult();

	StringStream ss_tdat;
	ses.SerializeTrainData(ss_tdat);
	String traindata_data = ss_tdat.GetResult();

	GroupRegistry reg;
	reg.Build(lay);
	String canonical_group = reg.Resolve(trained_slot_key);
	if(canonical_group.IsEmpty())
		canonical_group = trained_slot_key;

	AnnMdlEntry& e = mdl.GetOrAdd(canonical_group);
	e.slot_id = canonical_group;
	e.net_str = net_json;
	e.net_data = net_data;
	e.traindata_data = traindata_data;
	e.session_data.Clear();
	e.session_ref.Clear();

	Vector<String> aliases = reg.LegacyAliases(canonical_group);
	for(const String& alias : aliases) {
		AnnMdlEntry& le = mdl.GetOrAdd(alias);
		le.slot_id = alias;
		le.net_str = net_json;
		le.net_data = net_data;
		le.traindata_data = traindata_data;
		le.session_data.Clear();
		le.session_ref.Clear();
	}
	bool annmdl_ok = mdl.Save(annlay_path);

	AiTrainState state;
	{
		Mutex::Lock __(lock);
		state.group_key = trained_slot_key;
		state.slot_ids = clone(trained_slot_ids);
		state.epochs_completed = trained_epochs;
		state.target_epochs = target_epochs;
		state.timestamp = MakeIsoTimestampPanel();
		state.loss_history = clone(loss_history);
		state.val_acc_history = clone(val_acc_history);
	}
	bool sidecar_ok = state.Save(annlay_path);

	int count = 0;
	{
		Mutex::Lock __(lock);
		count = trained_slot_ids.GetCount();
	}
	SetStatus(Format("Saved annlay%s%s for %d slot(s): %s",
	                 annmdl_ok ? "+annmdl" : " (annmdl failed)",
	                 sidecar_ok ? "+trainstate" : " (trainstate failed)",
	                 count, annlay_path));
}

void AnnLayTrainPanel::SetControlsEnabled() {
	bool running;
	{
		Mutex::Lock __(lock);
		running = train_running;
	}
	btn_start.Enable(!running);
	btn_stop.Enable(running);
	btn_save.Enable(!running);
}

void AnnLayTrainPanel::Refresher() {
	graph.RefreshData();
	layer_view.Refresh();
	pred_view.Refresh();

	String msg;
	bool running;
	int epochs;
	double loss;
	double acc;
	{
		Mutex::Lock __(lock);
		msg = last_message;
		running = train_running;
		epochs = trained_epochs;
		loss = last_loss;
		acc = last_val_acc;
	}
	lbl_status.SetLabel(Format("%s | epoch=%d loss=%.6f val_acc=%.4f%s",
	                           msg, epochs, loss, acc, running ? " [running]" : ""));
	SetControlsEnabled();
	SetTimeCallback(250, THISBACK(Refresher));
}

END_UPP_NAMESPACE
