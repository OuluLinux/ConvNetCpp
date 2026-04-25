#include "SlotTrainerWindow.h"
#include "AnnotationEditorCommon.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include <AnnLayCore/GroupRegistry.h>
#include <AnnLayCore/AiTrainState.h>

NAMESPACE_UPP

namespace {

bool IsImageExt(const String& ext) {
	String e = ToLower(ext);
	return e == ".png" || e == ".jpg" || e == ".jpeg";
}

bool DirHasImages(const String& dir) {
	FindFile ff;
	if(!ff.Search(AppendFileName(dir, "*")))
		return false;
	do {
		if(ff.IsFile() && IsImageExt(GetFileExt(ff.GetName())))
			return true;
	}
	while(ff.Next());
	return false;
}

// Class-root layout: immediate subdirs are class names and contain image files.
bool LooksLikeClassRoot(const String& dir) {
	if(!DirectoryExists(dir))
		return false;
	FindFile ff;
	if(!ff.Search(AppendFileName(dir, "*")))
		return false;
	do {
		if(!ff.IsFolder())
			continue;
		String name = ff.GetName();
		if(name == "." || name == "..")
			continue;
		if(DirHasImages(ff.GetPath()))
			return true;
	}
	while(ff.Next());
	return false;
}

}

SlotTrainerWindow::SlotTrainerWindow() {
	Title("Slot Trainer");
	Sizeable().Zoomable();
	SetRect(0, 0, 1400, 900);
	AddFrame(menu_bar);
	menu_bar.Set(THISBACK(MainMenu));
	Add(v_split.SizePos());
	v_split << layer_view << pred_view;
	v_split.Vert();
	v_split.SetPos(6400);

	settings_panel.Add(lbl_crops_dir.SetLabel("Crops directory").HSizePos(8, 8).TopPos(6, 20));
	settings_panel.Add(edit_crops_dir.LeftPos(8, 228).TopPos(28, 24));
	settings_panel.Add(btn_browse_crops.LeftPos(240, 70).TopPos(28, 24));
	settings_panel.Add(lbl_annprj.SetLabel("AnnPrj path").HSizePos(8, 8).TopPos(58, 20));
	settings_panel.Add(edit_annprj_path.LeftPos(8, 228).TopPos(80, 24));
	settings_panel.Add(btn_browse_annprj.LeftPos(240, 70).TopPos(80, 24));
	settings_panel.Add(lbl_images_dir.SetLabel("Images dir").HSizePos(8, 8).TopPos(110, 20));
	settings_panel.Add(edit_images_dir.LeftPos(8, 228).TopPos(132, 24));
	settings_panel.Add(btn_browse_images.LeftPos(240, 70).TopPos(132, 24));
	settings_panel.Add(lbl_annlay.SetLabel("AnnLay path").HSizePos(8, 8).TopPos(162, 20));
	settings_panel.Add(edit_annlay_path.LeftPos(8, 228).TopPos(184, 24));
	settings_panel.Add(btn_browse_annlay.LeftPos(240, 70).TopPos(184, 24));
	settings_panel.Add(lbl_slot.SetLabel("Target slot").HSizePos(8, 8).TopPos(214, 20));
	settings_panel.Add(drop_slot.LeftPos(8, 302).TopPos(236, 24));
	settings_panel.Add(lbl_epochs.SetLabel("Epochs").HSizePos(8, 8).TopPos(266, 20));
	settings_panel.Add(edit_epochs.LeftPos(8, 80).TopPos(288, 24));
	settings_panel.Add(opt_auto_stop.SetLabel("Stop when done").LeftPos(100, 200).TopPos(288, 24));
	settings_panel.Add(btn_start.LeftPos(8, 94).TopPos(322, 24));
	settings_panel.Add(btn_stop.LeftPos(108, 94).TopPos(322, 24));
	settings_panel.Add(lbl_status.LeftPos(8, 302).TopPos(356, 80));
	settings_panel.Add(lbl_metrics.LeftPos(8, 302).TopPos(440, 60));

	edit_crops_dir.SetText("");
	btn_browse_crops.SetLabel("Browse Crops");
	edit_annprj_path.SetText("");
	btn_browse_annprj.SetLabel("Browse AnnPrj");
	edit_images_dir.SetText("");
	btn_browse_images.SetLabel("Browse Images");
	edit_annlay_path.SetText("");
	btn_browse_annlay.SetLabel("Browse AnnLay");
	drop_slot.Clear();
	edit_epochs.MinMax(1, 100000);
	edit_epochs.SetData(50);
	opt_auto_stop.SetData(true);
	btn_start.SetLabel("Start");
	btn_stop.SetLabel("Stop");
	lbl_status.SetLabel("Ready.");
	lbl_metrics.SetLabel("");
	lbl_metrics.SetInk(LtBlue());
	lbl_metrics.SetFont(StdFont().Bold());

	btn_browse_crops << THISBACK(BrowseCropsDir);
	btn_browse_annprj << THISBACK(BrowseAnnprjPath);
	btn_browse_images << THISBACK(BrowseImagesDir);
	btn_browse_annlay << THISBACK(BrowseAnnLayPath);
	btn_start << THISBACK(StartTrain);
	btn_stop << [=] { StopTrain(false); };
	drop_slot << THISBACK(OnSlotSelectionChanged);

	graph.SetSession(ses);
	graph.SetModeLoss();
	graph.SetInterval(1);
	graph.SetUpdateHz(5.0);
	graph.SetAverage(5);

	layer_view.SetSession(ses);
	layer_view.SetColor();
	pred_view.SetSession(ses);
	pred_view.SetAugmentation(0, false);

	UpdateTrainingModeUI();
	SetControlsEnabled();
	PostCallback(THISBACK(Refresher));
}

SlotTrainerWindow::~SlotTrainerWindow() {
	closing.store(true);
	StopTrain(true);
}

void SlotTrainerWindow::RehookSession() {
	ASSERT(IsMainThread());
	ASSERT(!train_running.load());
	graph.SetSession(ses);
	layer_view.SetSession(ses);
	pred_view.SetSession(ses);
	ses.SetTestPredict(true);
}

void SlotTrainerWindow::DockInit() {
	DockLeft(Dockable(settings_panel, "Controls").SizeHint(Size(320, 500)));
	DockLeft(Dockable(graph, "Loss").SizeHint(Size(320, 240)));
}

void SlotTrainerWindow::MainMenu(Bar& bar) {
	bar.Sub("File", [=](Bar& b) {
		b.Add("Exit", [=] { Break(); });
	});
	bar.Sub("Training", [=](Bar& b) {
		b.Add("Start", THISBACK(StartTrain));
		b.Add("Stop", [=] { StopTrain(false); });
	});
}

void SlotTrainerWindow::Configure(const String& annlay,
                                  const String& crops_dir,
                                  const String& slot_key,
                                  const String& annprj,
                                  const String& images_dir) {
	if(!annlay.IsEmpty()) {
		edit_annlay_path.SetData(annlay);
		LoadAnnLay();
	}
	if(!crops_dir.IsEmpty())
		edit_crops_dir.SetData(crops_dir);
	if(!annprj.IsEmpty())
		edit_annprj_path.SetData(annprj);
	if(!images_dir.IsEmpty())
		edit_images_dir.SetData(images_dir);
	else if(!annprj.IsEmpty()) {
		String def_images = AppendFileName(GetFileDirectory(annprj), "images");
		if(DirectoryExists(def_images))
			edit_images_dir.SetData(def_images);
	}
	if(!slot_key.IsEmpty()) {
		int idx = drop_slot.FindKey(slot_key);
		if(idx >= 0)
			drop_slot.SetIndex(idx);
	}
	OnSlotSelectionChanged();
}

void SlotTrainerWindow::SetStatus(const String& s) {
	Mutex::Lock __(lock);
	last_message = s;
}

void SlotTrainerWindow::BrowseCropsDir() {
	FileSel fs;
	if(fs.ExecuteSelectDir("Select crops root directory"))
		edit_crops_dir.SetData(fs.Get());
}

void SlotTrainerWindow::BrowseAnnprjPath() {
	FileSel fs;
	fs.Type("AnnPrj", "*.annprj");
	if(fs.ExecuteOpen("Select .annprj")) {
		String p = fs.Get();
		edit_annprj_path.SetData(p);
		String def_images = AppendFileName(GetFileDirectory(p), "images");
		if(DirectoryExists(def_images))
			edit_images_dir.SetData(def_images);
	}
}

void SlotTrainerWindow::BrowseImagesDir() {
	FileSel fs;
	if(fs.ExecuteSelectDir("Select images directory"))
		edit_images_dir.SetData(fs.Get());
}

void SlotTrainerWindow::BrowseAnnLayPath() {
	FileSel fs;
	fs.Type("AnnLay", "*.annlay");
	if(fs.ExecuteOpen("Select .annlay")) {
		edit_annlay_path.SetData(fs.Get());
		LoadAnnLay();
	}
}

bool SlotTrainerWindow::IsCurrentGroupBool() const {
	int i = drop_slot.GetIndex();
	if(i < 0)
		return false;
	String key = drop_slot.GetKey(i);
	GroupRegistry reg;
	reg.Build(lay);
	if(reg.IsBoolGroup(key))
		return true;
	Vector<String> ids = ResolveTargetSlots(key);
	for(const String& id : ids) {
		const AnnLaySlot* s = lay.FindSlot(id);
		if(s)
			return s->method == ANNLAY_CLASSIFIER_BOOL;
	}
	return false;
}

void SlotTrainerWindow::UpdateTrainingModeUI() {
	bool is_bool = IsCurrentGroupBool();
	lbl_crops_dir.Enable(!is_bool);
	edit_crops_dir.Enable(!is_bool);
	btn_browse_crops.Enable(!is_bool);
	lbl_annprj.Enable(is_bool);
	edit_annprj_path.Enable(is_bool);
	btn_browse_annprj.Enable(is_bool);
	lbl_images_dir.Enable(is_bool);
	edit_images_dir.Enable(is_bool);
	btn_browse_images.Enable(is_bool);
}

void SlotTrainerWindow::RefreshPreviewForSelection() {
	String slot_key;
	int i = drop_slot.GetIndex();
	if(i >= 0)
		slot_key = drop_slot.GetKey(i);

	Vector<String> slot_ids = ResolveTargetSlots(slot_key);
	RestoreSessionFromWeights(slot_ids, slot_key);
	
	// Load initial preview data
	String ap = edit_annprj_path.GetData();
	String idir = edit_images_dir.GetData();
	if(IsCurrentGroupBool()) {
		Vector<Vector<double>> true_samples, false_samples;
		for(const String& id : slot_ids) {
			const AnnLaySlot* s = lay.FindSlot(id);
			if(s) AnchoredSlotClassifier::BuildBoolDataset(*s, lay, ap, idir, true_samples, false_samples);
		}
		
		int train_n = true_samples.GetCount() + false_samples.GetCount();
		if(train_n > 0) {
			const AnnLaySlot* ref = lay.FindSlot(slot_ids[0]);
			int w = ref->crop_size.cx;
			int h = ref->crop_size.cy;
			int d = (ref->color_mode == "color" ? 3 : 1);
			ses.Data().BeginData(2, train_n, w, h, d, 0);
			int idx = 0;
			for(const auto& s : true_samples) { ses.Data().Get(idx) <<= s; ses.Data().SetLabel(idx++, 1); }
			for(const auto& s : false_samples) { ses.Data().Get(idx) <<= s; ses.Data().SetLabel(idx++, 0); }
			ses.Data().EndData();
		}
	} else {
		VectorMap<int, Vector<Vector<double>>> class_samples;
		const AnnLaySlot* ref = lay.FindSlot(slot_ids[0]);
		if(ref) {
			Vector<Vector<double>> samples;
			Vector<int> labels;
			AnchoredSlotClassifier::BuildLabelDataset(*ref, lay, ap, idir, String(), samples, labels);
			
			if(!samples.IsEmpty()) {
				int w = ref->crop_size.cx;
				int h = ref->crop_size.cy;
				int d = (ref->color_mode == "color" ? 3 : 1);
				int cls_count = ref->classes.GetCount();
				ses.Data().BeginData(cls_count, samples.GetCount(), w, h, d, 0);
				for(int cls = 0; cls < cls_count; cls++) ses.Data().SetClass(cls, ref->classes[cls]);
				for(int k = 0; k < samples.GetCount(); k++) {
					ses.Data().Get(k) <<= samples[k];
					ses.Data().SetLabel(k, labels[k]);
				}
				ses.Data().EndData();
			}
		}
	}

	layer_view.RefreshLayers();
	pred_view.Clear();
	if(ses.GetNetwork().GetLayers().GetCount() > 0 && ses.Data().GetDataCount() > 0)
		pred_view.RefreshData();

	graph.Clear();
	if(sidecar_loaded && slot_key == sidecar_group_key) {
		for(double v : sidecar_loss_history)
			graph.AddValue(v);
		SetStatus(Format("Previous run: epochs=%d/%d loss=%.4f val_acc=%.4f  [%s]",
		                 sidecar_epochs_completed,
		                 max(0, sidecar_target_epochs),
		                 sidecar_loss_history.IsEmpty() ? 0.0 : sidecar_loss_history.Top(),
		                 sidecar_val_acc_history.IsEmpty() ? 0.0 : sidecar_val_acc_history.Top(),
		                 sidecar_timestamp));
		if(train_is_bool && sidecar_epochs_completed > 0) {
			lbl_metrics.SetLabel(Format("Validation Metrics:\nPrec: %.4f | Rec: %.4f | F1: %.4f",
			                            sidecar_last_precision, sidecar_last_recall, sidecar_last_f1));
		}
	}
	else if(!slot_ids.IsEmpty()) {
		if(AnySlotHasWeights(slot_ids, slot_key))
			SetStatus("Loaded existing model for selected slot group.");
		else
			SetStatus("Selected slot group has no saved model yet.");
	}
}

void SlotTrainerWindow::OnSlotSelectionChanged() {
	UpdateTrainingModeUI();
	RefreshPreviewForSelection();
}

Vector<String> SlotTrainerWindow::ResolveTargetSlots(const String& slot_key) const {
	Vector<String> out;
	String base_key = slot_key;
	int hash = base_key.Find('#');
	if(hash >= 0)
		base_key = base_key.Left(hash);

	VectorMap<String, Vector<String>> groups = AnchoredSlotClassifier::GetSlotGroups(lay);

	if(out.IsEmpty()) {
		int g = groups.Find(slot_key); // Use full key as fallback
		if(g >= 0) {
			for(const String& id : groups[g]) {
				const AnnLaySlot* slot = lay.FindSlot(id);
				if(slot && (slot->method == ANNLAY_CLASSIFIER_LABEL || slot->method == ANNLAY_CLASSIFIER_BOOL))
					out.Add(id);
			}
		}
		else {
			for(const AnnLaySlot& slot : lay.slots) {
				if(slot.id == base_key &&
				   (slot.method == ANNLAY_CLASSIFIER_LABEL || slot.method == ANNLAY_CLASSIFIER_BOOL)) {
					out.Add(slot.id);
					break;
				}
			}
		}
	}
	return out;
}
bool SlotTrainerWindow::LoadAnnLay() {
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
//...
		train_group_key.Clear();
		loss_history.Clear();
		val_acc_history.Clear();
		train_done = false;
		trained_epochs = 0;
		target_epochs = max(1, (int)edit_epochs.GetData());
		last_loss = 0;
		last_val_acc = 0;
	}
	ReloadSlots();

	ConvNet::AiTrainState state;
	sidecar_loaded = state.Load(p);
	if(sidecar_loaded) {
		sidecar_group_key = state.group_key;
		sidecar_slot_ids = clone(state.slot_ids);
		sidecar_loss_history = clone(state.loss_history);
		sidecar_val_acc_history = clone(state.val_acc_history);
		sidecar_epochs_completed = state.epochs_completed;
		sidecar_target_epochs = state.target_epochs;
		sidecar_last_precision = state.last_precision;
		sidecar_last_recall = state.last_recall;
		sidecar_last_f1 = state.last_f1;
		sidecar_timestamp = state.timestamp;

		{
			Mutex::Lock __(lock);
			train_group_key = sidecar_group_key;
			trained_slot_ids = clone(sidecar_slot_ids);
			loss_history = clone(sidecar_loss_history);
			val_acc_history = clone(sidecar_val_acc_history);
			trained_epochs = sidecar_epochs_completed;
			last_loss = loss_history.IsEmpty() ? 0.0 : loss_history.Top();
			last_val_acc = val_acc_history.IsEmpty() ? 0.0 : val_acc_history.Top();
		}
	}
	else {
		sidecar_group_key.Clear();
		sidecar_slot_ids.Clear();
		sidecar_loss_history.Clear();
		sidecar_val_acc_history.Clear();
		sidecar_epochs_completed = 0;
		sidecar_target_epochs = 0;
		sidecar_timestamp.Clear();
	}

	OnSlotSelectionChanged();
	if(!sidecar_loaded)
		SetStatus(Format("Loaded annlay: %s (%d slots)", GetFileName(p), lay.slots.GetCount()));
	return true;
}

void SlotTrainerWindow::ReloadSlots() {
	String keep;
	int keep_i = drop_slot.GetIndex();
	if(keep_i >= 0)
		keep = drop_slot.GetKey(keep_i);
	drop_slot.Clear();
	VectorMap<String, Vector<String>> groups = AnchoredSlotClassifier::GetSlotGroups(lay);
	for(int g = 0; g < groups.GetCount(); g++) {
		const Vector<String>& ids = groups[g];
		if(ids.IsEmpty())
			continue;

		String display = AnchoredSlotClassifier::GetSlotGroupDisplayName(lay, groups.GetKey(g));
		if(ids.GetCount() > 1) {
			String label = display + Format(" (%d slots): ", ids.GetCount());
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
			drop_slot.Add(groups.GetKey(g), display);
		}
	}
	int idx = drop_slot.FindKey(keep);
	if(idx >= 0)
		drop_slot.SetIndex(idx);
	else if(drop_slot.GetCount())
		drop_slot.SetIndex(0);
	UpdateTrainingModeUI();
}

String SlotTrainerWindow::BuildNetJson(int w, int h, int d, int cls_count) const {
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

Vector<double> SlotTrainerWindow::ImageToSample(const Image& crop, Size crop_size) const {
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

bool SlotTrainerWindow::AnySlotHasWeights(const Vector<String>& slot_ids, const String& slot_key) const {
	if(annlay_path.IsEmpty())
		return false;
	AnnMdl mdl;
	if(!mdl.Load(annlay_path))
		return false;
	
	GroupRegistry reg;
	reg.Build(lay);
	String canonical = reg.Resolve(slot_key);
	String suffix = reg.HeadSuffixForGroup(canonical);

	for(const String& id : slot_ids) {
		const AnnMdlEntry* e = mdl.FindEntry(id + suffix);
		if(e && !e->net_str.IsEmpty())
			return true;
	}
	if(!canonical.IsEmpty()) {
		const AnnMdlEntry* e = mdl.FindEntry(canonical);
		if(e && !e->net_str.IsEmpty())
			return true;
	}
	return false;
}

void SlotTrainerWindow::RestoreSessionFromWeights(const Vector<String>& slot_ids, const String& slot_key) {
	ASSERT(IsMainThread());
	ASSERT(!train_running.load());
	ses.Clear();
	if(annlay_path.IsEmpty())
		return;
	AnnMdl mdl;
	if(!mdl.Load(annlay_path))
		return;

	GroupRegistry reg;
	reg.Build(lay);
	String canonical = reg.Resolve(slot_key);
	String suffix    = reg.HeadSuffixForGroup(slot_key);

	// Try canonical group key first, then per-slot fallback
	bool loaded = !canonical.IsEmpty() && AnnMdl::LoadIntoSession(mdl, canonical, ses);
	if(!loaded) {
		for(const String& id : slot_ids) {
			if(AnnMdl::LoadIntoSession(mdl, id + suffix, ses)) {
				loaded = true;
				break;
			}
		}
	}

	if(loaded) {
		ses.SetTestPredict(true);
		ConvNet::LayerBase* in = ses.GetInput();
		if(in && in->input_width > 0 && in->input_height > 0 && in->input_depth > 0) {
			Vector<double> sample;
			sample.SetCount(in->input_width * in->input_height * in->input_depth, 0.0);
			ses.Predict(sample);
		}
	}
}

void SlotTrainerWindow::StartTrain() {
	ASSERT(IsMainThread());
	Cout() << "[SlotTrainer] StartTrain: begin\n";
	if(train_running.load()) {
		Cout() << "[SlotTrainer] StartTrain: rejected, training already running\n";
		SetStatus("Training already running.");
		return;
	}

	String selected_slot_key;
	int selected_i = drop_slot.GetIndex();
	if(selected_i >= 0)
		selected_slot_key = drop_slot.GetKey(selected_i);

	if(!LoadAnnLay())
		return;
	if(!selected_slot_key.IsEmpty()) {
		int idx = drop_slot.FindKey(selected_slot_key);
		if(idx >= 0)
			drop_slot.SetIndex(idx);
	}

	String slot_key;
	int slot_i = drop_slot.GetIndex();
	if(slot_i >= 0)
		slot_key = drop_slot.GetKey(slot_i);
	Vector<String> target_slots = ResolveTargetSlots(slot_key);
	Cout() << "[SlotTrainer] StartTrain: slot_key='" << slot_key << "' targets=" << target_slots.GetCount() << "\n";
	for(int i = 0; i < target_slots.GetCount(); i++)
		Cout() << "[SlotTrainer]   target[" << i << "]=" << target_slots[i] << "\n";
	if(target_slots.IsEmpty()) {
		Cout() << "[SlotTrainer] StartTrain: no target slots\n";
		SetStatus("No matching classifier slots to train.");
		return;
	}

	bool is_bool = false;
	const AnnLaySlot* ref_slot = lay.FindSlot(target_slots[0]);
	if(ref_slot) {
		GroupRegistry reg;
		reg.Build(lay);
		is_bool = (ref_slot->method == ANNLAY_CLASSIFIER_BOOL) ||
		          (reg.HeadRole(slot_key) == "presence");
	}

	String crops_dir = TrimBoth(edit_crops_dir.GetData());
	String annprj_path = TrimBoth(edit_annprj_path.GetData());
	String images_dir = TrimBoth(edit_images_dir.GetData());

	if(is_bool) {
		if(annprj_path.IsEmpty()) {
			Cout() << "[SlotTrainer] StartTrain: bool mode missing annprj\n";
			SetStatus("Missing .annprj path for bool training.");
			return;
		}
		if(!FileExists(annprj_path)) {
			Cout() << "[SlotTrainer] StartTrain: annprj missing '" << annprj_path << "'\n";
			SetStatus("annprj file does not exist: " + annprj_path);
			return;
		}
		if(images_dir.IsEmpty()) {
			String def_images = AppendFileName(GetFileDirectory(annprj_path), "images");
			if(DirectoryExists(def_images)) {
				images_dir = def_images;
				edit_images_dir.SetData(images_dir);
			}
		}
		if(images_dir.IsEmpty() || !DirectoryExists(images_dir)) {
			Cout() << "[SlotTrainer] StartTrain: images dir invalid '" << images_dir << "'\n";
			String msg = "Images directory does not exist: " + images_dir;
			SetStatus(msg);
			PromptOK(msg);
			return;
		}
	}
	else {
		Cout() << "[SlotTrainer] StartTrain: label mode crops input='" << crops_dir << "'\n";
		if(crops_dir.IsEmpty()) {
			Cout() << "[SlotTrainer] StartTrain: missing crops dir\n";
			String msg = "Missing crops directory.";
			SetStatus(msg);
			PromptOK(msg);
			return;
		}
		if(!DirectoryExists(crops_dir)) {
			Cout() << "[SlotTrainer] StartTrain: crops dir missing '" << crops_dir << "'\n";
			String msg = "Crops directory does not exist: " + crops_dir;
			SetStatus(msg);
			PromptOK(msg);
			return;
		}

		String expected_disp = AnchoredSlotClassifier::GetSlotGroupDisplayName(lay, slot_key);
		String resolved_crops;
		Vector<String> tried;

		auto TryPath = [&](const String& p) {
			if(resolved_crops.IsEmpty()) {
				tried.Add(p);
				if(LooksLikeClassRoot(p))
					resolved_crops = p;
			}
		};

		// 1. Sibling of current class-root: <parent>/<expected_disp>
		if(LooksLikeClassRoot(crops_dir)) {
			String parent = GetFileDirectory(crops_dir);
			TryPath(AppendFileName(parent, expected_disp));
		}

		// 2. Exact match (already class-root and basename matches)
		if(resolved_crops.IsEmpty() && LooksLikeClassRoot(crops_dir)) {
			if(GetFileName(crops_dir) == expected_disp)
				resolved_crops = crops_dir;
		}

		// 3. <crops>/<display>
		TryPath(AppendFileName(crops_dir, expected_disp));

		// 4. <crops>/pass1/<display>
		TryPath(AppendFileName(AppendFileName(crops_dir, "pass1"), expected_disp));

		// 5. <crops>/pass2/<display>
		TryPath(AppendFileName(AppendFileName(crops_dir, "pass2"), expected_disp));

		// 6. Existing slot-id fallbacks
		if(target_slots.GetCount() == 1) {
			String sid = target_slots[0];
			TryPath(AppendFileName(crops_dir, sid));
			TryPath(AppendFileName(AppendFileName(crops_dir, "pass1"), sid));
			TryPath(AppendFileName(AppendFileName(crops_dir, "pass2"), sid));
		}

		if(resolved_crops.IsEmpty()) {
			Cout() << "[SlotTrainer] StartTrain: failed to resolve class-root for display='" << expected_disp << "' from '" << crops_dir << "'\n";
			String msg = "Failed to find class subdirs for selected group: " + expected_disp + "\nTried paths:\n";
			for(const String& t : tried)
				msg << " - " << t << "\n";
			SetStatus("Resolution failed.");
			PromptOK(msg);
			return;
		}

		if(resolved_crops != crops_dir) {
			Cout() << "[SlotTrainer] StartTrain: auto-switched crops from '" << crops_dir << "' to '" << resolved_crops << "' for slot_key=" << slot_key << "\n";
			SetStatus(Format("Auto-switched crops to: %s", GetFileName(resolved_crops)));
			edit_crops_dir.SetData(resolved_crops);
		}
		else {
			Cout() << "[SlotTrainer] StartTrain: resolved crops='" << resolved_crops << "'\n";
		}
		crops_dir = resolved_crops;
	}

	if(AnySlotHasWeights(target_slots, slot_key)) {
		Cout() << "[SlotTrainer] StartTrain: existing model detected, asking for confirmation\n";
		if(!PromptYesNo("This slot group already has trained weights.\nThis training run will NOT overwrite them until you confirm at the end.\nContinue?"))
			return;
	}

	int epochs = max(1, (int)edit_epochs.GetData());
	bool auto_stop = opt_auto_stop.GetData();

	graph.Clear();

	{
		Mutex::Lock __(lock);
		train_crops_dir = crops_dir;
		train_annprj_path = annprj_path;
		train_images_dir = images_dir;
		train_group_key = slot_key;
		train_auto_stop = auto_stop;
		trained_slot_key = slot_key;
		trained_slot_ids = pick(target_slots);
		loss_history.Clear();
		val_acc_history.Clear();
		target_epochs = epochs;
		train_done = false;
		trained_epochs = 0;
		last_loss = 0;
		last_val_acc = 0;
		train_is_bool = is_bool;
		last_message = is_bool
			? Format("Bool training started: %s [%s]", annprj_path, images_dir)
			: Format("Training started from crops: %s", crops_dir);
	}
	stop_requested.store(false);
	train_running.store(true);
	Cout() << "[SlotTrainer] StartTrain: launching worker is_bool=" << (is_bool ? "true" : "false")
	       << " epochs=" << epochs
	       << " crops='" << crops_dir
	       << "' annprj='" << annprj_path
	       << "' images='" << images_dir << "'\n";

	// Re-register hooks before launching: the backend will call ses.Clear()
	// which wipes WhenStepInterval/WhenSessionLoaded. Re-registering now
	// ensures any WhenSessionLoaded that fires during MakeLayers is caught.
	graph.SetSession(ses);
	layer_view.SetSession(ses);
	pred_view.SetSession(ses);

	train_thread.Run([=] { TrainingMain(); }, true);
	SetControlsEnabled();
}


END_UPP_NAMESPACE
