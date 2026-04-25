#include "ProjectManager.h"
#include "AnnotationEditorCommon.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include "AnchoredSlotExporter.h"
#include <AnnLayCore/AnchoredSlotRecognizer.h>
#include <AnnLayCore/RecognitionScript.h>
#include <AnnLayCore/AiTrainState.h>

NAMESPACE_UPP

namespace {

String GetFallbackBaseAnnlayPath(const String& annlay_path) {
	String p = TrimBoth(annlay_path);
	if(!p.EndsWith("_trained.annlay"))
		return String();
	String dir = GetFileDirectory(p);
	String title = GetFileTitle(p);
	if(!title.EndsWith("_trained"))
		return String();
	String base_title = title.Left(title.GetCount() - 8); // strip "_trained"
	return AppendFileName(dir, base_title + ".annlay");
}

}

void ProjectManager::LoadFromControls(AnnSln& s) const {
	s.name          = TrimBoth(edit_name.GetData());
	s.annprj        = TrimBoth(edit_annprj.GetData());
	s.annlay        = TrimBoth(edit_annlay.GetData());
	s.mlui          = TrimBoth(edit_mlui.GetData());
	s.crops_dir     = TrimBoth(edit_crops_dir.GetData());
	s.templates_dir = TrimBoth(edit_templates_dir.GetData());
	s.images_dir    = TrimBoth(edit_images_dir.GetData());
	s.train_epochs  = max(1, (int)edit_train_epochs.GetData());
}

void ProjectManager::SaveToControls(const AnnSln& s) {
	edit_name.SetData(s.name);
	edit_annprj.SetData(s.annprj);
	edit_annlay.SetData(s.annlay);
	edit_mlui.SetData(s.mlui);
	edit_crops_dir.SetData(s.crops_dir);
	edit_templates_dir.SetData(s.templates_dir);
	edit_images_dir.SetData(s.images_dir);
	edit_train_epochs.SetData(max(1, s.train_epochs));
	RefreshSlotGroups();
	RefreshVerifiedCount();
}

void ProjectManager::DumpSlotArrayCtrl(const char* section, const ArrayCtrl& arr) const {
	Cout() << section << "_count=" << arr.GetCount() << "\n";
	for(int i = 0; i < arr.GetCount(); i++) {
		Cout() << section << "[" << i << "]"
		       << " key=\"" << arr.Get(i, "KEY").ToString() << "\""
		       << " group=\"" << arr.Get(i, 0).ToString() << "\""
		       << " slots=\"" << arr.Get(i, 1).ToString() << "\""
		       << " samples=\"" << arr.Get(i, 2).ToString() << "\""
		       << " val_acc=\"" << arr.Get(i, 3).ToString() << "\""
		       << " trained=\"" << arr.Get(i, 4).ToString() << "\""
		       << "\n";
	}
}

void ProjectManager::DumpToStdout() const {
	Cout() << "[ProjectManager]\n";
	Cout() << "current_sln_path=" << current_sln_path << "\n";
	Cout() << "project_name=" << edit_name.GetData().ToString() << "\n";
	Cout() << "annprj=" << edit_annprj.GetData().ToString() << "\n";
	Cout() << "annlay=" << edit_annlay.GetData().ToString() << "\n";
	Cout() << "mlui=" << edit_mlui.GetData().ToString() << "\n";
	Cout() << "crops_dir=" << edit_crops_dir.GetData().ToString() << "\n";
	Cout() << "templates_dir=" << edit_templates_dir.GetData().ToString() << "\n";
	Cout() << "images_dir=" << edit_images_dir.GetData().ToString() << "\n";
	Cout() << "train_epochs=" << edit_train_epochs.GetData().ToString() << "\n";
	Cout() << "offset_mode=" << drop_offset_mode.GetData().ToString() << "\n";
	Cout() << "bool_gate_policy=" << drop_bool_policy.GetData().ToString() << "\n";
	Cout() << "selected_slot_key_p1=" << selected_slot_key_p1 << "\n";
	Cout() << "selected_slot_key_p2=" << selected_slot_key_p2 << "\n";
	Cout() << "verified_count=" << verified_count << "\n";
	Cout() << "total_image_count=" << total_image_count << "\n";
	Cout() << "verified_label_p1=\"" << lbl_verified_p1.GetData().ToString() << "\"\n";
	Cout() << "verified_label_p2=\"" << lbl_verified_p2.GetData().ToString() << "\"\n";

	DumpSlotArrayCtrl("pass1_slots", arr_slots_p1);
	DumpSlotArrayCtrl("pass2_slots", arr_slots_p2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Open / Save / New
// ─────────────────────────────────────────────────────────────────────────────

bool ProjectManager::LaunchSelf(const Vector<String>& args, bool close_window) {
	LocalProcess proc;
	if(!proc.Start(GetExeFilePath(), args)) {
		PromptOK("Failed to launch AnnotationEditor subprocess.");
		return false;
	}
	if(close_window) Break();
	return true;
}

void ProjectManager::OpenSln(const String& path) {
	String sln_path = TrimBoth(path);
	if(sln_path.IsEmpty()) {
		PromptOK("Empty .annsln path.");
		return;
	}
	if(!IsFullPath(sln_path))
		sln_path = AppendFileName(GetCurrentDirectory(), sln_path);
	sln_path = NormalizePath(sln_path);

	AnnSln s;
	if(!s.Load(sln_path)) {
		PromptOK("Failed to load .annsln file: " + sln_path);
		return;
	}

	String sln_dir = GetFileDirectory(sln_path);
	auto ResolveFromSln = [&](const String& p) -> String {
		String q = TrimBoth(p);
		if(q.IsEmpty()) return String();
		if(IsFullPath(q)) return NormalizePath(q);
		String sln_rel = NormalizePath(AppendFileName(sln_dir, q));
		if(FileExists(sln_rel) || DirectoryExists(sln_rel)) return sln_rel;
		String cwd_rel = NormalizePath(AppendFileName(GetCurrentDirectory(), q));
		if(FileExists(cwd_rel) || DirectoryExists(cwd_rel)) return cwd_rel;
		return sln_rel;
	};

	s.annprj        = ResolveFromSln(s.annprj);
	s.annlay        = ResolveFromSln(s.annlay);
	s.mlui          = ResolveFromSln(s.mlui);
	s.crops_dir     = ResolveFromSln(s.crops_dir);
	s.templates_dir = ResolveFromSln(s.templates_dir);
	s.images_dir    = ResolveFromSln(s.images_dir);

	current_sln_path = sln_path;
	current_sln_cfg = s;
	SaveToControls(s);

	// Set default QA paths if empty
	String models_dir = AppendFileName(sln_dir, "models");
	if(TrimBoth(edit_baseline_json.GetData().ToString()).IsEmpty())
		edit_baseline_json.SetData(NormalizePath(AppendFileName(models_dir, "chip_box_gate_baseline_v2.json")));
	if(TrimBoth(edit_offset_baseline_json.GetData().ToString()).IsEmpty())
		edit_offset_baseline_json.SetData(NormalizePath(AppendFileName(models_dir, "offset_baseline.json")));
	if(TrimBoth(edit_split_json.GetData().ToString()).IsEmpty())
		edit_split_json.SetData(NormalizePath(AppendFileName(sln_dir, "split.json")));
	if(TrimBoth(edit_qa_out_dir.GetData().ToString()).IsEmpty())
		edit_qa_out_dir.SetData(NormalizePath(AppendFileName(models_dir, "qa")));
}

void ProjectManager::OnOpen() {
	FileSel fs;
	fs.Type("Annotation Solution", "*.annsln");
	if(!fs.ExecuteOpen("Open .annsln")) return;
	OpenSln(fs.Get());
}

void ProjectManager::OnSave() {
	if(current_sln_path.IsEmpty()) { OnSaveAs(); return; }
	AnnSln s = current_sln_cfg;
	LoadFromControls(s);
	if(!s.Save(current_sln_path))
		PromptOK("Failed to save .annsln file.");
	else
		current_sln_cfg = s;
}

void ProjectManager::OnSaveAs() {
	FileSel fs;
	fs.Type("Annotation Solution", "*.annsln");
	if(!fs.ExecuteSaveAs("Save .annsln")) return;
	String path = fs.Get();
	if(ToLower(GetFileExt(path)) != ".annsln") path << ".annsln";
	AnnSln s = current_sln_cfg;
	LoadFromControls(s);
	if(!s.Save(path)) {
		PromptOK("Failed to save .annsln file.");
		return;
	}
	current_sln_path = path;
	current_sln_cfg = s;
}

void ProjectManager::OnNew() {
	current_sln_path.Clear();
	AnnSln s;
	current_sln_cfg = s;
	SaveToControls(s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Open Annotation Editor
// ─────────────────────────────────────────────────────────────────────────────

void ProjectManager::OnOpenAnnotationEditor() {
	String annprj = TrimBoth(edit_annprj.GetData());
	if(annprj.IsEmpty()) { PromptOK("annprj path is empty."); return; }
	if(!IsFullPath(annprj)) {
		String cwd_rel = NormalizePath(AppendFileName(GetCurrentDirectory(), annprj));
		String sln_rel = current_sln_path.IsEmpty() ? String()
		               : NormalizePath(AppendFileName(GetFileDirectory(current_sln_path), annprj));
		if(FileExists(cwd_rel))
			annprj = cwd_rel;
		else if(!sln_rel.IsEmpty() && FileExists(sln_rel))
			annprj = sln_rel;
		else
			annprj = cwd_rel;
	}
	else {
		annprj = NormalizePath(annprj);
	}
	if(!FileExists(annprj)) {
		PromptOK(DeQtf("annprj file does not exist:\n" + annprj));
		return;
	}

	open_request_type = OPEN_ANNOTATION_EDITOR;
	request_annprj    = annprj;
	request_crops_dir.Clear();
	request_annlay.Clear();
	request_slot_key.Clear();
	request_images_dir.Clear();
	Break();
}

// ─────────────────────────────────────────────────────────────────────────────
// Slot groups
// ─────────────────────────────────────────────────────────────────────────────

void ProjectManager::PopulateSlotArrayCtrl(ArrayCtrl& arr, int pass_index) {
	String& keep_key = (pass_index == 1) ? selected_slot_key_p1 : selected_slot_key_p2;
	if(keep_key.IsEmpty() && arr.GetCount() > 0)
		keep_key = arr.Get("KEY");

	arr.Clear();
	String annlay_path = TrimBoth(edit_annlay.GetData());
	String crops       = TrimBoth(edit_crops_dir.GetData());
	if(annlay_path.IsEmpty()) return;

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		String fallback = GetFallbackBaseAnnlayPath(annlay_path);
		if(fallback.IsEmpty() || !FileExists(fallback) || !lay.Load(fallback))
			return;
	}

	VectorMap<String, Vector<String>> groups = AnchoredSlotClassifier::GetSlotGroups(lay);

	// Build pass subdir (pass1/ or pass2/)
	String pass_subdir = (pass_index == 1) ? "pass1" : "pass2";
	Value samples_per_group;
	if(!crops.IsEmpty()) {
		String manifest_path = AppendFileName(AppendFileName(crops, pass_subdir), "manifest.json");
		if(FileExists(manifest_path)) {
			Value mroot = ParseJSON(LoadFile(manifest_path));
			if(!IsNull(mroot) && IsValueMap(mroot)) {
				Value spg = mroot["samples_per_group"];
				if(IsValueMap(spg))
					samples_per_group = spg;
			}
		}
	}

	for(int i = 0; i < groups.GetCount(); i++) {
		String key  = groups.GetKey(i);
		String disp = AnchoredSlotClassifier::GetSlotGroupDisplayName(lay, key);

		// samples from manifest.json
		String samples_str = "-";
		String val_acc_str = "-";
		String trained_str = "-";

		if(IsValueMap(samples_per_group)) {
			Value cnt = samples_per_group[disp];
			if(!IsNull(cnt))
				samples_str = AsString((int)cnt);
		}

		// Load .trainstate sidecar
		if(!annlay_path.IsEmpty()) {
			AiTrainState ts;
			if(AiTrainState::LoadGroup(annlay_path, key, ts)) {
				if(!ts.val_acc_history.IsEmpty() && IsFin(ts.val_acc_history.Top()))
					val_acc_str = Format("%.3f", ts.val_acc_history.Top());
				if(!ts.timestamp.IsEmpty()) {
					trained_str = ts.timestamp;
					trained_str.Replace("T", " ");
				}
			}
		}

		arr.Add(disp, groups[i].GetCount(), samples_str, val_acc_str, trained_str);
		arr.Set(i, "KEY", key);
	}

	int cursor = -1;
	for(int i = 0; i < arr.GetCount(); i++) {
		if((String)arr.Get(i, "KEY") == keep_key) { cursor = i; break; }
	}
	if(cursor < 0 && arr.GetCount() > 0) cursor = 0;
	if(cursor >= 0) {
		arr.SetCursor(cursor);
		keep_key = arr.Get(cursor, "KEY");
	}
	else {
		keep_key.Clear();
	}
}

void ProjectManager::RefreshSlotGroups() {
	PopulateSlotArrayCtrl(arr_slots_p1, 1);
	PopulateSlotArrayCtrl(arr_slots_p2, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Verified count
// ─────────────────────────────────────────────────────────────────────────────

void ProjectManager::RefreshVerifiedCount() {
	String annprj_path = TrimBoth(edit_annprj.GetData());
	verified_count    = 0;
	total_image_count = 0;

	if(!annprj_path.IsEmpty() && FileExists(annprj_path)) {
		String json = LoadFile(annprj_path);
		if(!json.IsEmpty()) {
			Value root = ParseJSON(json);
			if(!IsNull(root) && IsValueMap(root)) {
				Value datasets = root["datasets"];
				if(IsValueArray(datasets)) {
					for(int di = 0; di < datasets.GetCount(); di++) {
						Value images = datasets[di]["images"];
						if(!IsValueArray(images)) continue;
						for(int ii = 0; ii < images.GetCount(); ii++) {
							total_image_count++;
							// check metadata_verified
							Value img = images[ii];
							Value meta_keys = img["image_meta_keys"];
							Value meta_vals = img["image_meta_values"];
							if(IsValueArray(meta_keys) && IsValueArray(meta_vals)) {
								for(int ki = 0; ki < meta_keys.GetCount(); ki++) {
									if((String)meta_keys[ki] == "metadata_verified") {
										String v = meta_vals[ki].ToString();
										if(v == "true" || v == "1") verified_count++;
										break;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	String lbl = Format("Verified: %d / %d", verified_count, total_image_count);
	lbl_verified_p1.SetLabel(lbl);
	lbl_verified_p2.SetLabel(lbl);
}

// ─────────────────────────────────────────────────────────────────────────────
// Train
// ─────────────────────────────────────────────────────────────────────────────

void ProjectManager::OnTrainSelected(int pass_index) {
	ArrayCtrl& arr  = (pass_index == 1) ? arr_slots_p1 : arr_slots_p2;
	String& key_ref = (pass_index == 1) ? selected_slot_key_p1 : selected_slot_key_p2;

	int row = arr.GetCursor();
	if(row < 0 || row >= arr.GetCount()) {
		PromptOK("Select a slot group first.");
		return;
	}
	String slot_key = key_ref;
	if(slot_key.IsEmpty()) slot_key = arr.Get(row, "KEY");
	if(slot_key.IsEmpty()) { PromptOK("Selected row has empty slot key."); return; }

	String annlay = TrimBoth(edit_annlay.GetData());
	String crops  = TrimBoth(edit_crops_dir.GetData());
	String annprj = TrimBoth(edit_annprj.GetData());
	if(annlay.IsEmpty()) { PromptOK("annlay is required."); return; }

	AnnLay lay;
	if(!lay.Load(annlay)) { PromptOK("Failed to load annlay."); return; }

	VectorMap<String, Vector<String>> groups = AnchoredSlotClassifier::GetSlotGroups(lay);
	Vector<String> group_slots;
	int g = groups.Find(slot_key);
	if(g >= 0) group_slots = clone(groups[g]);

	bool is_bool = false;
	for(const String& id : group_slots) {
		const AnnLaySlot* s = lay.FindSlot(id);
		if(s) { is_bool = (s->method == ANNLAY_CLASSIFIER_BOOL); break; }
	}

	if(is_bool && annprj.IsEmpty()) {
		PromptOK("annprj is required for bool slot training.");
		return;
	}
	if(!is_bool && crops.IsEmpty()) {
		PromptOK("crops_dir is required for label slot training.");
		return;
	}

	// Derive images_dir
	String images_dir = TrimBoth(edit_images_dir.GetData());
	if(images_dir.IsEmpty() && !annprj.IsEmpty()) {
		String def_images = AppendFileName(GetFileDirectory(annprj), "images");
		if(DirectoryExists(def_images))
			images_dir = def_images;
	}

	open_request_type   = OPEN_SLOT_TRAINER;
	request_annprj      = annprj;
	request_annlay      = annlay;
	request_slot_key    = slot_key;
	request_images_dir  = images_dir;

	if(is_bool) {
		request_crops_dir.Clear();
	}
	else {
		String disp     = AnchoredSlotClassifier::GetSlotGroupDisplayName(lay, slot_key);
		String pass_sub = (pass_index == 1) ? "pass1" : "pass2";
		request_crops_dir = AppendFileName(AppendFileName(crops, pass_sub), disp);
	}
	Break();
}

void ProjectManager::OnTrainCli(int pass_index) {
	String annlay    = TrimBoth(edit_annlay.GetData());
	String crops_dir = TrimBoth(edit_crops_dir.GetData());
	if(annlay.IsEmpty() || crops_dir.IsEmpty()) {
		PromptOK("annlay and crops_dir are required.");
		return;
	}
	String pass_sub = (pass_index == 1) ? "pass1" : "pass2";
	String pass_crops = AppendFileName(crops_dir, pass_sub);

	Vector<String> args;
	args.Add("--train-slots-from-crops");
	args.Add(annlay);
	args.Add(pass_crops);
	args.Add("--epochs");
	args.Add(AsString(max(1, (int)edit_train_epochs.GetData())));

	TopWindow dlg;
	dlg.Title("Training... (headless)");
	dlg.Sizeable().Zoomable();
	LineEdit log_edit;
	log_edit.SetReadOnly();
	Button btn_close;
	btn_close.SetLabel("Close");
	btn_close.Disable();
	dlg.Add(log_edit.HSizePos().VSizePos(0, 32));
	dlg.Add(btn_close.BottomPos(4, 24).RightPos(4, 80));
	btn_close << [&dlg] { dlg.Break(); };
	dlg.SetRect(0, 0, 700, 500);
	dlg.Open();

	LocalProcess proc;
	if(!proc.Start(GetExeFilePath(), args)) {
		log_edit.Insert(log_edit.GetLength(), "Failed to launch training subprocess.\n");
		btn_close.Enable();
		dlg.Run();
		return;
	}

	while(proc.IsRunning()) {
		String chunk = proc.Get();
		if(!chunk.IsEmpty()) {
			log_edit.Insert(log_edit.GetLength(), chunk);
			log_edit.SetCursor(log_edit.GetLength());
		}
		ProcessEvents();
		Sleep(50);
	}
	String tail = proc.Get();
	if(!tail.IsEmpty()) {
		log_edit.Insert(log_edit.GetLength(), tail);
		log_edit.SetCursor(log_edit.GetLength());
	}
	log_edit.Insert(log_edit.GetLength(), "\n--- Done ---\n");
	btn_close.Enable();
	dlg.Run();
}

void ProjectManager::OnRecomputeAnchors() {
	String annlay_path = TrimBoth(edit_annlay.GetData());
	String annprj_path = TrimBoth(edit_annprj.GetData());

	if(annlay_path.IsEmpty() || annprj_path.IsEmpty()) {
		PromptOK("Both annlay and annprj paths are required.");
		return;
	}

	if(!FileExists(annlay_path)) {
		PromptOK("annlay file does not exist:\n" + annlay_path);
		return;
	}
	if(!FileExists(annprj_path)) {
		PromptOK("annprj file does not exist:\n" + annprj_path);
		return;
	}

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		PromptOK("Failed to load annlay:\n" + annlay_path);
		lbl_recompute_p1_status.SetLabel("Error: Load failed");
		return;
	}

	lbl_recompute_p1_status.SetLabel("Recomputing...");
	ProcessEvents();

	lay.ComputeAnchorsFromAnnprj(annprj_path);

	if(!lay.Save(annlay_path)) {
		PromptOK("Failed to save updated annlay:\n" + annlay_path);
		lbl_recompute_p1_status.SetLabel("Error: Save failed");
		return;
	}

	String ts = FormatTime(GetSysTime(), "hh:mm:ss");
	lbl_recompute_p1_status.SetLabel("Recomputed at " + ts);

	RefreshSlotGroups();
}

// ─────────────────────────────────────────────────────────────────────────────
// Export Crops
// ─────────────────────────────────────────────────────────────────────────────

void ProjectManager::OnExportCrops(int pass_index) {
	String annlay        = TrimBoth(edit_annlay.GetData());
	String annprj        = TrimBoth(edit_annprj.GetData());
	String crops_dir     = TrimBoth(edit_crops_dir.GetData());
	String templates_dir = TrimBoth(edit_templates_dir.GetData());
	String images_dir    = TrimBoth(edit_images_dir.GetData());

	if(annlay.IsEmpty())    { PromptOK("annlay is required.");    return; }
	if(crops_dir.IsEmpty()) { PromptOK("crops_dir is required."); return; }
	if(annprj.IsEmpty())    { PromptOK("annprj is required.");    return; }

	// Derive images_dir if not set
	if(images_dir.IsEmpty() && !annprj.IsEmpty()) {
		String def_images = AppendFileName(GetFileDirectory(annprj), "images");
		if(DirectoryExists(def_images))
			images_dir = def_images;
	}

	AnnLay lay;
	if(!lay.Load(annlay)) { PromptOK("Failed to load annlay."); return; }

	Label& status_lbl = (pass_index == 1) ? lbl_export_pass1_status : lbl_export_pass2_status;
	status_lbl.SetLabel("Exporting...");
	ProcessEvents();

	AnchoredSlotExporter exporter;
	AnchoredSlotExportResult result = exporter.Export(
		lay, annprj, images_dir, templates_dir, crops_dir, pass_index);

	if(!result.error.IsEmpty()) {
		status_lbl.SetLabel("Error: " + result.error);
		PromptOK("Export failed:\n" + result.error);
		return;
	}

	// Build summary
	String summary = Format("%d images", result.images_processed);
	status_lbl.SetLabel(summary);

	// Refresh slot list to show updated sample counts
	RefreshSlotGroups();
}

void ProjectManager::OnOpenFrameRecognizer(int pass_index) {
	String sln_path = TrimBoth(current_sln_path);
	if(sln_path.IsEmpty()) {
		PromptOK("No project loaded.");
		return;
	}

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		PromptOK("Failed to load .annsln.");
		return;
	}

	String sln_dir = GetFileDirectory(sln_path);
	String annprj = sln.annprj;
	if(!IsFullPath(annprj))
		annprj = NormalizePath(AppendFileName(sln_dir, annprj));
	String model_set = pass_index == 1 ? "pass1" : "pass2";

	if(!frame_recognizer_win_)
		frame_recognizer_win_.Create();
	
	frame_recognizer_win_->SetOffsetMode(StringToOffsetMode(drop_offset_mode.GetData()));
	frame_recognizer_win_->SetBoolGatePolicy(StringToBoolGatePolicy(drop_bool_policy.GetData()));
	
	frame_recognizer_win_->OpenSlideshow(sln, sln_dir, annprj, model_set);
}

// ─────────────────────────────────────────────────────────────────────────────
// Run Recognition
// ─────────────────────────────────────────────────────────────────────────────

void ProjectManager::OnRunRecognition() {
	String annlay    = TrimBoth(edit_annlay.GetData());
	String annprj    = TrimBoth(edit_annprj.GetData());
	String images_dir = TrimBoth(edit_images_dir.GetData());

	if(annlay.IsEmpty()) { PromptOK("annlay is required."); return; }
	if(annprj.IsEmpty()) { PromptOK("annprj is required."); return; }

	if(images_dir.IsEmpty() && !annprj.IsEmpty()) {
		String def_images = AppendFileName(GetFileDirectory(annprj), "images");
		if(DirectoryExists(def_images))
			images_dir = def_images;
	}

	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(StringToOffsetMode(drop_offset_mode.GetData()));
	rec.SetBoolGatePolicy(StringToBoolGatePolicy(drop_bool_policy.GetData()));
	if(!rec.Load(annlay)) {
		PromptOK("Failed to load annlay for recognition.");
		return;
	}

	AnnSln sln_cfg;
	String script_path;
	if(!current_sln_path.IsEmpty() && FileExists(current_sln_path) && sln_cfg.Load(current_sln_path)) {
		String rel_script = TrimBoth(sln_cfg.recognition_script);
		if(!rel_script.IsEmpty()) {
			if(IsFullPath(rel_script))
				script_path = NormalizePath(rel_script);
			else
				script_path = NormalizePath(AppendFileName(GetFileDirectory(current_sln_path), rel_script));
		}
	}
	if(script_path.IsEmpty()) {
		PromptOK("recognition_script is not configured in the current .annsln.");
		return;
	}
	if(!FileExists(script_path)) {
		PromptOK("recognition_script file does not exist:\n" + script_path);
		return;
	}

	RecognitionScript script;
	if(!script.Load(script_path)) {
		PromptOK("Failed to load recognition script:\n" + script_path + "\n\n" + script.GetLastError());
		return;
	}

	// Load annprj
	String prj_json = LoadFile(annprj);
	if(prj_json.IsEmpty()) { PromptOK("Failed to load annprj."); return; }
	Value root = ParseJSON(prj_json);
	if(IsNull(root) || !IsValueMap(root)) { PromptOK("Invalid annprj JSON."); return; }

	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) { PromptOK("No datasets in annprj."); return; }

	// Count total images for progress
	int total = 0;
	for(int di = 0; di < datasets.GetCount(); di++) {
		Value imgs = datasets[di]["images"];
		if(IsValueArray(imgs)) total += imgs.GetCount();
	}

	lbl_recognize_p1_status.SetLabel(Format("0 / %d", total));
	lbl_recognize_p2_status.SetLabel(Format("0 / %d", total));
	ProcessEvents();

	int done = 0;
	int script_errors = 0;
	String first_script_error;

	ValueArray new_datasets;
	for(int di = 0; di < datasets.GetCount(); di++) {
		Value ds = datasets[di];
		Value images = ds["images"];
		ValueArray new_images;
		int img_count = IsValueArray(images) ? images.GetCount() : 0;
		for(int ii = 0; ii < img_count; ii++) {
			Value img_rec = images[ii];
			String filename = img_rec["file_name"];
			if(filename.IsEmpty()) filename = img_rec["filename"];

			// Build image path
			String img_path;
			if(IsFullPath(filename))
				img_path = filename;
			else if(!images_dir.IsEmpty())
				img_path = AppendFileName(images_dir, filename);
			else
				img_path = AppendFileName(GetFileDirectory(annprj), filename);

				ValueMap new_rec;
				// Copy all original fields
				if(IsValueMap(img_rec)) {
					ValueMap orig = img_rec;
					for(int fi = 0; fi < orig.GetCount(); fi++)
						new_rec.Add(orig.GetKey(fi), orig[fi]);
				}

				VectorMap<String, String> image_metadata;
				Value meta_keys = img_rec["image_meta_keys"];
				Value meta_vals = img_rec["image_meta_values"];
				if(IsValueArray(meta_keys) && IsValueArray(meta_vals)) {
					for(int mi = 0; mi < min(meta_keys.GetCount(), meta_vals.GetCount()); mi++) {
						String key = meta_keys[mi].ToString();
						if(!key.IsEmpty())
							image_metadata.GetAdd(key) = meta_vals[mi].ToString();
					}
				}

				if(FileExists(img_path)) {
					Image img = StreamRaster::LoadFileAny(img_path);
					if(!img.IsEmpty()) {
						Vector<SlotResult> results = rec.Recognize(img);
						ValueArray rec_arr;
						for(int ri = 0; ri < results.GetCount(); ri++) {
							ValueMap sr;
							sr.Add("slot_id",     results[ri].slot_id);
							sr.Add("method",      results[ri].method);
							sr.Add("raw_text",    results[ri].raw_text);
							sr.Add("top_class",   results[ri].top_class);
							sr.Add("class_index", results[ri].class_index);
							sr.Add("confidence",  results[ri].confidence);
							rec_arr.Add(sr);
						}
						new_rec.GetAdd("recognition") = rec_arr;

						VectorMap<String, String> script_meta = script.Run(results);
						String run_err = script.GetLastError();
						if(!run_err.IsEmpty()) {
							script_errors++;
							if(first_script_error.IsEmpty())
								first_script_error = run_err;
						}
						else {
							for(int sm = 0; sm < script_meta.GetCount(); sm++)
								image_metadata.GetAdd(script_meta.GetKey(sm)) = script_meta[sm];
						}
					}
				}

				ValueArray out_keys;
				ValueArray out_vals;
				for(int mi = 0; mi < image_metadata.GetCount(); mi++) {
					out_keys.Add(image_metadata.GetKey(mi));
					out_vals.Add(image_metadata[mi]);
				}
				new_rec.GetAdd("image_meta_keys") = out_keys;
				new_rec.GetAdd("image_meta_values") = out_vals;

				new_images.Add(new_rec);
				done++;

			if(done % 5 == 0 || done == total) {
				String prog = Format("%d / %d", done, total);
				lbl_recognize_p1_status.SetLabel(prog);
				lbl_recognize_p2_status.SetLabel(prog);
				ProcessEvents();
			}
		}

		ValueMap new_ds;
		if(IsValueMap(ds)) {
			ValueMap orig = ds;
			for(int fi = 0; fi < orig.GetCount(); fi++)
				new_ds.Add(orig.GetKey(fi), orig[fi]);
		}
		new_ds.GetAdd("images") = new_images;
		new_datasets.Add(new_ds);
	}

	// Rebuild root
	ValueMap new_root;
	if(IsValueMap(root)) {
		ValueMap orig = root;
		for(int fi = 0; fi < orig.GetCount(); fi++)
			new_root.Add(orig.GetKey(fi), orig[fi]);
	}
	new_root.GetAdd("datasets") = new_datasets;

	if(!SaveFile(annprj, StoreAsJson(new_root, true))) {
		PromptOK("Failed to save annprj with recognition results.");
		return;
	}

	String prog = Format("%d / %d done", done, total);
	lbl_recognize_p1_status.SetLabel(prog);
	lbl_recognize_p2_status.SetLabel(prog);
	RefreshVerifiedCount();

	if(script_errors > 0) {
		PromptOK(Format("Recognition finished with %d script errors.\nFirst error:\n%s",
		                script_errors, first_script_error));
	}
}

bool ProjectManager::RunCli(const Vector<String>& args, const String& action_name) {
	lbl_qa_status.SetLabel(action_name + ": running...");
	lbl_qa_status.SetInk(SBlack());
	ProcessEvents();

	String out;
	LocalProcess proc;
	if(!proc.Start(GetExeFilePath(), args)) {
		lbl_qa_status.SetLabel(action_name + ": failed to start subprocess");
		lbl_qa_status.SetInk(SRed());
		return false;
	}

	while(proc.IsRunning()) {
		out << proc.Get();
		ProcessEvents();
		Sleep(10);
	}
	out << proc.Get();

	int exit_code = proc.GetExitCode();
	String ts = FormatTime(GetSysTime(), "hh:mm:ss");

	if(exit_code == 0) {
		lbl_qa_status.SetLabel(Format("%s: PASSED at %s", action_name, ts));
		lbl_qa_status.SetInk(DkGreen());
		return true;
	}
	else {
		lbl_qa_status.SetLabel(Format("%s: FAILED (code %d) at %s", action_name, exit_code, ts));
		lbl_qa_status.SetInk(SRed());
		
		String cmd = GetExeFilePath();
		for(const String& a : args) cmd << " " << a;
		
		PromptOK(DeQtf(Format("%s failed!\n\nCommand:\n%s\n\nExit code: %d\n\nLast output:\n%s", 
		               action_name, cmd, exit_code, out.Right(500))));
		return false;
	}
}

void ProjectManager::OnAuditBool() {
	String annlay = TrimBoth(edit_annlay.GetData());
	String out_dir = TrimBoth(edit_qa_out_dir.GetData());
	if(annlay.IsEmpty()) { PromptOK("annlay is required."); return; }
	
	RealizeDirectory(out_dir);
	String out_file = AppendFileName(out_dir, "audit_report.json");

	Vector<String> args;
	args.Add("--audit-bool-gates");
	args.Add(annlay);
	args.Add("--strict");
	args.Add("--out");
	args.Add(out_file);

	if(RunCli(args, "Audit")) {
		lbl_qa_status.SetLabel(lbl_qa_status.GetData().ToString() + " (Report: " + GetFileName(out_file) + ")");
	}
}

void ProjectManager::OnFixBool() {
	String annlay = TrimBoth(edit_annlay.GetData());
	if(annlay.IsEmpty()) { PromptOK("annlay is required."); return; }

	Vector<String> args;
	args.Add("--fix-bool-gates");
	args.Add(annlay);

	if(RunCli(args, "Fix Bool Gates")) {
		RefreshSlotGroups();
	}
}

void ProjectManager::OnEvalBool() {
	String annlay = TrimBoth(edit_annlay.GetData());
	if(annlay.IsEmpty()) { PromptOK("annlay is required."); return; }
	
	// We need a solution file for --eval-bool-gates
	if(current_sln_path.IsEmpty()) {
		PromptOK("Save the project to a .annsln file first.");
		return;
	}

	String split = TrimBoth(edit_split_json.GetData());
	String policy = drop_bool_policy.GetData();
	String mode = drop_offset_mode.GetData();

	Vector<String> args;
	args.Add("--eval-bool-gates");
	args.Add(current_sln_path);
	args.Add("--all");
	if(!split.IsEmpty()) {
		args.Add("--split-file");
		args.Add(split);
	}
	args.Add("--bool-gate-policy");
	args.Add(policy);
	args.Add("--offset-mode");
	args.Add(mode);

	RunCli(args, "Eval");
}

void ProjectManager::OnCheckBaseline() {
	if(current_sln_path.IsEmpty()) {
		PromptOK("Save the project to a .annsln file first.");
		return;
	}

	String baseline = TrimBoth(edit_baseline_json.GetData());
	if(baseline.IsEmpty()) { PromptOK("baseline.json path is required."); return; }
	if(!FileExists(baseline)) { PromptOK("baseline.json not found: " + baseline); return; }

	String split = TrimBoth(edit_split_json.GetData());
	String out_dir = TrimBoth(edit_qa_out_dir.GetData());
	String policy = drop_bool_policy.GetData();
	String mode = drop_offset_mode.GetData();
	int min_bool_support = edit_min_bool_support.GetData();
	bool fail_on_low_support = opt_fail_on_low_support.Get();
	bool allow_mismatch = opt_allow_baseline_mismatch.Get();
	bool refresh_baselines = opt_refresh_baselines.Get();

	RealizeDirectory(out_dir);
	String out_file = AppendFileName(out_dir, "check_report.json");

	Vector<String> args;
	args.Add("--check-bool-baseline");
	args.Add(current_sln_path);
	args.Add("--baseline");
	args.Add(baseline);
	if(!split.IsEmpty()) {
		args.Add("--split-file");
		args.Add(split);
	}
	args.Add("--check-out");
	args.Add(out_file);
	args.Add("--bool-gate-policy");
	args.Add(policy);
	args.Add("--offset-mode");
	args.Add(mode);
	if(allow_mismatch)
		args.Add("--allow-baseline-mismatch");
	if(min_bool_support > 0) {
		args.Add("--min-bool-support");
		args.Add(AsString(min_bool_support));
		args.Add("--fail-on-low-support");
		args.Add(fail_on_low_support ? "1" : "0");
	}

	if(RunCli(args, "Check Baseline")) {
		lbl_qa_status.SetLabel(lbl_qa_status.GetData().ToString() + " (Report: " + GetFileName(out_file) + ")");
	}
}

void ProjectManager::OnRunPreflight(bool headless) {
	String sln = current_sln_path;
	String annlay = TrimBoth(edit_annlay.GetData());
	String bool_baseline = TrimBoth(edit_baseline_json.GetData());
	String offset_baseline = TrimBoth(edit_offset_baseline_json.GetData());
	String out_dir = TrimBoth(edit_qa_out_dir.GetData());
	String split = TrimBoth(edit_split_json.GetData());
	String mode = drop_offset_mode.GetData();
	String policy = drop_bool_policy.GetData();
	bool fail_on_warnings = opt_fail_on_warnings.Get();
	int min_bool_support = edit_min_bool_support.GetData();
	bool fail_on_low_support = opt_fail_on_low_support.Get();
	bool allow_mismatch = opt_allow_baseline_mismatch.Get();
	bool refresh_baselines = opt_refresh_baselines.Get();

	if(sln.IsEmpty()) { if(!headless) PromptOK("Solution path is empty. Save project first."); else Cout() << "Preflight: Solution path is empty.\n"; return; }
	if(annlay.IsEmpty()) { if(!headless) PromptOK("annlay path is empty."); else Cout() << "Preflight: annlay path is empty.\n"; return; }
	if(bool_baseline.IsEmpty() || !FileExists(bool_baseline)) { if(!headless) PromptOK("Bool baseline not found: " + bool_baseline); else Cout() << "Preflight: Bool baseline not found: " << bool_baseline << "\n"; return; }
	if(offset_baseline.IsEmpty() || !FileExists(offset_baseline)) { if(!headless) PromptOK("Offset baseline not found: " + offset_baseline); else Cout() << "Preflight: Offset baseline not found: " << offset_baseline << "\n"; return; }
	if(out_dir.IsEmpty()) { if(!headless) PromptOK("Output directory is required."); else Cout() << "Preflight: Output directory is required.\n"; return; }

	RealizeDirectory(out_dir);
	
	if(!headless) btn_run_preflight.Disable();

	Vector<String> args;
	args.Add("--run-phase6-preflight");
	args.Add(sln);
	args.Add("--bool-baseline");
	args.Add(bool_baseline);
	args.Add("--offset-baseline");
	args.Add(offset_baseline);
	args.Add("--out-dir");
	args.Add(out_dir);
	if(!split.IsEmpty()) {
		args.Add("--split-file");
		args.Add(split);
	}
	args.Add("--offset-mode");
	args.Add(mode);
	args.Add("--bool-gate-policy");
	args.Add(policy);
	args.Add("--fail-on-warnings");
	args.Add(fail_on_warnings ? "1" : "0");
	if(allow_mismatch)
		args.Add("--allow-baseline-mismatch");
	if(min_bool_support > 0) {
		args.Add("--min-bool-support");
		args.Add(AsString(min_bool_support));
		args.Add("--fail-on-low-support");
		args.Add(fail_on_low_support ? "1" : "0");
	}

	lbl_qa_status.SetLabel("Preflight running...");
	lbl_qa_status.SetInk(SBlack());
	ProcessEvents();

	String out;
	LocalProcess proc;
	if(!proc.Start(GetExeFilePath(), args)) {
		lbl_qa_status.SetLabel("Preflight: failed to start subprocess");
		lbl_qa_status.SetInk(SRed());
		if(!headless) btn_run_preflight.Enable();
		return;
	}

	while(proc.IsRunning()) {
		out << proc.Get();
		ProcessEvents();
		Sleep(10);
	}
	out << proc.Get();

	int exit_code = proc.GetExitCode();
	bool passed = (exit_code == 0);

	String summary_path = AppendFileName(out_dir, "phase6_preflight_summary.json");
	Value summary = ParseJSON(LoadFile(summary_path));
	int total_warnings = 0;
	String failed_step;
	if(!IsNull(summary)) {
		total_warnings = summary["total_warnings"];
		if(!passed) failed_step = summary["failed_step"];
	}

	if(!passed) {
		String failure_label = "Preflight FAILED";
		if(!failed_step.IsEmpty()) failure_label << " at step: " << failed_step;
		lbl_qa_status.SetLabel(failure_label);
		lbl_qa_status.SetInk(SRed());
		
		String cmd = GetExeFilePath();
		for(const String& a : args) cmd << " " << a;
		
		String message = Format("Preflight failed!\n\nCommand:\n%s\n\nExit code: %d", cmd, exit_code);
		message << "\n\nLast output:\n" << out.Right(1000);

		if(out.Find("Target slot exists but required gate is missing") >= 0) {
			message << "\n\nHINT: Run 'bin/AnnotationEditor --fix-bool-gates " << GetFileName(annlay) << "' to fix missing gate slots.";
		}
		if(out.Find("WARNING: Baseline provenance mismatch detected!") >= 0) {
			message << "\n\nHINT: Baselines are stale or mismatched. Click 'Refresh Phase 6 Baselines' to regenerate them.";
		}

		if(!headless) {
			PromptOK(DeQtf(message));
		}
		else {
			Cout() << message << "\n";
		}
	}
	else {
		String ts = FormatTime(GetSysTime(), "hh:mm:ss");
		String status = Format("Preflight: PASSED at %s (Warnings: %d)", ts, total_warnings);
		lbl_qa_status.SetLabel(status);
		lbl_qa_status.SetInk(DkGreen());
	}
	
	if(!headless) btn_run_preflight.Enable();
}

void ProjectManager::TestPreflight(const String& sln_path, const String& bool_baseline, const String& offset_baseline, const String& out_dir, const String& split, bool fail_on_warnings) {
	OpenSln(sln_path);
	edit_baseline_json.SetData(bool_baseline);
	edit_offset_baseline_json.SetData(offset_baseline);
	edit_qa_out_dir.SetData(out_dir);
	edit_split_json.SetData(split);
	opt_fail_on_warnings.Set(fail_on_warnings);
	OnRunPreflight(true);
}

void ProjectManager::OnRefreshBaselines() {
	if(current_sln_path.IsEmpty()) {
		PromptOK("Save the project to a .annsln file first.");
		return;
	}

	String sln = current_sln_path;
	String out_dir = TrimBoth(edit_qa_out_dir.GetData());
	String split = TrimBoth(edit_split_json.GetData());
	String policy = drop_bool_policy.GetData();
	String mode = drop_offset_mode.GetData();

	if(out_dir.IsEmpty()) { PromptOK("Output directory is required."); return; }

	RealizeDirectory(out_dir);
	btn_refresh_baselines.Disable();
	lbl_qa_status.SetLabel("Refreshing baselines...");
	lbl_qa_status.SetInk(SBlack());
	ProcessEvents();

	Vector<String> args;
	args.Add("--refresh-phase6-baselines");
	args.Add(sln);
	args.Add("--out-dir");
	args.Add(out_dir);
	if(!split.IsEmpty()) {
		args.Add("--split-file");
		args.Add(split);
	}
	args.Add("--bool-gate-policy");
	args.Add(policy);
	args.Add("--offset-mode");
	args.Add(mode);

	String out;
	LocalProcess proc;
	if(!proc.Start(GetExeFilePath(), args)) {
		lbl_qa_status.SetLabel("Refresh failed: could not start subprocess");
		lbl_qa_status.SetInk(SRed());
		btn_refresh_baselines.Enable();
		return;
	}

	while(proc.IsRunning()) {
		out << proc.Get();
		ProcessEvents();
		Sleep(10);
	}
	out << proc.Get();

	int exit_code = proc.GetExitCode();
	if(exit_code != 0) {
		lbl_qa_status.SetLabel("Refresh FAILED");
		lbl_qa_status.SetInk(SRed());
		PromptOK(DeQtf(Format("Baseline refresh failed!\n\nExit code: %d\n\nOutput:\n%s", exit_code, out.Right(1000))));
	} else {
		lbl_qa_status.SetLabel("Refresh PASSED");
		lbl_qa_status.SetInk(DkGreen());

		// Update fields
		edit_baseline_json.SetData(NormalizePath(AppendFileName(out_dir, "bool_baseline.json")));
		edit_offset_baseline_json.SetData(NormalizePath(AppendFileName(out_dir, "offset_baseline.json")));

		String generated_split = NormalizePath(AppendFileName(out_dir, "eval_split.json"));
		if(FileExists(generated_split)) {
			edit_split_json.SetData(generated_split);
		}

		PromptOK("Phase 6 baselines refreshed successfully.");
	}

	btn_refresh_baselines.Enable();
}

void ProjectManager::OnRunExitGate() {
	if(current_sln_path.IsEmpty()) {
		PromptOK("Save the project to a .annsln file first.");
		return;
	}

	String sln = current_sln_path;
	String out_dir = TrimBoth(edit_qa_out_dir.GetData());
	String split = TrimBoth(edit_split_json.GetData());
	String mode = drop_offset_mode.GetData();
	String policy = drop_bool_policy.GetData();
	bool fail_on_warnings = opt_fail_on_warnings.Get();
	int min_bool_support = edit_min_bool_support.GetData();
	bool fail_on_low_support = opt_fail_on_low_support.Get();
	bool allow_mismatch = opt_allow_baseline_mismatch.Get();
	bool refresh_baselines = opt_refresh_baselines.Get();

	if(out_dir.IsEmpty()) { PromptOK("Output directory is required."); return; }

	RealizeDirectory(out_dir);
	btn_run_exit_gate.Disable();
	lbl_qa_status.SetLabel("Exit Gate: running...");
	lbl_qa_status.SetInk(SBlack());
	ProcessEvents();

	Vector<String> args;
	args.Add("--run-phase6-exit-gate");
	args.Add(sln);
	args.Add("--out-dir");
	args.Add(out_dir);
	if(!split.IsEmpty()) {
		args.Add("--split-file");
		args.Add(split);
	}
	args.Add("--offset-mode");
	args.Add(mode);
	args.Add("--bool-gate-policy");
	args.Add(policy);
	args.Add("--fail-on-warnings");
	args.Add(fail_on_warnings ? "1" : "0");
	if(allow_mismatch)
		args.Add("--allow-baseline-mismatch");
	args.Add("--min-bool-support");
	args.Add(AsString(min_bool_support));
	args.Add("--fail-on-low-support");
	args.Add(fail_on_low_support ? "1" : "0");
	args.Add("--refresh-baselines");
	args.Add(refresh_baselines ? "1" : "0");

	String out;
	LocalProcess proc;
	if(!proc.Start(GetExeFilePath(), args)) {
		lbl_qa_status.SetLabel("Exit Gate failed: could not start subprocess");
		lbl_qa_status.SetInk(SRed());
		btn_run_exit_gate.Enable();
		return;
	}

	while(proc.IsRunning()) {
		out << proc.Get();
		ProcessEvents();
		Sleep(10);
	}
	out << proc.Get();

	int exit_code = proc.GetExitCode();
	String summary_path = AppendFileName(out_dir, "phase6_exit_gate_summary.json");
	
	if(exit_code == 0) {
		lbl_qa_status.SetLabel("Exit Gate: PASSED (Summary: " + GetFileName(summary_path) + ")");
		lbl_qa_status.SetInk(DkGreen());
		
		// Auto-update paths
		edit_baseline_json.SetData(NormalizePath(AppendFileName(out_dir, "bool_baseline.json")));
		edit_offset_baseline_json.SetData(NormalizePath(AppendFileName(out_dir, "offset_baseline.json")));
		String generated_split = NormalizePath(AppendFileName(out_dir, "eval_split.json"));
		if(FileExists(generated_split)) edit_split_json.SetData(generated_split);
		
		PromptOK("Phase 6 Exit Gate PASSED.\nSummary: " + GetFileName(summary_path));
	} else {
		lbl_qa_status.SetLabel("Exit Gate: FAILED");
		lbl_qa_status.SetInk(SRed());
		PromptOK(DeQtf(Format("Phase 6 Exit Gate FAILED!\n\nExit code: %d\n\nOutput:\n%s", exit_code, out.Right(1000))));
	}

	btn_run_exit_gate.Enable();
}

END_UPP_NAMESPACE
