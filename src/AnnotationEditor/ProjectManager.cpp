#include "ProjectManager.h"
#include "AnnotationEditorCommon.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include "AnchoredSlotExporter.h"
#include <AnnLayCore/AnchoredSlotRecognizer.h>
#include <AnnLayCore/RecognitionScript.h>

NAMESPACE_UPP

// ─────────────────────────────────────────────────────────────────────────────
// Layout helpers
// ─────────────────────────────────────────────────────────────────────────────

static const int kRowH = 24;
static const int kGap  = 4;
static const int kLblW = 100;

// Return y position after adding a label + editstring row (HSizePos for edit)
static int AddLabeledEdit(TopWindow& win, Label& lbl, EditField& ed,
                          int y, const char* text) {
	lbl.SetLabel(text);
	win.Add(lbl.LeftPos(12, kLblW).TopPos(y, kRowH));
	win.Add(ed.HSizePos(kLblW + 16, 12).TopPos(y, kRowH));
	return y + kRowH + kGap;
}

// ─────────────────────────────────────────────────────────────────────────────
// ProjectManager constructor
// ─────────────────────────────────────────────────────────────────────────────

void ProjectManager::SetupSlotArrayCtrl(ArrayCtrl& arr) {
	arr.AddColumn("Group");
	arr.AddColumn("Slots");
	arr.AddColumn("Samples");
	arr.AddColumn("Val.Acc");
	arr.AddColumn("Trained");
	arr.AddIndex("KEY");
	arr.ColumnWidths("160 50 70 70 150");
	arr.EvenRowColor();
	arr.MultiSelect(false);
}

ProjectManager::ProjectManager() {
	Title("Project Manager");
	Sizeable().Zoomable();
	SetRect(0, 0, 900, 820);

	// ── Project section header ──
	int y = 12;
	Add(lbl_header.SetLabel("Project:").LeftPos(12, 70).TopPos(y, kRowH));
	Add(edit_name.LeftPos(84, 260).TopPos(y, kRowH));
	Add(btn_open.SetLabel("Open...").RightPos(216, 90).TopPos(y, kRowH));
	Add(btn_save.SetLabel("Save").RightPos(120, 90).TopPos(y, kRowH));
	Add(btn_new.SetLabel("New").RightPos(24, 90).TopPos(y, kRowH));
	y += kRowH + kGap;

	y = AddLabeledEdit(*this, lbl_annprj,        edit_annprj,        y, "annprj");
	y = AddLabeledEdit(*this, lbl_annlay,         edit_annlay,        y, "annlay");
	y = AddLabeledEdit(*this, lbl_mlui,           edit_mlui,          y, "mlui");
	y = AddLabeledEdit(*this, lbl_crops_dir,      edit_crops_dir,     y, "crops_dir");
	y = AddLabeledEdit(*this, lbl_templates_dir,  edit_templates_dir, y, "templates_dir");
	y = AddLabeledEdit(*this, lbl_images_dir,     edit_images_dir,    y, "images_dir");

	Add(lbl_train_epochs.SetLabel("train_epochs").LeftPos(12, kLblW).TopPos(y, kRowH));
	Add(edit_train_epochs.LeftPos(kLblW + 16, 80).TopPos(y, kRowH));
	edit_train_epochs.MinMax(1, 1000000);
	
	Add(lbl_offset_mode.SetLabel("Offset Mode").LeftPos(220, 80).TopPos(y, kRowH));
	Add(drop_offset_mode.LeftPos(310, 100).TopPos(y, kRowH));
	drop_offset_mode.Add("auto");
	drop_offset_mode.Add("none");
	drop_offset_mode.Add("combined");
	drop_offset_mode.Add("split");
	drop_offset_mode.SetData("auto");

	Add(lbl_bool_policy.SetLabel("Gate Policy").LeftPos(430, 80).TopPos(y, kRowH));
	Add(drop_bool_policy.LeftPos(520, 100).TopPos(y, kRowH));
	drop_bool_policy.Add("permissive");
	drop_bool_policy.Add("strict");
	drop_bool_policy.SetData("strict");

	y += kRowH + 8;

	// ── Pass 1 LabelBox ──────────────────────────────────────────────────────
	const int kBoxX  = 8;
	const int kBoxW  = -8;  // HSizePos right margin
	const int kInner = 24;  // inner left margin within box
	const int kInnerR = 20; // inner right margin

	int p1y = y;
	Add(box_pass1.SetLabel("Pass 1 — Bootstrap").HSizePos(kBoxX, 8).TopPos(p1y, 10)); // height set later

	int p1inner = p1y + 20;

	// Step 1: Annotate
	{
		int sy = p1inner;
		Add(box_p1_annotate.SetLabel("Step 1: Annotate").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		Add(btn_open_annotation_editor.SetLabel("Open Annotation Editor")
		    .LeftPos(kInner + 8, 220).TopPos(sy, kRowH));
		sy += kRowH + 6;
		box_p1_annotate.HSizePos(kInner, kInnerR).TopPos(p1inner, sy - p1inner);
		p1inner = sy + 6;
	}

	// Step 2: Export Crops (Pass 1)
	{
		int sy = p1inner;
		Add(box_p1_export.SetLabel("Step 2: Export Crops").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		Add(btn_export_pass1.SetLabel("Export Pass 1 Crops")
		    .LeftPos(kInner + 8, 200).TopPos(sy, kRowH));
		Add(lbl_export_pass1_status.SetLabel("").HSizePos(kInner + 8 + 210, kInnerR + 8).TopPos(sy + 4, kRowH));
		sy += kRowH + 6;
		box_p1_export.HSizePos(kInner, kInnerR).TopPos(p1inner, sy - p1inner);
		p1inner = sy + 6;
	}

	// Step 3: Train (Pass 1)
	{
		int sy = p1inner;
		Add(box_p1_train.SetLabel("Step 3: Train Classifiers").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		SetupSlotArrayCtrl(arr_slots_p1);
		Add(arr_slots_p1.HSizePos(kInner + 8, kInnerR + 8).TopPos(sy, 130));
		sy += 130 + 4;
		Add(btn_train_selected_p1.SetLabel("Train Selected")
		    .LeftPos(kInner + 8, 150).TopPos(sy, kRowH));
		Add(btn_train_cli_p1.SetLabel("Train All (headless)")
		    .LeftPos(kInner + 8 + 160, 180).TopPos(sy, kRowH));
		sy += kRowH + 6;
		box_p1_train.HSizePos(kInner, kInnerR).TopPos(p1inner, sy - p1inner);
		p1inner = sy + 6;
	}

	// Step 4: Recompute Anchors
	{
		int sy = p1inner;
		Add(box_p1_recompute.SetLabel("Step 4: Recompute Anchors").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		Add(btn_recompute_p1.SetLabel("Recompute From annprj")
		    .LeftPos(kInner + 8, 200).TopPos(sy, kRowH));
		Add(lbl_recompute_p1_status.SetLabel("").HSizePos(kInner + 8 + 210, kInnerR + 8).TopPos(sy + 4, kRowH));
		sy += kRowH + 6;
		box_p1_recompute.HSizePos(kInner, kInnerR).TopPos(p1inner, sy - p1inner);
		p1inner = sy + 6;
	}

	// Step 5: Recognize (Pass 1)
	{
		int sy = p1inner;
		Add(box_p1_recognize.SetLabel("Step 5: Run Recognition").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		Add(btn_run_recognition_p1.SetLabel("Run Headless")
		    .LeftPos(kInner + 8, 150).TopPos(sy, kRowH));
		Add(btn_open_recognizer_p1.SetLabel("Open FrameRecognizer")
		    .LeftPos(kInner + 8 + 160, 200).TopPos(sy, kRowH));
		Add(lbl_recognize_p1_status.SetLabel("").HSizePos(kInner + 8 + 370, kInnerR + 8).TopPos(sy + 4, kRowH));
		sy += kRowH + 6;
		box_p1_recognize.HSizePos(kInner, kInnerR).TopPos(p1inner, sy - p1inner);
		p1inner = sy + 6;
	}

	// Step 6: Review (Pass 1)
	{
		int sy = p1inner;
		Add(box_p1_review.SetLabel("Step 6: Review").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		Add(lbl_verified_p1.SetLabel("Verified: 0 / 0")
		    .LeftPos(kInner + 8, 200).TopPos(sy + 4, kRowH));
		Add(btn_open_annotation_review_p1.SetLabel("Open Annotation Editor")
		    .LeftPos(kInner + 8 + 210, 220).TopPos(sy, kRowH));
		sy += kRowH + 6;
		box_p1_review.HSizePos(kInner, kInnerR).TopPos(p1inner, sy - p1inner);
		p1inner = sy + 6;
	}

	// Close Pass 1 box
	box_pass1.HSizePos(kBoxX, 8).TopPos(p1y, p1inner - p1y + 4);
	y = p1inner + 10;

	// ── Pass 2 LabelBox ──────────────────────────────────────────────────────
	int p2y = y;
	Add(box_pass2.SetLabel("Pass 2 — Refinement (requires verified images)").HSizePos(kBoxX, 8).TopPos(p2y, 10));

	int p2inner = p2y + 20;

	// Step 6: Export Crops (Pass 2)
	{
		int sy = p2inner;
		Add(box_p2_export.SetLabel("Step 6: Export Crops").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		Add(btn_export_pass2.SetLabel("Export Pass 2 Crops")
		    .LeftPos(kInner + 8, 200).TopPos(sy, kRowH));
		Add(lbl_export_pass2_status.SetLabel("").HSizePos(kInner + 8 + 210, kInnerR + 8).TopPos(sy + 4, kRowH));
		sy += kRowH + 6;
		box_p2_export.HSizePos(kInner, kInnerR).TopPos(p2inner, sy - p2inner);
		p2inner = sy + 6;
	}

	// Step 7: Train (Pass 2)
	{
		int sy = p2inner;
		Add(box_p2_train.SetLabel("Step 7: Train Classifiers").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		SetupSlotArrayCtrl(arr_slots_p2);
		Add(arr_slots_p2.HSizePos(kInner + 8, kInnerR + 8).TopPos(sy, 130));
		sy += 130 + 4;
		Add(btn_train_selected_p2.SetLabel("Train Selected")
		    .LeftPos(kInner + 8, 150).TopPos(sy, kRowH));
		Add(btn_train_cli_p2.SetLabel("Train All (headless)")
		    .LeftPos(kInner + 8 + 160, 180).TopPos(sy, kRowH));
		sy += kRowH + 6;
		box_p2_train.HSizePos(kInner, kInnerR).TopPos(p2inner, sy - p2inner);
		p2inner = sy + 6;
	}

	// Step 8: Recognize (Pass 2)
	{
		int sy = p2inner;
		Add(box_p2_recognize.SetLabel("Step 8: Run Recognition").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		Add(btn_run_recognition_p2.SetLabel("Run Headless")
		    .LeftPos(kInner + 8, 150).TopPos(sy, kRowH));
		Add(btn_open_recognizer_p2.SetLabel("Open FrameRecognizer")
		    .LeftPos(kInner + 8 + 160, 200).TopPos(sy, kRowH));
		Add(lbl_recognize_p2_status.SetLabel("").HSizePos(kInner + 8 + 370, kInnerR + 8).TopPos(sy + 4, kRowH));
		sy += kRowH + 6;
		box_p2_recognize.HSizePos(kInner, kInnerR).TopPos(p2inner, sy - p2inner);
		p2inner = sy + 6;
	}

	// Step 9: Review (Pass 2)
	{
		int sy = p2inner;
		Add(box_p2_review.SetLabel("Step 9: Review").HSizePos(kInner, kInnerR).TopPos(sy, 10));
		sy += 18;
		Add(lbl_verified_p2.SetLabel("Verified: 0 / 0")
		    .LeftPos(kInner + 8, 200).TopPos(sy + 4, kRowH));
		Add(btn_open_annotation_review_p2.SetLabel("Open Annotation Editor")
		    .LeftPos(kInner + 8 + 210, 220).TopPos(sy, kRowH));
		sy += kRowH + 6;
		box_p2_review.HSizePos(kInner, kInnerR).TopPos(p2inner, sy - p2inner);
		p2inner = sy + 6;
	}

	// Close Pass 2 box
	box_pass2.HSizePos(kBoxX, 8).TopPos(p2y, p2inner - p2y + 4);
	y = p2inner + 10;

	// ── QA / Validation ──────────────────────────────────────────────────────
	int qay = y;
	Add(box_qa.SetLabel("QA / Validation (Bool Gates)").HSizePos(kBoxX, 8).TopPos(qay, 10));
	int qainner = qay + 20;
	
	Add(lbl_baseline_json.SetLabel("bool_baseline.json").LeftPos(kInner, kLblW).TopPos(qainner, kRowH));
	Add(edit_baseline_json.HSizePos(kInner + kLblW + 8, kInnerR + 8).TopPos(qainner, kRowH));
	qainner += kRowH + kGap;

	Add(lbl_offset_baseline_json.SetLabel("offset_baseline.json").LeftPos(kInner, kLblW).TopPos(qainner, kRowH));
	Add(edit_offset_baseline_json.HSizePos(kInner + kLblW + 8, kInnerR + 8).TopPos(qainner, kRowH));
	qainner += kRowH + kGap;

	Add(lbl_split_json.SetLabel("split.json").LeftPos(kInner, kLblW).TopPos(qainner, kRowH));
	Add(edit_split_json.HSizePos(kInner + kLblW + 8, kInnerR + 8).TopPos(qainner, kRowH));
	qainner += kRowH + kGap;

	Add(lbl_qa_out_dir.SetLabel("report_out").LeftPos(kInner, kLblW).TopPos(qainner, kRowH));
	Add(edit_qa_out_dir.HSizePos(kInner + kLblW + 8, kInnerR + 8).TopPos(qainner, kRowH));
	qainner += kRowH + kGap + 4;

	Add(btn_audit_bool.SetLabel("Audit Bool Gates").LeftPos(kInner, 140).TopPos(qainner, kRowH));
	Add(btn_fix_bool.SetLabel("Fix Bool Gates").LeftPos(kInner + 150, 130).TopPos(qainner, kRowH));
	Add(btn_eval_bool.SetLabel("Eval Bool Gates").LeftPos(kInner + 150 + 140, 130).TopPos(qainner, kRowH));
	Add(btn_check_baseline.SetLabel("Check Bool Baseline").HSizePos(kInner + 150 + 140 + 140, kInnerR + 8).TopPos(qainner, kRowH));
	qainner += kRowH + kGap;

	Add(btn_refresh_baselines.SetLabel("Refresh Phase 6 Baselines").LeftPos(kInner, 200).TopPos(qainner, kRowH));
	Add(btn_run_preflight.SetLabel("Run Phase 6 Preflight").LeftPos(kInner + 210, 180).TopPos(qainner, kRowH));
	Add(btn_run_exit_gate.SetLabel("Run Phase 6 Exit Gate").HSizePos(kInner + 210 + 190, kInnerR + 8).TopPos(qainner, kRowH));
	qainner += kRowH + kGap;

	Add(opt_fail_on_warnings.SetLabel("Fail on Warnings").LeftPos(kInner, 150).TopPos(qainner, kRowH));
	Add(opt_fail_on_low_support.SetLabel("Fail on Low Support").LeftPos(kInner + 160, 150).TopPos(qainner, kRowH));
	Add(opt_allow_baseline_mismatch.SetLabel("Allow Baseline Mismatch").LeftPos(kInner + 320, 200).TopPos(qainner, kRowH));
	Add(opt_refresh_baselines.SetLabel("Refresh Baselines").LeftPos(kInner + 530, 150).TopPos(qainner, kRowH));
	opt_fail_on_low_support.Set(true);
	opt_refresh_baselines.Set(true);
	qainner += kRowH + kGap;

	Add(lbl_min_bool_support.SetLabel("Min Bool Support").LeftPos(kInner, kLblW).TopPos(qainner, kRowH));
	Add(edit_min_bool_support.LeftPos(kInner + kLblW + 8, 60).TopPos(qainner, kRowH));
	edit_min_bool_support.MinMax(0, 1000);
	edit_min_bool_support.SetData(0);
	qainner += kRowH + kGap;

	Add(lbl_qa_status.SetLabel("Ready").HSizePos(kInner, kInnerR).TopPos(qainner, kRowH));
	qainner += kRowH + 6;

	box_qa.HSizePos(kBoxX, 8).TopPos(qay, qainner - qay);
	y = qainner + 16;

	// Resize window to fit content
	SetRect(0, 0, 900, y + 12);

	// ── Initial values ───────────────────────────────────────────────────────
	edit_name.SetData("");
	edit_annprj.SetData("");
	edit_annlay.SetData("");
	edit_mlui.SetData("");
	edit_crops_dir.SetData("");
	edit_templates_dir.SetData("");
	edit_images_dir.SetData("");
	edit_train_epochs.SetData(50);

	// ── Wire up callbacks ─────────────────────────────────────────────────────
	btn_open << THISBACK(OnOpen);
	btn_save << THISBACK(OnSave);
	btn_new  << THISBACK(OnNew);

	btn_open_annotation_editor     << THISBACK(OnOpenAnnotationEditor);
	btn_open_annotation_review_p1  << THISBACK(OnOpenAnnotationEditor);
	btn_open_annotation_review_p2  << THISBACK(OnOpenAnnotationEditor);

	btn_export_pass1 << [=] { OnExportCrops(1); };
	btn_export_pass2 << [=] { OnExportCrops(2); };

	btn_recompute_p1 << THISBACK(OnRecomputeAnchors);

	btn_train_selected_p1 << [=] { OnTrainSelected(1); };
	btn_train_selected_p2 << [=] { OnTrainSelected(2); };
	btn_train_cli_p1      << [=] { OnTrainCli(1); };
	btn_train_cli_p2      << [=] { OnTrainCli(2); };

	arr_slots_p1.WhenSel = [=] {
		if(arr_slots_p1.GetCount() > 0) selected_slot_key_p1 = arr_slots_p1.Get("KEY");
	};
	arr_slots_p1.WhenLeftDouble = [=] { OnTrainSelected(1); };
	arr_slots_p2.WhenSel = [=] {
		if(arr_slots_p2.GetCount() > 0) selected_slot_key_p2 = arr_slots_p2.Get("KEY");
	};
	arr_slots_p2.WhenLeftDouble = [=] { OnTrainSelected(2); };

	btn_run_recognition_p1 << [=] { OnRunRecognition(); };
	btn_run_recognition_p2 << [=] { OnRunRecognition(); };
	btn_open_recognizer_p1 << [=] { OnOpenFrameRecognizer(1); };
	btn_open_recognizer_p2 << [=] { OnOpenFrameRecognizer(2); };

	btn_audit_bool << THISBACK(OnAuditBool);
	btn_fix_bool   << THISBACK(OnFixBool);
	btn_eval_bool << THISBACK(OnEvalBool);
	btn_check_baseline << THISBACK(OnCheckBaseline);
	btn_run_preflight << [=] { OnRunPreflight(false); };
	btn_refresh_baselines << THISBACK(OnRefreshBaselines);
	btn_run_exit_gate << THISBACK(OnRunExitGate);

	edit_annlay    << [=] { RefreshSlotGroups(); };
	edit_crops_dir << [=] { RefreshSlotGroups(); };
	edit_annprj    << [=] { RefreshVerifiedCount(); };

	RefreshSlotGroups();
	RefreshVerifiedCount();
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadFromControls / SaveToControls
// ─────────────────────────────────────────────────────────────────────────────


END_UPP_NAMESPACE
