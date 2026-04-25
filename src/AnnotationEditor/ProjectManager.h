#ifndef _AnnotationEditor_ProjectManager_h_
#define _AnnotationEditor_ProjectManager_h_

#include <CtrlLib/CtrlLib.h>
#include <AnnLayCore/AnnLay.h>
#include <AnnLayCore/AnnSln.h>
#include "../FrameRecognizer/FrameRecognizer.h"

NAMESPACE_UPP

class ProjectManager : public TopWindow {
public:
	typedef ProjectManager CLASSNAME;
	enum OpenRequestType {
		OPEN_NONE,
		OPEN_ANNOTATION_EDITOR,
		OPEN_SLOT_TRAINER,
	};

	ProjectManager();
	void OpenSln(const String& path);
	void DumpToStdout() const;
	OpenRequestType GetOpenRequestType() const { return open_request_type; }
	String GetRequestedAnnprj() const { return request_annprj; }
	String GetRequestedCropsDir() const { return request_crops_dir; }
	String GetRequestedAnnlay() const { return request_annlay; }
	String GetRequestedSlotKey() const { return request_slot_key; }
	String GetRequestedImagesDir() const { return request_images_dir; }
	String GetCurrentSlnPath() const { return current_sln_path; }

	void TestPreflight(const String& sln_path, const String& bool_baseline, const String& offset_baseline, const String& out_dir, const String& split = "", bool fail_on_warnings = false);

private:
	Label lbl_header;
	EditString edit_name;
	Button btn_open;
	Button btn_save;
	Button btn_new;

	Label lbl_annprj;
	EditString edit_annprj;
	Label lbl_annlay;
	EditString edit_annlay;
	Label lbl_mlui;
	EditString edit_mlui;
	Label lbl_crops_dir;
	EditString edit_crops_dir;
	Label lbl_templates_dir;
	EditString edit_templates_dir;
	Label lbl_images_dir;
	EditString edit_images_dir;
	Label lbl_train_epochs;
	EditInt edit_train_epochs;

	Label lbl_runtime_settings;
	Label lbl_offset_mode;
	DropList drop_offset_mode;
	Label lbl_bool_policy;
	DropList drop_bool_policy;

	// Pass 1
	LabelBox box_pass1;
	LabelBox box_p1_annotate;
	Button btn_open_annotation_editor;
	LabelBox box_p1_export;
	Button btn_export_pass1;
	Label lbl_export_pass1_status;
	LabelBox box_p1_train;
	ArrayCtrl arr_slots_p1;
	Button btn_train_selected_p1;
	Button btn_train_cli_p1;
	LabelBox box_p1_recompute;
	Button btn_recompute_p1;
	Label lbl_recompute_p1_status;
	LabelBox box_p1_recognize;
	Button btn_run_recognition_p1;
	Button btn_open_recognizer_p1;
	Label lbl_recognize_p1_status;
	LabelBox box_p1_review;
	Label lbl_verified_p1;
	Button btn_open_annotation_review_p1;

	// Pass 2
	LabelBox box_pass2;
	LabelBox box_p2_export;
	Button btn_export_pass2;
	Label lbl_export_pass2_status;
	LabelBox box_p2_train;
	ArrayCtrl arr_slots_p2;
	Button btn_train_selected_p2;
	Button btn_train_cli_p2;
	LabelBox box_p2_recognize;
	Button btn_run_recognition_p2;
	Button btn_open_recognizer_p2;
	Label lbl_recognize_p2_status;
	LabelBox box_p2_review;
	Label lbl_verified_p2;
	Button btn_open_annotation_review_p2;

	// QA / Validation
	LabelBox box_qa;
	Label lbl_baseline_json;
	EditString edit_baseline_json;
	Label lbl_offset_baseline_json;
	EditString edit_offset_baseline_json;
	Label lbl_split_json;
	EditString edit_split_json;
	Label lbl_qa_out_dir;
	EditString edit_qa_out_dir;
	Button btn_audit_bool;
	Button btn_fix_bool;
	Button btn_eval_bool;
	Button btn_check_baseline;
	Button btn_run_preflight;
	Button btn_refresh_baselines;
	Button btn_run_exit_gate;
	Option opt_fail_on_warnings;
	Option opt_allow_baseline_mismatch;
	Option opt_refresh_baselines;
	Label lbl_min_bool_support;
	EditInt edit_min_bool_support;
	Option opt_fail_on_low_support;
	Label lbl_qa_status;

	String current_sln_path;
	AnnSln current_sln_cfg;
	One<FrameRecognizerWindow> frame_recognizer_win_;
	OpenRequestType open_request_type = OPEN_NONE;
	String request_annprj;
	String request_crops_dir;
	String request_annlay;
	String request_slot_key;
	String request_images_dir;
	String selected_slot_key_p1;
	String selected_slot_key_p2;
	int    verified_count = 0;
	int    total_image_count = 0;
	int    recognize_progress = 0;

	void LoadFromControls(AnnSln& s) const;
	void SaveToControls(const AnnSln& s);
	bool LaunchSelf(const Vector<String>& args, bool close_window);

	void OnOpen();
	void OnSave();
	void OnSaveAs();
	void OnNew();

	void OnOpenAnnotationEditor();
	void RefreshSlotGroups();
	void OnTrainSelected(int pass_index);
	void OnTrainCli(int pass_index);
	void OnRecomputeAnchors();
	void OnExportCrops(int pass_index);
	void OnRunRecognition();
	void OnOpenFrameRecognizer(int pass_index);
	void OnAuditBool();
	void OnFixBool();
	void OnEvalBool();
	void OnCheckBaseline();
	void OnRunPreflight(bool headless = false);
	void OnRefreshBaselines();
	void OnRunExitGate();
	void RefreshVerifiedCount();
	void SetupSlotArrayCtrl(ArrayCtrl& arr);
	void PopulateSlotArrayCtrl(ArrayCtrl& arr, int pass_index);
	void DumpSlotArrayCtrl(const char* section, const ArrayCtrl& arr) const;
	bool RunCli(const Vector<String>& args, const String& action_name);
};

END_UPP_NAMESPACE

#endif
