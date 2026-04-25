#ifndef _AnnotationEditor_AppMainArgs_h_
#define _AnnotationEditor_AppMainArgs_h_

#include <Core/Core.h>
#include <AnnLayCore/AnchoredSlotRecognizer.h>
#include "AnchoredSlotExporter.h"

namespace Upp {

struct AppArguments {
	enum Mode {
		MODE_INVALID,
		MODE_PROJECT_MANAGER,
		MODE_PROJECT_MANAGER_WITH_SLN,
		MODE_SLOT_TRAINER_WINDOW,
		MODE_ANNOTATION_EDITOR_SINGLE_PROJECT,
		MODE_ANNOTATION_EDITOR_ARGS,
		MODE_TRAIN_HEADLESS,
		MODE_TRAIN_SLOTS,
		MODE_TRAIN_SLOTS_FROM_CROPS,
		MODE_RECOGNIZE_SLOTS,
		MODE_SMOKE_TEST,
		MODE_EXPORT_CROPS,
		MODE_EVAL_OFFSET_HEADS,
		MODE_AUDIT_OFFSET_HEADS,
		MODE_PRUNE_OFFSET_HEADS,
		MODE_EVAL_BOOL_GATES,
		MODE_AUDIT_BOOL_GATES,
		MODE_FIX_BOOL_GATES,
		MODE_TUNE_BOOL_GATES,
		MODE_MINE_BOOL_HARD_NEGATIVES,
		MODE_MERGE_HARDNEG,
		MODE_RECOMPUTE_ANCHORS,
		MODE_MAKE_EVAL_SPLIT,
		MODE_SAVE_BOOL_BASELINE,
		MODE_CHECK_BOOL_BASELINE,
		MODE_REPORT_BOOL_GATES,
		MODE_SAVE_OFFSET_BASELINE,
		MODE_CHECK_OFFSET_BASELINE,
		MODE_TEST_PREFLIGHT,
		MODE_RUN_PHASE6_PREFLIGHT,
		MODE_REFRESH_PHASE6_BASELINES,
		MODE_RUN_PHASE6_EXIT_GATE,
		MODE_TRACE_FRAME_LOG,
		MODE_REPORT_CARD_VISIBILITY_CONSISTENCY,
		MODE_TUNE_CARD_VISIBILITY_BY_CONSISTENCY,
		MODE_TUNE_CARD_JOINT_PIPELINE,
		MODE_SENTINEL_TEST_RB,
		MODE_PIPELINE_COMPLEATNESS_TEST,
		MODE_TEST_FR_INVARIANTS,
		MODE_TEST_XOFFSET_CONVLAYERS,
		MODE_EVAL_OCR_MODES,
		MODE_DUMP_NN_STEPS,
		MODE_FRAME_RECOGNIZER_WINDOW,
		MODE_FRAME_RECOGNIZER_DUMP_STEPS,
		MODE_FRAME_RECOGNIZER_DUMP_MEMORY_ALLOC,
		MODE_MIGRATE_ANNMDL_SESSIONS,
		MODE_MIGRATE_ANNMDL_V3,
		MODE_MERGE_MODES,
		MODE_DUMP_PROJECT_MANAGER,
		MODE_NORMALIZE_COMPOSITE_HEADS,
		MODE_TRAIN_ALL,
		MODE_DEBUG_GUI_TRAIN,
	};

	Mode mode = MODE_INVALID;
	String error;

	String joint_tune_sln_path;
	String joint_tune_out_path;
	bool   joint_tune_apply = false;

	String annprj_path;
	String annsln_path;
	String crops_dir;
	String annlay_path;
	String open_train_annprj_path;
	String open_train_images_dir;
	String open_train_slot_key;

	String train_annlay_path;
	String train_annprj_path;
	String train_images_dir;
	String train_card_strip;
	int    train_epochs = 15;

	String crops_annlay_path;
	String crops_crops_dir;
	int    crops_epochs = 50;
	String crops_slot_key;
	bool   crops_bool_balance_by_slot = true;
	int    crops_bool_slot_cap = 0;
	bool   crops_ignore_auto_stop = false;

	String recognize_annlay_path;
	String recognize_image_path;
	String smoke_sln_path;
	bool   smoke_all = false;
	int    smoke_count = 5;
	String export_sln_path;
	int    export_pass = 1;
	OffsetStyle offset_style_arg = OFFSET_STYLE_BOTH;

	String eval_sln_path;
	int    eval_count = 10;
	bool   eval_all = false;
	OffsetMode offset_mode_arg = OFFSET_AUTO;
	BoolGatePolicy bool_gate_policy_arg = BOOL_GATE_PERMISSIVE;

	String audit_annlay_path;
	String audit_out_path;
	bool   audit_strict = false;
	String prune_annlay_path;
	OffsetStyle prune_keep = OFFSET_STYLE_COMBINED;
	bool   prune_dry_run = false;

	String fix_bool_annlay_path;
	bool   fix_bool_dry_run = false;

	String recompute_annlay_path;
	String recompute_annprj_path;

	String make_split_sln_path;
	String make_split_out_path;
	double make_split_ratio = 0.2;
	int    make_split_seed = 42;

	String save_baseline_sln_path;
	String save_baseline_out_path;
	String check_baseline_sln_path;
	String check_baseline_in_path;
	String check_bool_out_path;
	String report_bool_out_path;

	String save_offset_baseline_out_path;
	String check_offset_baseline_in_path;
	String check_offset_out_path;
	double max_meanl1_rise = 0.50;
	double max_p90_rise = 1.00;

	double max_f1_drop = 0.05;
	double max_recall_drop = 0.05;
	double min_holdout_f1 = 0.85;
	int    min_bool_support = 0;
	bool   fail_on_low_support_arg = true;

	String split_file;
	String subset = "all";

	String tune_sln_path;
	String tune_group = "chip_box_gate";
	bool   tune_all = false;
	int    tune_count = 50;
	bool   tune_apply = false;
	bool   tune_per_slot = false;
	double tune_min_recall = 0.50;
	String tune_optimize = "f1";
	double tune_max_step = 0.05;

	String mine_sln_path;
	String mine_group;
	String mine_out_dir;
	bool   mine_all = false;
	int    mine_count = 100;
	double mine_threshold = -1.0;

	String merge_base_dir;
	String merge_hardneg_dir;
	bool   merge_dry_run = false;

	int    eval_min_positive = 0;
	bool fail_on_warnings_arg = false;
	bool allow_baseline_mismatch = true;
	bool exit_gate_refresh_baselines = true;
	String open_project;
	String trace_out_path;
	String trace_image_path;
	String consistency_sln_path;
	String consistency_out_path;
	String consistency_mismatch_out_path;
	int    consistency_mismatch_limit = 10;
	String sentinel_sln_path;
	String pipeline_test_sln_path;
	String test_fr_invariants_sln_path;
	String test_xoffset_sln_path;
	bool   test_xoffset_verbose = false;
	String eval_ocr_out_path;
	bool   eval_ocr_apply = false;
	double min_recog_gain = 0.0;
	double min_conf_gain = 0.0;
	int    open_fr_pass = 1;
	String dump_nn_slot = "board_card_1";
	String dump_nn_stage = "VISIBLE";

	String dump_steps_img_name;
	String dump_steps_sln_path;
	String dump_steps_model_set = "pass1";
	bool   frame_recognizer_dump_memory_alloc = false;

	String migrate_annmdl_sln_path;
	String migrate_annmdl_out_dir;
	String migrate_v3_sln_path;
	String migrate_v3_out_dir;

	String merge_src_path;
	String merge_dst_path;

	String normalize_composite_sln_path;
	bool   normalize_composite_dry_run = false;

	String train_headless_sln_path;
	String train_headless_crops_dir;
	int    train_headless_epochs = 50;
	String train_headless_slot_key;
	bool   train_headless_ignore_auto_stop = false;
	bool   train_headless_debug = false;

	String train_all_sln_path;
	int    train_all_pass = 1;
	int    train_all_epochs = 50;

	String debug_gui_train_group_key;
	String debug_gui_train_sln_path;
	int    debug_gui_train_epochs = 1;
	int    debug_gui_train_timeout_sec = 60;
	double debug_gui_train_loss_interval = 1.0;
	double debug_gui_train_pred_interval = 3.0;
	int    debug_gui_train_pred_samples = 10;

	bool Parse(const Vector<String>& cmdline);

private:
	bool ParseDumpProjectManager(const Vector<String>& args);
	bool ParseTrainHeadless(const Vector<String>& args);
	bool ParseOpenTrainPanel(const Vector<String>& args);
	bool ParseTrainSlots(const Vector<String>& args);
	bool ParseTrainSlotsFromCrops(const Vector<String>& args);
	bool ParseRecognizeSlots(const Vector<String>& args);
	bool ParseSmokeTest(const Vector<String>& args);
	bool ParseEvalOffsetHeads(const Vector<String>& args);
	bool ParseAuditOffsetHeads(const Vector<String>& args);
	bool ParsePruneOffsetHeads(const Vector<String>& args);
	bool ParseEvalBoolGates(const Vector<String>& args);
	bool ParseAuditBoolGates(const Vector<String>& args);
	bool ParseFixBoolGates(const Vector<String>& args);
	bool ParseMakeEvalSplit(const Vector<String>& args);
	bool ParseRecomputeAnchors(const Vector<String>& args);
	bool ParseTuneBoolGates(const Vector<String>& args);
	bool ParseMineBoolHardNegatives(const Vector<String>& args);
	bool ParseMergeHardneg(const Vector<String>& args);
	bool ParseExportCrops(const Vector<String>& args);
	bool ParseSaveBoolBaseline(const Vector<String>& args);
	bool ParseCheckBoolBaseline(const Vector<String>& args);
	bool ParseReportBoolGates(const Vector<String>& args);
	bool ParseSaveOffsetBaseline(const Vector<String>& args);
	bool ParseCheckOffsetBaseline(const Vector<String>& args);
	bool ParseTestPreflight(const Vector<String>& args);
	bool ParseRunPhase6Preflight(const Vector<String>& args);
	bool ParseRefreshPhase6Baselines(const Vector<String>& args);
	bool ParseRunPhase6ExitGate(const Vector<String>& args);
	bool ParseTraceFrameLog(const Vector<String>& args);
	bool ParseReportCardVisibilityConsistency(const Vector<String>& args);
	bool ParseTuneCardVisibilityByConsistency(const Vector<String>& args);
	bool ParseTuneCardJointPipeline(const Vector<String>& args);
	bool ParseSentinelTestRB(const Vector<String>& args);
	bool ParsePipelineCompletenessTest(const Vector<String>& args);
	bool ParseTestFRInvariants(const Vector<String>& args);
	bool ParseTestXOffsetConvLayers(const Vector<String>& args);
	bool ParseEvalOcrModes(const Vector<String>& args);
	bool ParseOpenFrameRecognizer(const Vector<String>& args);
	bool ParseDumpNNSteps(const Vector<String>& args);
	bool ParseFrameRecognizerDumpSteps(const Vector<String>& args);
	bool ParseFrameRecognizerDumpMemoryAlloc(const Vector<String>& args);
	bool ParseMigrateAnnmdlSessions(const Vector<String>& args);
	bool ParseMigrateAnnmdlV3(const Vector<String>& args);
	bool ParseMergeModes(const Vector<String>& args);
	bool ParseNormalizeCompositeHeads(const Vector<String>& args);
	bool ParseTrainAll(const Vector<String>& args);
	bool ParseDebugGuiTrain(const Vector<String>& args);
	bool ParseAnnotationEditorArgs(const Vector<String>& args);

	static bool HasExt(const String& path, const char* ext);
	static bool IsExistingFileWithExt(const String& path, const char* ext);
	static Vector<String> Tail(const Vector<String>& v, int begin);
	static OffsetMode ParseOffsetMode(const String& s);
	static BoolGatePolicy ParseBoolGatePolicy(const String& s);
	static OffsetStyle ParseOffsetStyle(const String& s);
};

class AnnotationEditorWindow;
void ApplyAnnotationEditorArguments(AnnotationEditorWindow& win, const AppArguments& args);

}

#endif
