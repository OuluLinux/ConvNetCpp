#include "AppMainArgs.h"

namespace Upp {

bool AppArguments::ParseTuneCardVisibilityByConsistency(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("count", 0, "Number of images to run", true, "n");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset to evaluate (train|holdout|all)", true, "name");
	cl.AddArg("apply", 0, "Apply best thresholds to .annlay", false);
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --tune-element-visibility-by-consistency <sln_path> [--apply] [--all] [--count N] [--split-file path] [--subset name] [--offset-mode mode]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_TUNE_CARD_VISIBILITY_BY_CONSISTENCY;
	consistency_sln_path = cl.GetPositional(0);
	tune_apply = cl.IsArg("apply");
	eval_all = cl.IsArg("all");
	if(cl.IsArg("count"))
		eval_count = max(1, StrInt(cl.GetArg("count")));
	if(cl.IsArg("split-file"))
		split_file = cl.GetArg("split-file");
	if(cl.IsArg("subset"))
		subset = cl.GetArg("subset");
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	return true;
}

bool AppArguments::ParseTuneCardJointPipeline(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("out", 0, "Output JSON report path", true, "path");
	cl.AddArg("apply", 0, "Write best thresholds to .annlay", false);
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("count", 0, "Number of images to run", true, "n");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset to run (train|holdout|all)", true, "name");
	cl.AddArg("offset-mode", 0, "Recognition offset mode (auto|none|combined|split)", true, "mode");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --tune-element-joint-pipeline <sln_path> [--out path] [--apply] [--all] [--count N] [--split-file path] [--subset name] [--offset-mode mode]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_TUNE_CARD_JOINT_PIPELINE;
	joint_tune_sln_path = cl.GetPositional(0);
	joint_tune_out_path = cl.GetArg("out");
	joint_tune_apply = cl.IsArg("apply");

	eval_all = cl.IsArg("all");
	if(cl.IsArg("count")) eval_count = StrInt(cl.GetArg("count"));
	split_file = cl.GetArg("split-file");
	subset = cl.GetArg("subset");
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));

	return true;
}

bool AppArguments::ParseSentinelTestRB(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --sentinel-test-rb <sln_path>";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_SENTINEL_TEST_RB;
	sentinel_sln_path = cl.GetPositional(0);
	return true;
}

bool AppArguments::ParsePipelineCompletenessTest(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --test-pipeline-completeness <sln_path>";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_PIPELINE_COMPLEATNESS_TEST;
	pipeline_test_sln_path = cl.GetPositional(0);
	return true;
}

bool AppArguments::ParseTestFRInvariants(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --test-fr-invariants <sln_path>";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_TEST_FR_INVARIANTS;
	sentinel_sln_path = cl.GetPositional(0); // reuse sentinel_sln_path
	return true;
}

bool AppArguments::ParseTestXOffsetConvLayers(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("verbose", 0, "Enable verbose output", false);
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --test-xoffset-convlayers <sln_path> [--verbose]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_TEST_XOFFSET_CONVLAYERS;
	test_xoffset_sln_path = cl.GetPositional(0);
	test_xoffset_verbose = cl.IsArg("verbose");
	return true;
}

bool AppArguments::ParseEvalOcrModes(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("out", 0, "Output JSON path", true, "path");
	cl.AddArg("apply", 0, "Apply best mode to .annlay", false);
	cl.AddArg("count", 0, "Max images to process", true, "N");
	cl.AddArg("min-recog-gain", 0, "Min recognition rate gain to apply", true, "float");
	cl.AddArg("min-conf-gain", 0, "Min confidence gain to apply", true, "float");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset to eval (train|holdout|all)", true, "name");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);

	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --eval-ocr-modes <sln_path> [--out <json>] [--apply] [--count <N>] [--min-recog-gain <f>] [--min-conf-gain <f>] [--split-file <path>] [--subset <name>]";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_EVAL_OCR_MODES;
	annsln_path = cl.GetPositional(0);
	eval_ocr_out_path = cl.GetArg("out");
	eval_ocr_apply = cl.IsArg("apply");
	if(cl.IsArg("count")) smoke_count = StrInt(cl.GetArg("count"));
	if(cl.IsArg("min-recog-gain")) min_recog_gain = ScanDouble(cl.GetArg("min-recog-gain"));
	if(cl.IsArg("min-conf-gain")) min_conf_gain = ScanDouble(cl.GetArg("min-conf-gain"));
	if(cl.IsArg("split-file")) split_file = cl.GetArg("split-file");
	if(cl.IsArg("subset")) subset = cl.GetArg("subset");

	return true;
}
bool AppArguments::ParseOpenFrameRecognizer(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("pass", 0, "Pass index (1 or 2)", true, "index");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --open-fr <sln_path> [--pass <1|2>]";
		mode = MODE_INVALID;
		return false;
	}
	
	mode = MODE_FRAME_RECOGNIZER_WINDOW;
	annsln_path = cl.GetPositional(0);
	if(cl.IsArg("pass")) open_fr_pass = StrInt(cl.GetArg("pass"));
	else open_fr_pass = 1;
	return true;
}

bool AppArguments::ParseDumpNNSteps(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("slot", 0, "Slot ID to dump (e.g. board_card_1)", true, "id");
	cl.AddArg("stage", 0, "Stage to dump (RECOGNIZE, LEVEL, CATEGORY)", true, "name");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --dump-nn-steps <sln_path> [--slot <id>] [--stage <name>]";
		mode = MODE_INVALID;
		return false;
	}
	
	mode = MODE_DUMP_NN_STEPS;
	annsln_path = cl.GetPositional(0);
	if(cl.IsArg("slot")) dump_nn_slot = cl.GetArg("slot");
	if(cl.IsArg("stage")) dump_nn_stage = cl.GetArg("stage");
	else dump_nn_stage = "RECOGNIZE";
	return true;
}

bool AppArguments::ParseFrameRecognizerDumpSteps(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("img_name", "Image name (e.g. Frame-0809.jpeg)", STRING_V);
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	cl.AddArg("model-set", 0, "Model set name (default: pass1)", true, "name");
	cl.AddArg("mem", 0, "Enable memory allocation dump", false);

	if(!cl.Parse(args) || cl.GetPositionalCount() < 2) {
		error = "Usage: --frame-recognizer-dump-steps <img_name> <sln_path> [--model-set <name>] [--mem]";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_FRAME_RECOGNIZER_DUMP_STEPS;
	dump_steps_img_name = cl.GetPositional(0);
	dump_steps_sln_path = cl.GetPositional(1);
	dump_steps_model_set = cl.IsArg("model-set") ? TrimBoth(cl.GetArg("model-set")) : "pass1";
	if(dump_steps_model_set.IsEmpty())
		dump_steps_model_set = "pass1";
	frame_recognizer_dump_memory_alloc = cl.IsArg("mem");
	return true;
}

bool AppArguments::ParseFrameRecognizerDumpMemoryAlloc(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	cl.AddArg("frame", 0, "Image filename to recognize after loading", true, "img_name");
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --frame-recognizer-dump-memory-alloc <sln_path> [--frame <img_name>]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_FRAME_RECOGNIZER_DUMP_MEMORY_ALLOC;
	annsln_path = cl.GetPositional(0);
	frame_recognizer_dump_memory_alloc = true;
	if(cl.IsArg("frame"))
		dump_steps_img_name = cl.GetArg("frame");
	return true;
}
bool AppArguments::ParseMigrateAnnmdlSessions(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("out-dir", 0, "Directory for external session blobs", true, "path");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --migrate-annmdl-sessions <sln_path> [--out-dir <path>]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_MIGRATE_ANNMDL_SESSIONS;
	migrate_annmdl_sln_path = cl.GetPositional(0);
	if(cl.IsArg("out-dir"))
		migrate_annmdl_out_dir = cl.GetArg("out-dir");
	return true;
}

bool AppArguments::ParseMigrateAnnmdlV3(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("out-dir", 0, "Directory for external blobs", true, "path");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --migrate-annmdl-v3 <sln_path> [--out-dir <path>]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_MIGRATE_ANNMDL_V3;
	migrate_v3_sln_path = cl.GetPositional(0);
	if(cl.IsArg("out-dir"))
		migrate_v3_out_dir = cl.GetArg("out-dir");
	return true;
}

bool AppArguments::ParseMergeModes(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("src_path", "Path to source .annmdl", STRING_V);
	cl.AddPositional("dst_path", "Path to destination .annmdl", STRING_V);

	if(!cl.Parse(args) || cl.GetPositionalCount() < 2) {
		error = "Usage: --merge-modes <src.annmdl> <dst.annmdl>";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_MERGE_MODES;
	merge_src_path = cl.GetPositional(0);
	merge_dst_path = cl.GetPositional(1);
	return true;
}

bool AppArguments::ParseNormalizeCompositeHeads(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("dry-run", 0, "Report what would change without modifying files", false);
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --normalize-composite-heads <sln_path> [--dry-run]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_NORMALIZE_COMPOSITE_HEADS;
	normalize_composite_sln_path = cl.GetPositional(0);
	normalize_composite_dry_run = cl.IsArg("dry-run");
	return true;
}

bool AppArguments::ParseTrainAll(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("pass", 0, "Pass index (1 or 2)", true, "n");
	cl.AddArg("epochs", 0, "Number of epochs", true, "n");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --train-all <sln_path.annsln> [--pass <1|2>] [--epochs <n>]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_TRAIN_ALL;
	train_all_sln_path = cl.GetPositional(0);
	train_all_pass = cl.IsArg("pass") ? StrInt(cl.GetArg("pass")) : 1;
	train_all_epochs = cl.IsArg("epochs") ? StrInt(cl.GetArg("epochs")) : 50;
	return true;
}

bool AppArguments::ParseDebugGuiTrain(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("epochs", 0, "Number of epochs", true, "n");
	cl.AddArg("timeout-sec", 0, "Maximum runtime in seconds", true, "n");
	cl.AddArg("loss-interval", 0, "Loss print interval in seconds", true, "sec");
	cl.AddArg("pred-interval", 0, "Prediction dump interval in seconds", true, "sec");
	cl.AddArg("pred-samples", 0, "Number of validation samples to print", true, "n");
	cl.AddPositional("group_key", "Bool group key", STRING_V);
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 2) {
		error = "Usage: --debug-gui-train <group_key> <sln_path.annsln> [--epochs N] [--timeout-sec T] [--loss-interval S] [--pred-interval S] [--pred-samples N]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_DEBUG_GUI_TRAIN;
	debug_gui_train_group_key = cl.GetPositional(0);
	debug_gui_train_sln_path = cl.GetPositional(1);
	debug_gui_train_epochs = cl.IsArg("epochs") ? max(1, StrInt(cl.GetArg("epochs"))) : 1;
	debug_gui_train_timeout_sec = cl.IsArg("timeout-sec") ? max(1, StrInt(cl.GetArg("timeout-sec"))) : 60;
	debug_gui_train_loss_interval = cl.IsArg("loss-interval") ? max(0.0, ScanDouble(cl.GetArg("loss-interval"))) : 1.0;
	debug_gui_train_pred_interval = cl.IsArg("pred-interval") ? max(0.0, ScanDouble(cl.GetArg("pred-interval"))) : 3.0;
	debug_gui_train_pred_samples = cl.IsArg("pred-samples") ? max(0, StrInt(cl.GetArg("pred-samples"))) : 10;
	return true;
}

}
