#include "AppMainArgs.h"
#include "AnnotationEditorWindow.h"

namespace Upp {

bool AppArguments::Parse(const Vector<String>& cmdline) {
	bool dump_project_manager = false;
	Vector<String> filtered_cmdline;
	filtered_cmdline.Reserve(cmdline.GetCount());
	for(int i = 0; i < cmdline.GetCount(); i++) {
		if(cmdline[i] == "--dump-project-manager")
			dump_project_manager = true;
		else
			filtered_cmdline.Add(cmdline[i]);
	}
	if(dump_project_manager)
		return ParseDumpProjectManager(filtered_cmdline);

	if(cmdline.GetCount() == 1 && IsExistingFileWithExt(cmdline[0], ".annprj")) {
		mode = MODE_ANNOTATION_EDITOR_SINGLE_PROJECT;
		annprj_path = cmdline[0];
		return true;
	}

	if(cmdline.GetCount() == 1 && HasExt(cmdline[0], ".annsln")) {
		mode = MODE_PROJECT_MANAGER_WITH_SLN;
		annsln_path = cmdline[0];
		return true;
	}

	if(cmdline.IsEmpty()) {
		mode = MODE_PROJECT_MANAGER;
		return true;
	}

	if(cmdline[0] == "--open-train-panel")
		return ParseOpenTrainPanel(Tail(cmdline, 1));
	if(cmdline[0] == "--train-headless")
		return ParseTrainHeadless(Tail(cmdline, 1));
	if(cmdline[0] == "--train-slots")
		return ParseTrainSlots(Tail(cmdline, 1));
	if(cmdline[0] == "--train-slots-from-crops")
		return ParseTrainSlotsFromCrops(Tail(cmdline, 1));
	if(cmdline[0] == "--recognize-slots")
		return ParseRecognizeSlots(Tail(cmdline, 1));
	if(cmdline[0] == "--smoke-test")
		return ParseSmokeTest(Tail(cmdline, 1));
	if(cmdline[0] == "--eval-offset-heads")
		return ParseEvalOffsetHeads(Tail(cmdline, 1));
	if(cmdline[0] == "--audit-offset-heads")
		return ParseAuditOffsetHeads(Tail(cmdline, 1));
	if(cmdline[0] == "--prune-offset-heads")
		return ParsePruneOffsetHeads(Tail(cmdline, 1));
	if(cmdline[0] == "--eval-bool-gates")
		return ParseEvalBoolGates(Tail(cmdline, 1));
	if(cmdline[0] == "--audit-bool-gates")
		return ParseAuditBoolGates(Tail(cmdline, 1));
	if(cmdline[0] == "--fix-bool-gates")
		return ParseFixBoolGates(Tail(cmdline, 1));
	if(cmdline[0] == "--make-eval-split")
		return ParseMakeEvalSplit(Tail(cmdline, 1));
	if(cmdline[0] == "--save-bool-baseline")
		return ParseSaveBoolBaseline(Tail(cmdline, 1));
	if(cmdline[0] == "--check-bool-baseline")
		return ParseCheckBoolBaseline(Tail(cmdline, 1));
	if(cmdline[0] == "--save-offset-baseline")
		return ParseSaveOffsetBaseline(Tail(cmdline, 1));
	if(cmdline[0] == "--check-offset-baseline")
		return ParseCheckOffsetBaseline(Tail(cmdline, 1));
	if(cmdline[0] == "--test-preflight")
		return ParseTestPreflight(Tail(cmdline, 1));
	if(cmdline[0] == "--run-phase6-preflight")
		return ParseRunPhase6Preflight(Tail(cmdline, 1));
	if(cmdline[0] == "--run-phase6-exit-gate")
		return ParseRunPhase6ExitGate(Tail(cmdline, 1));
	if(cmdline[0] == "--trace-frame-log")
		return ParseTraceFrameLog(Tail(cmdline, 1));
	if(cmdline[0] == "--refresh-phase6-baselines")
		return ParseRefreshPhase6Baselines(Tail(cmdline, 1));
	if(cmdline[0] == "--report-bool-gates")
		return ParseReportBoolGates(Tail(cmdline, 1));
	if(cmdline[0] == "--recompute-anchors")
		return ParseRecomputeAnchors(Tail(cmdline, 1));
	if(cmdline[0] == "--tune-bool-gates")
		return ParseTuneBoolGates(Tail(cmdline, 1));
	if(cmdline[0] == "--mine-bool-hard-negatives")
		return ParseMineBoolHardNegatives(Tail(cmdline, 1));
	if(cmdline[0] == "--merge-hardneg")
		return ParseMergeHardneg(Tail(cmdline, 1));
	if(cmdline[0] == "--tune-element-visibility-by-consistency")
		return ParseTuneCardVisibilityByConsistency(Tail(cmdline, 1));
	if(cmdline[0] == "--tune-element-joint-pipeline")
		return ParseTuneCardJointPipeline(Tail(cmdline, 1));
	if(cmdline[0] == "--report-element-visibility-consistency")
		return ParseReportCardVisibilityConsistency(Tail(cmdline, 1));
	if(cmdline[0] == "--sentinel-test-rb")
		return ParseSentinelTestRB(Tail(cmdline, 1));
	if(cmdline[0] == "--test-pipeline-completeness")
		return ParsePipelineCompletenessTest(Tail(cmdline, 1));
	if(cmdline[0] == "--test-fr-invariants")
		return ParseTestFRInvariants(Tail(cmdline, 1));
	if(cmdline[0] == "--test-xoffset-convlayers")
		return ParseTestXOffsetConvLayers(Tail(cmdline, 1));
	if(cmdline[0] == "--eval-ocr-modes")
		return ParseEvalOcrModes(Tail(cmdline, 1));
	if(cmdline[0] == "--open-fr")
		return ParseOpenFrameRecognizer(Tail(cmdline, 1));
	if(cmdline[0] == "--frame-recognizer-dump-steps")
		return ParseFrameRecognizerDumpSteps(Tail(cmdline, 1));
	if(cmdline[0] == "--frame-recognizer-dump-memory-alloc")
		return ParseFrameRecognizerDumpMemoryAlloc(Tail(cmdline, 1));
	if(cmdline[0] == "--migrate-annmdl-sessions")
		return ParseMigrateAnnmdlSessions(Tail(cmdline, 1));
	if(cmdline[0] == "--migrate-annmdl-v3")
		return ParseMigrateAnnmdlV3(Tail(cmdline, 1));
	if(cmdline[0] == "--merge-modes")
		return ParseMergeModes(Tail(cmdline, 1));
	if(cmdline[0] == "--normalize-composite-heads")
		return ParseNormalizeCompositeHeads(Tail(cmdline, 1));
	if(cmdline[0] == "--train-all")
		return ParseTrainAll(Tail(cmdline, 1));
	if(cmdline[0] == "--debug-gui-train")
		return ParseDebugGuiTrain(Tail(cmdline, 1));
	if(cmdline[0] == "--dump-nn-steps")
		return ParseDumpNNSteps(Tail(cmdline, 1));
	if(cmdline[0] == "--export-crops")
		return ParseExportCrops(Tail(cmdline, 1));

	return ParseAnnotationEditorArgs(cmdline);
}

bool AppArguments::ParseDumpProjectManager(const Vector<String>& args) {
	if(args.GetCount() > 1) {
		error = "Usage: --dump-project-manager [sln_path.annsln]";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_DUMP_PROJECT_MANAGER;
	annsln_path.Clear();
	if(args.GetCount() == 1)
		annsln_path = args[0];
	return true;
}

bool AppArguments::ParseReportCardVisibilityConsistency(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("count", 0, "Number of images to run", true, "n");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset to evaluate (train|holdout|all)", true, "name");
	cl.AddArg("out", 0, "Output report JSON path", true, "path");
	cl.AddArg("mismatch-out", 0, "Output mismatches JSON path", true, "path");
	cl.AddArg("mismatch-limit", 0, "Cap per mismatch type", true, "n");
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("out")) {
		error = "Usage: --report-element-visibility-consistency <sln_path> --out <report.json> [--mismatch-out path] [--mismatch-limit N] [--all] [--count N] [--split-file path] [--subset name] [--offset-mode mode] [--bool-gate-policy strict|permissive]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_REPORT_CARD_VISIBILITY_CONSISTENCY;
	consistency_sln_path = cl.GetPositional(0);
	consistency_out_path = cl.GetArg("out");
	if(cl.IsArg("mismatch-out"))
		consistency_mismatch_out_path = cl.GetArg("mismatch-out");
	if(cl.IsArg("mismatch-limit"))
		consistency_mismatch_limit = StrInt(cl.GetArg("mismatch-limit"));
	eval_all = cl.IsArg("all");
	if(cl.IsArg("count"))
		eval_count = max(1, StrInt(cl.GetArg("count")));
	if(cl.IsArg("split-file"))
		split_file = cl.GetArg("split-file");
	if(cl.IsArg("subset"))
		subset = cl.GetArg("subset");
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	return true;
}

bool AppArguments::HasExt(const String& path, const char* ext) {
	return ToLower(GetFileExt(path)) == ext;
}

bool AppArguments::IsExistingFileWithExt(const String& path, const char* ext) {
	return HasExt(path, ext) && FileExists(path);
}

Vector<String> AppArguments::Tail(const Vector<String>& v, int begin) {
	Vector<String> out;
	for(int i = begin; i < v.GetCount(); i++)
		out.Add(v[i]);
	return out;
}

bool AppArguments::ParseTrainHeadless(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	cl.AddPositional("crops_dir", "Path to crops directory", STRING_V);
	cl.AddArg("epochs", 1, "Number of epochs (default 50)", false, "count");
	cl.AddArg("slot", 1, "Only train specific slot or group", false, "key");
	cl.AddArg("ignore-auto-stop", 0, "Don't early-stop even if accuracy is 100%", false);
	cl.AddArg("debug-train", 0, "Print detailed network state during training", false);

	if(!cl.Parse(args) || cl.GetPositionalCount() < 2) {
		error = "Usage: --train-headless <sln_path> <crops_dir> [--epochs <n>] [--slot <key>] [--ignore-auto-stop] [--debug-train]";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_TRAIN_HEADLESS;
	train_headless_sln_path = (String)cl.GetPositional(0);
	train_headless_crops_dir = (String)cl.GetPositional(1);
	String ep = cl.GetArg("epochs");
	train_headless_epochs = ep.IsEmpty() ? 50 : atoi(ep);
	train_headless_slot_key = cl.GetArg("slot");
	train_headless_ignore_auto_stop = cl.IsArg("ignore-auto-stop");
	train_headless_debug = cl.IsArg("debug-train");

	return true;
}

bool AppArguments::ParseOpenTrainPanel(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("crops-dir", 0, "Crops directory", true, "path");
	cl.AddArg("annlay", 0, "Annlay path", true, "path");
	cl.AddArg("annprj", 0, "Annprj path", true, "path");
	cl.AddArg("images-dir", 0, "Images directory", true, "path");
	cl.AddArg("slot", 0, "Slot id or group key", true, "id");
	if(!cl.Parse(args)) {
		error = "Usage: --open-train-panel [--crops-dir <path>] [--annlay <path>] [--annprj <path>] [--images-dir <path>] [--slot <key>]";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_SLOT_TRAINER_WINDOW;
	crops_dir = cl.GetArg("crops-dir");
	annlay_path = cl.GetArg("annlay");
	open_train_annprj_path = cl.GetArg("annprj");
	open_train_images_dir = cl.GetArg("images-dir");
	open_train_slot_key = cl.GetArg("slot");
	return true;
}

bool AppArguments::ParseTrainSlots(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("epochs", 0, "Number of training epochs", true, "n");
	cl.AddPositional("annlay_path", "Path to annlay", STRING_V);
	cl.AddPositional("annprj_path", "Path to annprj", STRING_V);
	cl.AddPositional("images_dir", "Path to images directory", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 3) {
		error = "Usage: --train-slots <annlay_path> <annprj_path> <images_dir> [--epochs <n>] [<card_strip_path>]";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_TRAIN_SLOTS;
	train_annlay_path = cl.GetPositional(0);
	train_annprj_path = cl.GetPositional(1);
	train_images_dir = cl.GetPositional(2);
	train_epochs = 15;
	train_card_strip.Clear();
	if(cl.IsArg("epochs"))
		train_epochs = StrInt(cl.GetArg("epochs"));

	Vector<String> rest = cl.GetRest();
	for(int i = 0; i < rest.GetCount(); i++) {
		if(rest[i] == "--epochs" && i + 1 < rest.GetCount()) {
			train_epochs = StrInt(rest[++i]);
		}
		else if(train_card_strip.IsEmpty()) {
			train_card_strip = rest[i];
		}
	}
	return true;
}

bool AppArguments::ParseTrainSlotsFromCrops(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("epochs", 0, "Number of training epochs", true, "n");
	cl.AddArg("slot", 0, "Slot id or group key", true, "id");
	cl.AddArg("bool-balance-by-slot", 0, "Enable per-slot balancing for bool groups (0|1)", true, "val");
	cl.AddArg("bool-slot-cap", 0, "Override per-slot sample cap", true, "n");
	cl.AddArg("ignore-auto-stop", 0, "Ignore loss-based early stop", false);
	cl.AddPositional("annlay_path", "Path to annlay", STRING_V);
	cl.AddPositional("crops_dir", "Path to crops directory", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 2) {
		error = "Usage: --train-slots-from-crops <annlay_path> <crops_dir> [--epochs N] [--slot <key>] [--bool-balance-by-slot 0|1] [--bool-slot-cap N] [--ignore-auto-stop]";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_TRAIN_SLOTS_FROM_CROPS;
	crops_annlay_path = cl.GetPositional(0);
	crops_crops_dir = cl.GetPositional(1);
	crops_epochs = 50;
	crops_slot_key.Clear();
	crops_ignore_auto_stop = cl.IsArg("ignore-auto-stop");
	if(cl.IsArg("epochs"))
		crops_epochs = max(1, StrInt(cl.GetArg("epochs")));
	if(cl.IsArg("slot"))
		crops_slot_key = cl.GetArg("slot");
	if(cl.IsArg("bool-balance-by-slot"))
		crops_bool_balance_by_slot = StrInt(cl.GetArg("bool-balance-by-slot")) != 0;
	if(cl.IsArg("bool-slot-cap"))
		crops_bool_slot_cap = max(0, StrInt(cl.GetArg("bool-slot-cap")));

	Vector<String> rest = cl.GetRest();
	for(int i = 0; i < rest.GetCount(); i++) {
		if(rest[i] == "--epochs" && i + 1 < rest.GetCount())
			crops_epochs = max(1, StrInt(rest[++i]));
		else if(rest[i] == "--slot" && i + 1 < rest.GetCount())
			crops_slot_key = rest[++i];
		else if(rest[i] == "--bool-balance-by-slot" && i + 1 < rest.GetCount())
			crops_bool_balance_by_slot = StrInt(rest[++i]) != 0;
		else if(rest[i] == "--bool-slot-cap" && i + 1 < rest.GetCount())
			crops_bool_slot_cap = max(0, StrInt(rest[++i]));
		else if(rest[i] == "--ignore-auto-stop")
			crops_ignore_auto_stop = true;
	}
	return true;
}

bool AppArguments::ParseRecognizeSlots(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddPositional("annlay_path", "Path to annlay", STRING_V);
	cl.AddPositional("image_path", "Path to image", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 2) {
		error = "Usage: --recognize-slots <annlay_path> <image_path> [--offset-mode mode] [--bool-gate-policy strict|permissive]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_RECOGNIZE_SLOTS;
	recognize_annlay_path = cl.GetPositional(0);
	recognize_image_path = cl.GetPositional(1);
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	return true;
}

OffsetMode AppArguments::ParseOffsetMode(const String& s) {
	String t = ToLower(TrimBoth(s));
	if(t == "none") return OFFSET_NONE;
	if(t == "combined") return OFFSET_COMBINED;
	if(t == "split") return OFFSET_SPLIT;
	return OFFSET_AUTO;
}

BoolGatePolicy AppArguments::ParseBoolGatePolicy(const String& s) {
	String t = ToLower(TrimBoth(s));
	if(t == "strict") return BOOL_GATE_STRICT;
	return BOOL_GATE_PERMISSIVE;
}

bool AppArguments::ParseSmokeTest(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("count", 0, "Number of images to run", true, "n");
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --smoke-test <sln_path> [--all] [--count N] [--offset-mode mode] [--bool-gate-policy strict|permissive]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_SMOKE_TEST;
	smoke_sln_path = cl.GetPositional(0);
	smoke_all = cl.IsArg("all");
	if(cl.IsArg("count"))
		smoke_count = max(1, StrInt(cl.GetArg("count")));
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	return true;
}

bool AppArguments::ParseEvalOffsetHeads(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("count", 0, "Number of images to run", true, "n");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset to evaluate (train|holdout|all)", true, "name");
	cl.AddArg("out", 0, "Output report JSON path", true, "path");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --eval-offset-heads <sln_path> [--all] [--count N] [--split-file path] [--subset name] [--out report.json]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_EVAL_OFFSET_HEADS;
	eval_sln_path = cl.GetPositional(0);
	eval_all = cl.IsArg("all");
	if(cl.IsArg("count"))
		eval_count = max(1, StrInt(cl.GetArg("count")));
	if(cl.IsArg("split-file"))
		split_file = cl.GetArg("split-file");
	if(cl.IsArg("subset"))
		subset = cl.GetArg("subset");
	if(cl.IsArg("out"))
		report_bool_out_path = cl.GetArg("out"); // reuse field
	return true;
}

bool AppArguments::ParseAuditOffsetHeads(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("annlay_path", "Path to .annlay", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --audit-offset-heads <annlay_path>";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_AUDIT_OFFSET_HEADS;
	audit_annlay_path = cl.GetPositional(0);
	return true;
}

bool AppArguments::ParsePruneOffsetHeads(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("keep", 0, "Offset style to keep (combined, split, both)", true, "style");
	cl.AddArg("dry-run", 0, "Print what would be removed without writing", false);
	cl.AddPositional("annlay_path", "Path to .annlay", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --prune-offset-heads <annlay_path> --keep <combined|split|both> [--dry-run]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_PRUNE_OFFSET_HEADS;
	prune_annlay_path = cl.GetPositional(0);
	prune_keep = cl.IsArg("keep") ? ParseOffsetStyle(cl.GetArg("keep")) : OFFSET_STYLE_BOTH;
	prune_dry_run = cl.IsArg("dry-run");
	return true;
}

OffsetStyle AppArguments::ParseOffsetStyle(const String& s) {
	String v = ToLower(s);
	if(v == "split")    return OFFSET_STYLE_SPLIT;
	if(v == "both")     return OFFSET_STYLE_BOTH;
	return OFFSET_STYLE_COMBINED;
}

bool AppArguments::ParseEvalBoolGates(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("count", 0, "Number of images to run", true, "n");
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddArg("min-positive", 0, "Minimum positive samples required", true, "n");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset to evaluate (train|holdout|all)", true, "name");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --eval-bool-gates <sln_path> [--all] [--count N] [--offset-mode mode] [--min-positive N] [--split-file path] [--subset name] [--bool-gate-policy strict|permissive]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_EVAL_BOOL_GATES;
	eval_sln_path = cl.GetPositional(0);
	eval_all = cl.IsArg("all");
	if(cl.IsArg("count"))
		eval_count = max(1, StrInt(cl.GetArg("count")));
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	if(cl.IsArg("min-positive"))
		eval_min_positive = max(0, StrInt(cl.GetArg("min-positive")));
	if(cl.IsArg("split-file"))
		split_file = cl.GetArg("split-file");
	if(cl.IsArg("subset"))
		subset = cl.GetArg("subset");
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	return true;
}

bool AppArguments::ParseFixBoolGates(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("dry-run", 0, "Print changes without saving", false);
	cl.AddPositional("annlay_path", "Path to .annlay", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --fix-bool-gates <annlay_path> [--dry-run]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_FIX_BOOL_GATES;
	fix_bool_annlay_path = cl.GetPositional(0);
	fix_bool_dry_run = cl.IsArg("dry-run");
	return true;
}

bool AppArguments::ParseMakeEvalSplit(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("out", 0, "Output JSON path", true, "path");
	cl.AddArg("holdout-ratio", 0, "Holdout ratio (0.0-1.0)", true, "R");
	cl.AddArg("seed", 0, "Random seed", true, "N");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("out")) {
		error = "Usage: --make-eval-split <sln_path> --out <json_path> [--holdout-ratio R] [--seed N]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_MAKE_EVAL_SPLIT;
	make_split_sln_path = cl.GetPositional(0);
	make_split_out_path = cl.GetArg("out");
	if(cl.IsArg("holdout-ratio"))
		make_split_ratio = ScanDouble(cl.GetArg("holdout-ratio"));
	if(cl.IsArg("seed"))
		make_split_seed = StrInt(cl.GetArg("seed"));
	return true;
}

bool AppArguments::ParseRecomputeAnchors(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddPositional("annlay_path", "Path to .annlay", STRING_V);
	cl.AddPositional("annprj_path", "Path to .annprj", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 2) {
		error = "Usage: --recompute-anchors <annlay_path> <annprj_path>";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_RECOMPUTE_ANCHORS;
	recompute_annlay_path = cl.GetPositional(0);
	recompute_annprj_path = cl.GetPositional(1);
	return true;
}

bool AppArguments::ParseTuneBoolGates(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("count", 0, "Number of images to run", true, "n");
	cl.AddArg("group", 0, "Slot group to tune", true, "name");
	cl.AddArg("apply", 0, "Write best threshold to .annlay", false);
	cl.AddArg("per-slot", 0, "Tune threshold per slot", false);
	cl.AddArg("min-recall", 0, "Minimum recall constraint", true, "R");
	cl.AddArg("optimize", 0, "Optimization objective (f1, f05, precision)", true, "obj");
	cl.AddArg("max-threshold-step", 0, "Max threshold step", true, "S");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset to tune (train|holdout|all)", true, "name");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --tune-bool-gates <sln_path> [--group name] [--all] [--count N] [--apply] [--per-slot] [--min-recall R] [--optimize f1|f05|precision] [--split-file path] [--subset name]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_TUNE_BOOL_GATES;
	tune_sln_path = cl.GetPositional(0);
	tune_all = cl.IsArg("all");
	if(cl.IsArg("count"))
		tune_count = max(1, StrInt(cl.GetArg("count")));
	if(cl.IsArg("group"))
		tune_group = cl.GetArg("group");
	tune_apply = cl.IsArg("apply");
	tune_per_slot = cl.IsArg("per-slot");
	if(cl.IsArg("min-recall"))
		tune_min_recall = ScanDouble(cl.GetArg("min-recall"));
	if(cl.IsArg("optimize"))
		tune_optimize = cl.GetArg("optimize");
	if(cl.IsArg("max-threshold-step"))
		tune_max_step = ScanDouble(cl.GetArg("max-threshold-step"));
	if(cl.IsArg("split-file"))
		split_file = cl.GetArg("split-file");
	if(cl.IsArg("subset"))
		subset = cl.GetArg("subset");
	return true;
}

bool AppArguments::ParseMineBoolHardNegatives(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("count", 0, "Number of images to run", true, "n");
	cl.AddArg("group", 0, "Slot group to mine", true, "name");
	cl.AddArg("out", 0, "Output directory for crops", true, "dir");
	cl.AddArg("threshold", 0, "Override threshold", true, "T");
	cl.AddArg("offset-mode", 0, "Offset mode", true, "mode");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --mine-bool-hard-negatives <sln_path> --group name --out dir [--all] [--count N] [--threshold T] [--offset-mode mode]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_MINE_BOOL_HARD_NEGATIVES;
	mine_sln_path = cl.GetPositional(0);
	mine_all = cl.IsArg("all");
	if(cl.IsArg("count"))
		mine_count = max(1, StrInt(cl.GetArg("count")));
	if(cl.IsArg("group"))
		mine_group = cl.GetArg("group");
	if(cl.IsArg("out"))
		mine_out_dir = cl.GetArg("out");
	if(cl.IsArg("threshold"))
		mine_threshold = ScanDouble(cl.GetArg("threshold"));
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	
	if(mine_group.IsEmpty() || mine_out_dir.IsEmpty()) {
		error = "--mine-bool-hard-negatives: --group and --out are required";
		mode = MODE_INVALID;
		return false;
	}
	return true;
}

bool AppArguments::ParseMergeHardneg(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("dry-run", 0, "Dry run", false);
	cl.AddPositional("base_dir", "Base group directory", STRING_V);
	cl.AddPositional("hardneg_dir", "Hardneg directory", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 2) {
		error = "Usage: --merge-hardneg <base_group_dir> <hardneg_dir> [--dry-run]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_MERGE_HARDNEG;
	merge_base_dir = cl.GetPositional(0);
	merge_hardneg_dir = cl.GetPositional(1);
	merge_dry_run = cl.IsArg("dry-run");
	return true;
}

bool AppArguments::ParseAuditBoolGates(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("strict", 0, "Fail on errors (non-zero exit)", false);
	cl.AddArg("out", 0, "Output audit report JSON", true, "path");
	cl.AddPositional("annlay_path", "Path to .annlay", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --audit-bool-gates <annlay_path> [--strict] [--out <report.json>]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_AUDIT_BOOL_GATES;
	audit_annlay_path = cl.GetPositional(0);
	audit_strict = cl.IsArg("strict");
	if(cl.IsArg("out"))
		audit_out_path = cl.GetArg("out");
	return true;
}

bool AppArguments::ParseExportCrops(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("pass", 0, "Export pass index (1 or 2)", true, "n");
	cl.AddArg("offset-heads", 0, "Offset head style (combined, split, both)", true, "style");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1) {
		error = "Usage: --export-crops <sln_path> [--pass 1|2] [--offset-heads style]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_EXPORT_CROPS;
	export_sln_path = cl.GetPositional(0);
	export_pass = 1;
	if(cl.IsArg("pass"))
		export_pass = StrInt(cl.GetArg("pass"));
	if(export_pass != 1 && export_pass != 2) {
		error = "--export-crops: --pass must be 1 or 2";
		mode = MODE_INVALID;
		return false;
	}
	if(cl.IsArg("offset-heads"))
		offset_style_arg = ParseOffsetStyle(cl.GetArg("offset-heads"));
	return true;
}

bool AppArguments::ParseSaveBoolBaseline(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("out", 0, "Output baseline JSON path", true, "path");
	cl.AddArg("group", 0, "Slot group", true, "name");
	cl.AddArg("all", 0, "Run all images", false);
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("out")) {
		error = "Usage: --save-bool-baseline <sln_path> --out <baseline.json> [--split-file path] [--group name] [--all] [--bool-gate-policy strict|permissive]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_SAVE_BOOL_BASELINE;
	save_baseline_sln_path = cl.GetPositional(0);
	save_baseline_out_path = cl.GetArg("out");
	if(cl.IsArg("split-file")) split_file = cl.GetArg("split-file");
	if(cl.IsArg("group")) tune_group = cl.GetArg("group");
	eval_all = cl.IsArg("all");
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	return true;
}

bool AppArguments::ParseCheckBoolBaseline(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("baseline", 0, "Baseline JSON to compare against", true, "path");
	cl.AddArg("group", 0, "Slot group", true, "name");
	cl.AddArg("max-f1-drop", 0, "Max allowed F1 drop", true, "D");
	cl.AddArg("max-recall-drop", 0, "Max allowed recall drop", true, "D");
	cl.AddArg("min-holdout-f1", 0, "Min allowed holdout F1", true, "V");
	cl.AddArg("min-bool-support", 0, "Minimum positive support required per slot", true, "N");
	cl.AddArg("fail-on-low-support", 0, "Fail if any slot has low support (0|1)", true, "val");
	cl.AddArg("check-out", 0, "Output check report JSON path", true, "path");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddArg("allow-baseline-mismatch", 0, "Allow mismatch in baseline provenance", false);
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("baseline")) {
		error = "Usage: --check-bool-baseline <sln_path> --baseline <baseline.json> [--split-file path] [--group name] [--check-out path] [--bool-gate-policy strict|permissive] [--offset-mode mode] [--allow-baseline-mismatch]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_CHECK_BOOL_BASELINE;
	check_baseline_sln_path = cl.GetPositional(0);
	check_baseline_in_path = cl.GetArg("baseline");
	if(cl.IsArg("split-file")) split_file = cl.GetArg("split-file");
	if(cl.IsArg("group")) tune_group = cl.GetArg("group");
	if(cl.IsArg("max-f1-drop")) max_f1_drop = ScanDouble(cl.GetArg("max-f1-drop"));
	if(cl.IsArg("max-recall-drop")) max_recall_drop = ScanDouble(cl.GetArg("max-recall-drop"));
	if(cl.IsArg("min-holdout-f1")) min_holdout_f1 = ScanDouble(cl.GetArg("min-holdout-f1"));
	if(cl.IsArg("min-bool-support")) min_bool_support = StrInt(cl.GetArg("min-bool-support"));
	if(cl.IsArg("fail-on-low-support")) fail_on_low_support_arg = (cl.GetArg("fail-on-low-support") == "1");
	if(cl.IsArg("check-out")) check_bool_out_path = cl.GetArg("check-out");
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	if(cl.IsArg("allow-baseline-mismatch"))
		allow_baseline_mismatch = true;
	eval_all = true; // Always evaluate all for regression check
	return true;
}

bool AppArguments::ParseReportBoolGates(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("out", 0, "Output report JSON path", true, "path");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("out")) {
		error = "Usage: --report-bool-gates <sln_path> --out <report.json> [--split-file path]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_REPORT_BOOL_GATES;
	save_baseline_sln_path = cl.GetPositional(0); // reuse
	report_bool_out_path = cl.GetArg("out");
	if(cl.IsArg("split-file")) split_file = cl.GetArg("split-file");
	eval_all = true;
	return true;
}

bool AppArguments::ParseSaveOffsetBaseline(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset (holdout|all)", true, "name");
	cl.AddArg("out", 0, "Output baseline JSON path", true, "path");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("out")) {
		error = "Usage: --save-offset-baseline <sln_path> --out <baseline.json> [--split-file path] [--subset holdout|all]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_SAVE_OFFSET_BASELINE;
	save_baseline_sln_path = cl.GetPositional(0); // reuse
	save_offset_baseline_out_path = cl.GetArg("out");
	if(cl.IsArg("split-file")) split_file = cl.GetArg("split-file");
	if(cl.IsArg("subset")) subset = cl.GetArg("subset");
	eval_all = true;
	return true;
}

bool AppArguments::ParseCheckOffsetBaseline(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("subset", 0, "Subset (holdout|all)", true, "name");
	cl.AddArg("baseline", 0, "Baseline JSON to compare against", true, "path");
	cl.AddArg("max-meanl1-rise", 0, "Max allowed MeanL1 rise", true, "D");
	cl.AddArg("max-p90-rise", 0, "Max allowed P90L1 rise", true, "D");
	cl.AddArg("check-out", 0, "Output check report JSON path", true, "path");
	cl.AddArg("allow-baseline-mismatch", 0, "Allow mismatch in baseline provenance", false);
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("baseline")) {
		error = "Usage: --check-offset-baseline <sln_path> --baseline <baseline.json> [--split-file path] [--subset name] [--allow-baseline-mismatch] ...";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_CHECK_OFFSET_BASELINE;
	check_baseline_sln_path = cl.GetPositional(0); // reuse
	check_offset_baseline_in_path = cl.GetArg("baseline");
	if(cl.IsArg("split-file")) split_file = cl.GetArg("split-file");
	if(cl.IsArg("subset")) subset = cl.GetArg("subset");
	if(cl.IsArg("max-meanl1-rise")) max_meanl1_rise = ScanDouble(cl.GetArg("max-meanl1-rise"));
	if(cl.IsArg("max-p90-rise")) max_p90_rise = ScanDouble(cl.GetArg("max-p90-rise"));
	if(cl.IsArg("check-out")) check_offset_out_path = cl.GetArg("check-out");
	if(cl.IsArg("allow-baseline-mismatch"))
		allow_baseline_mismatch = true;
	eval_all = true;
	return true;
}

bool AppArguments::ParseTestPreflight(const Vector<String>& args) {
	if(args.GetCount() < 4) {
		error = "Usage: --test-preflight <sln> <bool_base> <offset_base> <out_dir> [<split>] [fail_on_warnings:0|1] [allow_mismatch:0|1]";
		mode = MODE_INVALID;
		return false;
	}
	mode = MODE_TEST_PREFLIGHT;
	annsln_path = args[0];
	check_baseline_in_path = args[1];
	check_offset_baseline_in_path = args[2];
	mine_out_dir = args[3]; // reuse
	if(args.GetCount() > 4)
		split_file = args[4];
	if(args.GetCount() > 5)
		fail_on_warnings_arg = (args[5] == "1");
	if(args.GetCount() > 6)
		allow_baseline_mismatch = (args[6] == "1");
	return true;
}

bool AppArguments::ParseRunPhase6Preflight(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("bool-baseline", 0, "Path to bool baseline JSON", true, "path");
	cl.AddArg("offset-baseline", 0, "Path to offset baseline JSON", true, "path");
	cl.AddArg("out-dir", 0, "Output directory for reports", true, "dir");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddArg("fail-on-warnings", 0, "Fail on audit warnings (0|1)", true, "val");
	cl.AddArg("min-bool-support", 0, "Minimum positive support required per slot", true, "N");
	cl.AddArg("fail-on-low-support", 0, "Fail if any slot has low support (0|1)", true, "val");
	cl.AddArg("allow-baseline-mismatch", 0, "Allow mismatch in baseline provenance", false);
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("bool-baseline") || !cl.IsArg("offset-baseline") || !cl.IsArg("out-dir")) {
		error = "Usage: --run-phase6-preflight <sln_path> --bool-baseline <path> --offset-baseline <path> --out-dir <dir> [--split-file <path>] [--offset-mode mode] [--fail-on-warnings 0|1] [--allow-baseline-mismatch]";
		mode = MODE_INVALID;
		return false;
	}
	
	mode = MODE_RUN_PHASE6_PREFLIGHT;
	annsln_path = cl.GetPositional(0);
	check_baseline_in_path = cl.GetArg("bool-baseline");
	check_offset_baseline_in_path = cl.GetArg("offset-baseline");
	mine_out_dir = cl.GetArg("out-dir");
	
	if(cl.IsArg("split-file"))
		split_file = cl.GetArg("split-file");
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	if(cl.IsArg("fail-on-warnings"))
		fail_on_warnings_arg = (cl.GetArg("fail-on-warnings") == "1");
	if(cl.IsArg("min-bool-support"))
		min_bool_support = StrInt(cl.GetArg("min-bool-support"));
	if(cl.IsArg("fail-on-low-support"))
		fail_on_low_support_arg = (cl.GetArg("fail-on-low-support") == "1");
	if(cl.IsArg("allow-baseline-mismatch"))
		allow_baseline_mismatch = true;
		
	return true;
}

bool AppArguments::ParseRunPhase6ExitGate(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("out-dir", 0, "Output directory for report", true, "dir");
	cl.AddArg("split-file", 0, "Path to split JSON", true, "path");
	cl.AddArg("refresh-baselines", 0, "Refresh baselines (0|1)", true, "val");
	cl.AddArg("holdout-ratio", 0, "Holdout ratio for new split", true, "R");
	cl.AddArg("seed", 0, "Seed for new split", true, "N");
	cl.AddArg("bool-group", 0, "Bool group to baseline", true, "name");
	cl.AddArg("offset-subset", 0, "Offset subset (holdout|all)", true, "name");
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddArg("min-bool-support", 0, "Minimum positive support required per slot", true, "N");
	cl.AddArg("fail-on-low-support", 0, "Fail if any slot has low support (0|1)", true, "val");
	cl.AddArg("fail-on-warnings", 0, "Fail on audit warnings (0|1)", true, "val");
	cl.AddArg("min-holdout-f1", 0, "Min allowed holdout F1", true, "V");
	cl.AddArg("max-f1-drop", 0, "Max allowed F1 drop", true, "D");
	cl.AddArg("max-recall-drop", 0, "Max allowed recall drop", true, "D");
	cl.AddArg("allow-baseline-mismatch", 0, "Allow mismatch in baseline provenance", false);
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("out-dir")) {
		error = "Usage: --run-phase6-exit-gate <sln_path> --out-dir <dir> [--refresh-baselines 0|1] ...";
		mode = MODE_INVALID;
		return false;
	}
	
	mode = MODE_RUN_PHASE6_EXIT_GATE;
	annsln_path = cl.GetPositional(0);
	mine_out_dir = cl.GetArg("out-dir");
	
	exit_gate_refresh_baselines = true;
	if(cl.IsArg("refresh-baselines"))
		exit_gate_refresh_baselines = (cl.GetArg("refresh-baselines") == "1");
		
	if(cl.IsArg("split-file"))
		split_file = cl.GetArg("split-file");
	if(cl.IsArg("holdout-ratio"))
		make_split_ratio = ScanDouble(cl.GetArg("holdout-ratio"));
	if(cl.IsArg("seed"))
		make_split_seed = StrInt(cl.GetArg("seed"));
	if(cl.IsArg("bool-group"))
		tune_group = cl.GetArg("bool-group");
	if(cl.IsArg("offset-subset"))
		subset = cl.GetArg("offset-subset");
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	if(cl.IsArg("min-bool-support"))
		min_bool_support = StrInt(cl.GetArg("min-bool-support"));
	if(cl.IsArg("fail-on-low-support"))
		fail_on_low_support_arg = (cl.GetArg("fail-on-low-support") == "1");
	if(cl.IsArg("fail-on-warnings"))
		fail_on_warnings_arg = (cl.GetArg("fail-on-warnings") == "1");
	if(cl.IsArg("min-holdout-f1"))
		min_holdout_f1 = ScanDouble(cl.GetArg("min-holdout-f1"));
	if(cl.IsArg("max-f1-drop"))
		max_f1_drop = ScanDouble(cl.GetArg("max-f1-drop"));
	if(cl.IsArg("max-recall-drop"))
		max_recall_drop = ScanDouble(cl.GetArg("max-recall-drop"));
	if(cl.IsArg("allow-baseline-mismatch"))
		allow_baseline_mismatch = true;
		
	return true;
}

bool AppArguments::ParseTraceFrameLog(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("out", 0, "Output path for verbose log", true, "path");
	cl.AddArg("model-set", 0, "Model set name (pass1|pass2)", true, "name");
	cl.AddArg("offset-mode", 0, "Offset mode (auto|none|combined|split)", true, "mode");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	cl.AddPositional("image_path", "Path to image", STRING_V);
	
	if(!cl.Parse(args) || cl.GetPositionalCount() < 2 || !cl.IsArg("out")) {
		error = "Usage: --trace-frame-log <sln_path> <image_path> --out <log.txt> [...]";
		mode = MODE_INVALID;
		return false;
	}
	
	mode = MODE_TRACE_FRAME_LOG;
	annsln_path = cl.GetPositional(0);
	trace_image_path = cl.GetPositional(1);
	trace_out_path = cl.GetArg("out");
	
	if(cl.IsArg("model-set"))
		subset = cl.GetArg("model-set"); // reusing subset field for model-set name
	else
		subset = "pass1";
		
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
		
	return true;
}

bool AppArguments::ParseRefreshPhase6Baselines(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("out-dir", 0, "Output directory for baselines", true, "dir");
	cl.AddArg("split-file", 0, "Existing split JSON (optional)", true, "path");
	cl.AddArg("holdout-ratio", 0, "Holdout ratio for new split", true, "R");
	cl.AddArg("seed", 0, "Seed for new split", true, "N");
	cl.AddArg("bool-group", 0, "Bool group to baseline", true, "name");
	cl.AddArg("offset-subset", 0, "Offset subset (holdout|all)", true, "name");
	cl.AddArg("bool-gate-policy", 0, "Bool gate policy (strict|permissive)", true, "policy");
	cl.AddArg("offset-mode", 0, "Offset mode (auto, none, combined, split)", true, "mode");
	cl.AddPositional("sln_path", "Path to .annsln", STRING_V);
	
	if(!cl.Parse(args) || cl.GetPositionalCount() < 1 || !cl.IsArg("out-dir")) {
		error = "Usage: --refresh-phase6-baselines <sln_path> --out-dir <dir> [--split-file <path>] [--holdout-ratio R] [--seed N] ...";
		mode = MODE_INVALID;
		return false;
	}
	
	mode = MODE_REFRESH_PHASE6_BASELINES;
	annsln_path = cl.GetPositional(0);
	mine_out_dir = cl.GetArg("out-dir");
	
	if(cl.IsArg("split-file"))
		split_file = cl.GetArg("split-file");
	if(cl.IsArg("holdout-ratio"))
		make_split_ratio = ScanDouble(cl.GetArg("holdout-ratio"));
	if(cl.IsArg("seed"))
		make_split_seed = StrInt(cl.GetArg("seed"));
	if(cl.IsArg("bool-group"))
		tune_group = cl.GetArg("bool-group");
	if(cl.IsArg("offset-subset"))
		subset = cl.GetArg("offset-subset");
	if(cl.IsArg("bool-gate-policy"))
		bool_gate_policy_arg = ParseBoolGatePolicy(cl.GetArg("bool-gate-policy"));
	if(cl.IsArg("offset-mode"))
		offset_mode_arg = ParseOffsetMode(cl.GetArg("offset-mode"));
		
	return true;
}

bool AppArguments::ParseAnnotationEditorArgs(const Vector<String>& args) {
	CommandLineArguments cl;
	cl.AddArg("strict-baseline", 0, "Fail on baseline mismatch (instead of warning)", false);
	cl.AddPositional("project", "Optional project path", STRING_V, String());
	if(!cl.Parse(args)) {
		error = "Failed to parse command line.";
		mode = MODE_INVALID;
		return false;
	}

	mode = MODE_ANNOTATION_EDITOR_ARGS;
	allow_baseline_mismatch = !cl.IsArg("strict-baseline");
	if(cl.GetPositionalCount() > 0) {
		String p = cl.GetPositional(0);
		if(IsExistingFileWithExt(p, ".annprj"))
			open_project = p;
	}
	return true;
}

void ApplyAnnotationEditorArguments(AnnotationEditorWindow& win, const AppArguments& args) {
	if(!args.open_project.IsEmpty())
		win.LoadProject(args.open_project);
}


}
