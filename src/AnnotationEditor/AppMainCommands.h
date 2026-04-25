#ifndef _AnnotationEditor_AppMainCommands_h_
#define _AnnotationEditor_AppMainCommands_h_

#include "AppMainArgs.h"

namespace Upp {

bool RunCliMode(const AppArguments& args, int& exit_code, bool& handled_cli_mode);

bool RunTrainSlots(const AppArguments& args);
bool RunTrainSlotsFromCrops(const AppArguments& args);
bool RunTrainAll(const AppArguments& args);
bool RunDebugGuiTrain(const AppArguments& args);
bool RunRecognizeSlots(const AppArguments& args);
bool RunTrainHeadless(const AppArguments& args);
bool RunExportCrops(const AppArguments& args);
bool RunMineBoolHardNegatives(const AppArguments& args);
bool RunMergeHardneg(const AppArguments& args);
bool RunSaveOffsetBaseline(const AppArguments& args);
bool RunCheckOffsetBaseline(const AppArguments& args);
bool RunEvalOffsetHeads(const AppArguments& args);
bool RunTuneBoolGates(const AppArguments& args);
bool RunSaveBoolBaseline(const AppArguments& args);
bool RunCheckBoolBaseline(const AppArguments& args);
bool RunReportBoolGates(const AppArguments& args);
bool RunEvalBoolGates(const AppArguments& args);
bool RunAuditOffsetHeads(const AppArguments& args);
bool RunPruneOffsetHeads(const AppArguments& args);
bool RunFixBoolGates(const AppArguments& args);
bool RunMakeEvalSplit(const AppArguments& args);
bool RunRecomputeAnchors(const AppArguments& args);
bool RunAuditBoolGates(const AppArguments& args);
bool RunSmokeTest(const AppArguments& args);
bool RunRunPhase6Preflight(const AppArguments& args);
bool RunRefreshPhase6Baselines(const AppArguments& args);
bool RunRunPhase6ExitGate(const AppArguments& args);
bool RunTraceFrameLog(const AppArguments& args);
bool RunReportCardVisibilityConsistency(const AppArguments& args);
bool RunTuneCardVisibilityByConsistency(const AppArguments& args);
bool RunTuneCardJointPipeline(const AppArguments& args);
bool RunSentinelTestRB(const AppArguments& args);
bool RunPipelineCompletenessTest(const AppArguments& args);
bool RunTestFRInvariants(const AppArguments& args);
bool RunTestXOffsetConvLayers(const AppArguments& args);
bool RunEvalOcrModes(const AppArguments& args);
bool RunFrameRecognizerDumpSteps(const AppArguments& args);
bool RunFrameRecognizerDumpMemoryAlloc(const AppArguments& args);
bool RunDumpNNSteps(const AppArguments& args);
bool RunMigrateAnnmdlSessions(const AppArguments& args);
bool RunMigrateAnnmdlV3(const AppArguments& args);
bool RunMergeModes(const AppArguments& args);
bool RunMergeAnnMdl(const String& src_path, const String& dst_path);
bool RunNormalizeCompositeHeads(const AppArguments& args);

}

#endif
