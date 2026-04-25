#include "AppMainArgs.h"
#include "AppMainCommands.h"
#include "AppMainGuiFlow.h"
#include "AnnotationEditorWindow.h"

using namespace Upp;

GUI_APP_MAIN {
	RegisterMluiRuntimeStarter();
	AppArguments args;
	if(!args.Parse(CommandLine())) {
		if(!args.error.IsEmpty())
			Cout() << args.error << "\n";
		SetExitCode(1);
		return;
	}

	SetFrameRecognizerMemoryDumpEnabled(args.frame_recognizer_dump_memory_alloc);
	if(args.frame_recognizer_dump_memory_alloc)
		DumpFrameRecognizerMemoryEvent("AnnotationEditor.Start", "memory dump enabled from CLI");

	int exit_code = 0;
	bool handled_cli_mode = false;
	if(RunCliMode(args, exit_code, handled_cli_mode)) {
		SetExitCode(exit_code);
		return;
	}

	RunGuiFlow(args);
}
