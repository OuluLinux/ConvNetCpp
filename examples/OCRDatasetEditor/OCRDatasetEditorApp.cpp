#include "OCRDatasetEditor.h"

#ifdef flagMAIN

using namespace Upp;

GUI_APP_MAIN {
	StdLogSetup(LOG_FILE|LOG_COUT);

	OCRDatasetEditor editor;

	const Vector<String>& args = CommandLine();
	if (args.GetCount() > 0) {
		editor.LoadProject(args[0]);
	}

	editor.Run();
}

#endif
