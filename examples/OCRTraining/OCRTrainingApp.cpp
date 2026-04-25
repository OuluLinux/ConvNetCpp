#include "OCRTraining.h"

#ifdef flagMAIN

// #warning flagMAIN is defined!

using namespace Upp;

GUI_APP_MAIN {
	StdLogSetup(LOG_FILE|LOG_COUT);

	OCRTraining trainer;

	const Vector<String>& args = CommandLine();
	for (int i = 0; i < args.GetCount(); i++) {
		if (args[i] == "--dataset" && i + 1 < args.GetCount()) {
			// trainer.LoadDataset(args[i + 1]);
			i++;
		}
	}

	trainer.Run();
}

#endif
