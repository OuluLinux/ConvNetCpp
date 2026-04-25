#include "3DModelLearning.h"

GUI_APP_MAIN
{
	try {
		Model3DApp app;
		// app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
