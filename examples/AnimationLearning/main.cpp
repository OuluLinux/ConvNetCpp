#include "AnimationLearning.h"

GUI_APP_MAIN
{
	try {
		AnimationApp app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
