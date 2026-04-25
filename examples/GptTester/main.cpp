#include "GptTester.h"

GUI_APP_MAIN
{
	try {
		GptApp app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
