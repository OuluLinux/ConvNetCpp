#include "CLIPTester.h"

GUI_APP_MAIN
{
	try {
		CLIPApp app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
