#include "T5Tester.h"

GUI_APP_MAIN
{
	try {
		T5App app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
