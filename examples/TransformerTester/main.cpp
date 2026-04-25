#include "TransformerTester.h"

GUI_APP_MAIN
{
	try {
		TransformerApp app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
