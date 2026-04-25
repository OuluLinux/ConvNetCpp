#include "BERTTester.h"

GUI_APP_MAIN
{
	try {
		BERTTester app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
