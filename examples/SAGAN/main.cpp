#include "SAGAN.h"

GUI_APP_MAIN
{
	try {
		SAGAN app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
