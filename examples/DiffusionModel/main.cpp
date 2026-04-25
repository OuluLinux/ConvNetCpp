#include "DiffusionModel.h"

GUI_APP_MAIN
{
	try {
		DiffusionModel app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
