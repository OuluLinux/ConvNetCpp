#include "CycleGAN.h"

GUI_APP_MAIN
{
	try {
		CycleGAN app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
