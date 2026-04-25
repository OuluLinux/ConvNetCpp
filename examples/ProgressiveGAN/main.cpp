#include "ProgressiveGAN.h"

GUI_APP_MAIN
{
	try {
		ProgressiveGAN app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
