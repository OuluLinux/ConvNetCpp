#include "WGAN.h"

GUI_APP_MAIN
{
	try {
		WGAN app;
		// Some apps need Init() before Run()
		// app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
