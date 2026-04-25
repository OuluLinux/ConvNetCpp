#include "BigGAN.h"

GUI_APP_MAIN
{
	try {
		BigGAN app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
