#include "ConditionalGAN.h"

GUI_APP_MAIN
{
	try {
		ConditionalGAN app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
