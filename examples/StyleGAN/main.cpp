#include "StyleGAN.h"

GUI_APP_MAIN
{
	try {
		StyleGAN app;
		app.Init();
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
