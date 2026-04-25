#include "SwinTransformer.h"

GUI_APP_MAIN
{
	try {
		SwinTransformer app;
		app.Init(); 
		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
