#include "EfficientNet.h"

#define IMAGECLASS EfficientNetImg
#define IMAGEFILE <EfficientNet/EfficientNet.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		EfficientNetApp app;

		// The EfficientNet app will be initialized with compound scaling parameters
		// This example uses the B0 configuration as the baseline

		app.Init();

		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}