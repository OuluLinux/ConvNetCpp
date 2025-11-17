#include "AnimationLearning.h"

#define IMAGECLASS AnimImg
#define IMAGEFILE <AnimationLearning/AnimationLearning.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		AnimationApp app;

		// The animation learning app demonstrates various approaches for 
		// learning and generating skeleton-based animations using neural networks

		app.Init();

		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}