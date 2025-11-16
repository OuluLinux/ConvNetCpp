#include "DiffusionModel.h"

#define IMAGECLASS DiffusionModelImg
#define IMAGEFILE <DiffusionModel/DiffusionModel.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		DiffusionModel diffusion;

		// Initialize the Diffusion Model with appropriate parameters
		diffusion.l.SetTimesteps(1000);  // Standard DDPM timesteps

		diffusion.Init();

		diffusion.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}