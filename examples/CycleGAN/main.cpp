#include "CycleGAN.h"

#define IMAGECLASS CycleGANImg
#define IMAGEFILE <CycleGAN/CycleGAN.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		CycleGAN cyclegan;

		// Here you would typically load data for both domains X and Y
		// For now, we'll initialize with basic settings
		cyclegan.l.SetLambdaCycle(10.0); // Standard CycleGAN lambda for cycle consistency

		cyclegan.Init();

		cyclegan.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}