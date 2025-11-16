#include "BigGAN.h"

#define IMAGECLASS BigGANImg
#define IMAGEFILE <BigGAN/BigGAN.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		BigGAN biggan;

		// Initialize the BigGAN with appropriate parameters
		// This would typically involve loading a dataset with labels
		biggan.l.SetNumClasses(1000); // For ImageNet classes
		biggan.l.SetNoiseSize(128);   // Standard BigGAN noise size

		biggan.Init();

		biggan.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}