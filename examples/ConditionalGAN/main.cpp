#include "ConditionalGAN.h"

#define IMAGECLASS ConditionalGANImg
#define IMAGEFILE <ConditionalGAN/ConditionalGAN.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		ConditionalGAN condgan;

		// Initialize the conditional GAN with appropriate parameters
		// This would typically involve loading a dataset with labels
		condgan.l.SetNumClasses(10); // For example, 10 classes like in MNIST

		condgan.Init();

		condgan.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}