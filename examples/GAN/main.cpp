#include "GAN.h"
#include "LoaderMNIST.h"

#define IMAGECLASS GANImg
#define IMAGEFILE <GAN/GAN.iml>
#include <Draw/iml_source.h>


GUI_APP_MAIN
{
	try {
		GAN gan;
		
		LoaderMNIST l(gan.l.GetDiscriminator());
		l.Run();
		if (l.IsFail()) return;
		
		gan.Init();
		
		gan.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}
