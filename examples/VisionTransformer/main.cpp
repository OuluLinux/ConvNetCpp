#include "VisionTransformer.h"

GUI_APP_MAIN
{
	try {
		VisionTransformer vit;

		// Load CIFAR-10 data
		LoaderCIFAR10 l(vit.GetSession());
		l.Run();
		if (l.IsFail()) return;

		vit.PostReload();
		vit.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}