#include "SwinTransformer.h"

#define IMAGECLASS SwinTransformerImg
#define IMAGEFILE <SwinTransformer/SwinTransformer.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		SwinTransformer swin;

		// Here you would typically load data for training
		// Swin Transformer works well with image classification tasks

		swin.Init();

		swin.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}