#include "3DModelLearning.h"

#define IMAGECLASS Model3DImg
#define IMAGEFILE <3DModelLearning/3DModelLearning.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		Model3DApp app;

		// The 3D model learning app demonstrates various 3D representation techniques
		// including voxel-based, point cloud, occupancy networks, and DeepSDF

		app.Init();

		app.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}