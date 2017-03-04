#include "ClassifyImages.h"

GUI_APP_MAIN
{
	{
		#ifdef flagMNIST
		LoaderMNIST l;
		#elif defined flagCIFAR10
		LoaderCIFAR10 l;
		#endif
		l.Run();
		if (l.IsFail()) return;
	}
	
	{
		ClassifyImages().Run();
		Thread::ShutdownThreads();
	}
}
