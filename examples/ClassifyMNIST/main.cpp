#include "ClassifyMNIST.h"

GUI_APP_MAIN
{
	{
		Loader l;
		l.Run();
		if (l.IsFail()) return;
	}
	
	{
		ClassifyMNIST().Run();
		Thread::ShutdownThreads();
	}
}
