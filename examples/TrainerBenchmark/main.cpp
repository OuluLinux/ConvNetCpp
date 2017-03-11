#include "TrainerBenchmark.h"

GUI_APP_MAIN {
	{
		TrainerBenchmark tb;
		
		{
			LoaderMNIST l(tb.GetSessionData());
			l.Run();
			if (l.IsFail()) return;
		}
		
		{
			tb.PostReload();
			tb.Run();
		}
	}
	
	Thread::ShutdownThreads();
}
