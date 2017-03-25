#include "NetworkOptimization.h"

GUI_APP_MAIN {
	NetworkOptimization ap;
	ap.Run();
	Thread::ShutdownThreads();
}
