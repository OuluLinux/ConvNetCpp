#include "AnnotationDomain.h"

using namespace Upp;

GUI_APP_MAIN {
	CommandLineArguments cl;
	cl.SetStopOnPositional(true);
	cl.AddHelpText("Positional: [open_path] Optional path to .anngrf_sln/.annprj_graph/.anngrf");
	if(!cl.Parse(CommandLine())) {
		cl.PrintHelp();
		return;
	}

	String open_path;
	Vector<String> rest = cl.GetRest();
	if(rest.GetCount() > 0)
		open_path = rest[0];

	AnnotationDomainWindow win;
	if(!open_path.IsEmpty()) {
		if(!win.OpenPath(open_path))
			PromptOK("Failed to open: " + open_path);
	}
	win.Run();
}
