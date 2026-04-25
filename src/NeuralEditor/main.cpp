#include "NeuralEditor.h"

using namespace Upp;

namespace {

String FindRepoRoot() {
	String from_exe = NormalizePath(AppendFileName(GetFileDirectory(GetExeFilePath()), ".."));
	if(FileExists(AppendFileName(from_exe, "script/convert_examples_to_nn.py")))
		return from_exe;

	String cwd = NormalizePath(GetCurrentDirectory());
	if(FileExists(AppendFileName(cwd, "script/convert_examples_to_nn.py")))
		return cwd;

	return from_exe;
}

bool GenerateTemplateProject(const String& repo_root, const String& templ_name, String& sln_path, String& error_out) {
	String script = AppendFileName(repo_root, "script/convert_examples_to_nn.py");
	if(!FileExists(script)) {
		error_out = "Template generator script not found: " + script;
		return false;
	}

	Vector<String> args;
	args.Add(script);
	args.Add("--repo");
	args.Add(repo_root);
	args.Add("--new-template");
	args.Add(templ_name);

	LocalProcess proc;
	if(!proc.Start("python3", args)) {
		error_out = "Failed to start python3 template generator.";
		return false;
	}

	String out;
	while(proc.IsRunning()) {
		out << proc.Get();
		Sleep(10);
	}
	out << proc.Get();
	int exit_code = proc.GetExitCode();
	if(exit_code != 0) {
		error_out = Format("Template generation failed (exit %d):\n%s", exit_code, out);
		return false;
	}

	String project_name = templ_name;
	project_name = TrimBoth(project_name);
	int slash1 = project_name.ReverseFind('/');
	int slash2 = project_name.ReverseFind('\\');
	int slash = slash1 > slash2 ? slash1 : slash2;
	if(slash >= 0)
		project_name = project_name.Mid(slash + 1);
	if(project_name.IsEmpty()) {
		error_out = "Template name is empty.";
		return false;
	}

	sln_path = NormalizePath(AppendFileName(repo_root, "nn/" + project_name + "/" + project_name + ".nnsln"));
	if(!FileExists(sln_path)) {
		error_out = "Template generation did not produce .nnsln at: " + sln_path;
		return false;
	}

	LOG("NeuralEditor template generated: " + sln_path);
	return true;
}

}

GUI_APP_MAIN {
	CommandLineArguments cl;
	cl.AddArg("self-test", 0, "Run NeuralCompiler self-tests and exit", false);
	cl.AddArg("new-from-template", 'n', "Generate template project by name and open its .nnsln", true, "name");
	cl.SetStopOnPositional(true);
	cl.AddHelpText("Positional: [open_path] Optional path to .sln/.slnx/.grfproj/.grf (legacy: .nnsln/.nnprj/.nngrf)");
	if(!cl.Parse(CommandLine())) {
		cl.PrintHelp();
		return;
	}

	if(cl.IsArg("self-test")) {
		String report;
		bool ok = NeuralCompiler::RunSelfTests(report);
		Cout() << report;
		LOG(String("NeuralCompiler self-tests: ") + (ok ? "PASS" : "FAIL"));
		return;
	}

	String open_path;
	if(cl.IsArg("new-from-template")) {
		String repo_root = FindRepoRoot();
		String err;
		if(!GenerateTemplateProject(repo_root, cl.GetArg("new-from-template"), open_path, err)) {
			PromptOK(err);
			return;
		}
	}
	else {
		Vector<String> rest = cl.GetRest();
		if(rest.GetCount() > 0)
			open_path = rest[0];
	}

	NeuralEditorWindow win;
	if(!open_path.IsEmpty()) {
		if(!win.OpenPath(open_path))
			PromptOK("Failed to open: " + open_path);
	}
	win.Run();
}
