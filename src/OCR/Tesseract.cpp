#include "Tesseract.h"
#include <plugin/png/png.h>

namespace Upp {
namespace OCR {

#ifdef PLATFORM_WIN32
static String RunCommandCapture(const char *cmd, int& exit_code)
{
	String out;
	char buf[4096];
	FILE *pipe = _popen(cmd, "r");
	if(!pipe) {
		exit_code = -1;
		return "";
	}
	while(fgets(buf, sizeof(buf), pipe))
		out.Cat(buf);
	exit_code = _pclose(pipe);
	return out;
}
#else
static String RunCommandCapture(const char *cmd, int& exit_code)
{
	String out;
	char buf[4096];
	FILE *pipe = popen(cmd, "r");
	if(!pipe) {
		exit_code = -1;
		return "";
	}
	while(fgets(buf, sizeof(buf), pipe))
		out.Cat(buf);
	exit_code = pclose(pipe);
	return out;
}
#endif

bool HasTesseract()
{
	int rc = -1;
#ifdef PLATFORM_WIN32
	RunCommandCapture("tesseract --version > NUL 2>&1", rc);
#else
	RunCommandCapture("tesseract --version >/dev/null 2>&1", rc);
#endif
	return rc == 0;
}

bool RecognizeWithTesseract(const Image& patch, int psm, String& text, double& confidence, const String& whitelist, const String& blacklist)
{
	text.Clear();
	confidence = 0.0;
	if(patch.IsEmpty())
		return false;

	Image prep = patch;
	// Tesseract likes larger images
	if(prep.GetWidth() < 400 || prep.GetHeight() < 120) {
		double factor = max(400.0 / prep.GetWidth(), 120.0 / prep.GetHeight());
		prep = Rescale(prep, (int)(prep.GetWidth() * factor), (int)(prep.GetHeight() * factor));
	}

	String tmp = AppendFileName(GetTempDirectory(), "ocr_tess_" + AsString(GetTickCount()) + ".png");
	if(!PNGEncoder().SaveFile(tmp, prep)) {
		Cout() << "Failed to save temp PNG: " << tmp << "\n";
		return false;
	}

	String cmd;
	String psm_arg = AsString(max(0, psm));
	String extra_args;
	if(!whitelist.IsEmpty())
		extra_args << " -c tessedit_char_whitelist=\"" << whitelist << "\"";
	if(!blacklist.IsEmpty())
		extra_args << " -c tessedit_char_blacklist=\"" << blacklist << "\"";

#ifdef PLATFORM_WIN32
	cmd = "tesseract \"" + tmp + "\" stdout --psm " + psm_arg + extra_args + " -l eng 2> NUL";
#else
	cmd = "tesseract \"" + tmp + "\" stdout --psm " + psm_arg + extra_args + " -l eng 2>/dev/null";
#endif
	int rc = -1;
	String out = RunCommandCapture(cmd, rc);
	if(FileExists(tmp)) DeleteFile(tmp);
	if(rc != 0)
		return false;

	out.Replace("\f", " ");
	text = TrimBoth(out);
	if(!text.IsEmpty())
		confidence = 1.0; // Tesseract doesn't easily give confidence per line in stdout mode without more parsing
	return true;
}

} // namespace OCR
} // namespace Upp
