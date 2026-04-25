#include <Core/Core.h>
#include <OCR/OCR.h>

using namespace Upp;
using namespace OCR;

void ProcessDirectory(const String& dir_path, OCREngine& engine) {
	FindFile ff(AppendFileName(dir_path, "*.*"));

	int processed = 0;
	int total = 0;

	while (ff) {
		String path = AppendFileName(dir_path, ff.GetName());

		if (ff.IsFile()) {
			String ext = ToLower(GetFileExt(path));
			if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
				total++;
				Cout() << "Processing: " << ff.GetName() << "... ";

				FileIn in(path);
				Image img = StreamRaster::LoadAny(in);
				
				if (!img.IsEmpty()) {
					OCRResult result = engine.RecognizePage(img);

					// Save result to text file
					String out_path = ForceExt(path, ".txt");
					SaveFile(out_path, result.full_text);

					Cout() << "OK (" << Format("%.1f%%", result.avg_confidence * 100.0) << ")\n";
					processed++;
				}
				else {
					Cout() << "FAILED (Image load error)\n";
				}
			}
		}

		ff.Next();
	}

	Cout() << "\nFinished. Processed " << processed << "/" << total << " images\n";
}

CONSOLE_APP_MAIN {
	const Vector<String>& args = CommandLine();
	if (args.GetCount() < 1) {
		Cout() << "Usage: BatchOCR <directory_path> [classifier_model_path]\n";
		Exit(1);
	}

	String dir_path = args[0];
	String model_path = (args.GetCount() > 1) ? args[1] : "classifier.ocrmodel";

	// Setup engine
	OCRConfig config;
	config.classifier_model_path = model_path;

	OCREngine engine;
	if (!engine.Initialize(config)) {
		Cerr() << "Error: " << engine.GetLastError() << "\n";
		Exit(1);
	}

	ProcessDirectory(dir_path, engine);
}
