#include <Core/Core.h>
#include <OCR/OCR.h>

using namespace Upp;
using namespace OCR;

CONSOLE_APP_MAIN {
	const Vector<String>& args = CommandLine();
	if (args.GetCount() < 1) {
		Cout() << "Usage: SimpleOCR <image_path> [classifier_model_path]\n";
		Exit(1);
	}

	String image_path = args[0];
	String model_path = (args.GetCount() > 1) ? args[1] : "classifier.ocrmodel";

	// 1. Configure engine
	OCRConfig config;
	config.classifier_model_path = model_path;

	// 2. Initialize engine
	OCREngine engine;
	if (!engine.Initialize(config)) {
		Cerr() << "Error: " << engine.GetLastError() << "\n";
		Exit(1);
	}

	// 3. Load image
	FileIn in(image_path);
	if (!in.IsOpen()) {
		Cerr() << "Error: Could not open image file: " << image_path << "\n";
		Exit(1);
	}
	Image img = StreamRaster::LoadAny(in);
	if (img.IsEmpty()) {
		Cerr() << "Error: Failed to load image from: " << image_path << "\n";
		Exit(1);
	}

	// 4. Recognize
	OCRResult result = engine.RecognizePage(img);

	// 5. Output results
	Cout() << "Recognized Text:\n";
	Cout() << "----------------\n";
	Cout() << result.full_text << "\n";
	Cout() << "----------------\n";
	Cout() << "Confidence: " << Format("%.1f%%", result.avg_confidence * 100.0) << "\n";
	Cout() << "Characters: " << result.total_characters << "\n";
	Cout() << "Processing time: " << result.processing_time_ms << " ms\n";
}
