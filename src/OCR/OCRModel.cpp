#include "OCRModel.h"

namespace Upp {

namespace OCR {

static double GetCardGrayValue(const RGBA& pixel) {
	return (double)((pixel.r * 77 + pixel.g * 150 + pixel.b * 29) >> 8);
}


OCRModel::OCRModel() {
	model_type = OCR_MODEL_CLASSIFIER;
	version = CURRENT_VERSION;
}

bool OCRModel::Save(const String& path) {
	FileOut out(path);
	if (!out.IsOpen()) return false;
	
	Serialize(out);
	return !out.IsError();
}

bool OCRModel::Load(const String& path) {
	FileIn in(path);
	if (!in.IsOpen()) return false;

	Serialize(in);
	if (in.IsError()) {
		in.ClearError();
		in.Seek(0);
		session.Serialize(in);
		if (!in.IsError() && session.GetLayerCount() > 0) {
			version = CURRENT_VERSION;
			model_type = OCR_MODEL_CLASSIFIER;
			
			// Explicitly load metadata if it exists
			String base = path;
			if (path.EndsWith(".model.bin")) base = path.Left(path.GetCount() - 10);
			else if (path.EndsWith(".bin")) base = path.Left(path.GetCount() - 4);
			String meta_path = base + ".metadata.json";
			if (FileExists(meta_path)) {
				Value m = ParseJSON(LoadFile(meta_path));
				if (m.Is<ValueMap>()) {
					Value classes = m["classes"];
					if (classes.Is<ValueArray>()) {
						metadata.Add("classes", classes);
					}
				}
			}
			return true;
		}
	}
	return !in.IsError();
}

void OCRModel::Serialize(Stream& s) {
	// Magic header
	String magic = "OCRMODEL";
	s % magic;
	if (s.IsLoading() && magic != "OCRMODEL") {
		s.SetError();
		return;
	}
	
	// Version
	s % version;
	
	// Model Type
	s % model_type;
	
	// Architecture JSON
	s % architecture_json;
	
	// If loading, we need to rebuild the layers before loading weights
	if (s.IsLoading()) {
		if (!architecture_json.IsEmpty()) {
			session.MakeLayers(architecture_json);
		}
	}
	
	// Weights (via Session/Net serialization)
	session.Serialize(s);
	
	// Metadata
	String metadata_json;
	if (s.IsStoring()) {
		metadata_json = StoreAsJson(metadata);
	}
	s % metadata_json;
	if (s.IsLoading()) {
		LoadFromJson(metadata, metadata_json);
	}
}

Vector<double> OCRModel::Predict(const Image& img) {
	if (img.IsEmpty() || session.GetLayerCount() == 0) return Vector<double>();

	Size target(20, 32);
	ConvNet::Net& net = session.GetNetwork();
	if (net.GetLayers().GetCount() > 0) {
		target.cx = net.GetLayers()[0].output_width;
		target.cy = net.GetLayers()[0].output_height;
	}

	// Default normalization (Task 03)
	// Bypass NormalizeChar for cards as it crops tightly and loses context
	Image normalized;
	if (model_type == OCR_MODEL_CLASSIFIER) {
		normalized = ScaleImage(img, target);
	} else {
		normalized = NormalizeChar(img, target);
	}

	// Convert Image to Vector<double>
	Vector<double> input;
	Size sz = normalized.GetSize();
	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			input.Add(GetCardGrayValue(normalized[y][x]) / 255.0 - 0.5);
		}
	}

	return session.Predict(input);
}

int OCRModel::PredictClass(const Image& img) {
	Vector<double> probs = Predict(img);
	if (probs.IsEmpty()) return -1;
	
	int best = 0;
	double max_p = probs[0];
	for (int i = 1; i < probs.GetCount(); i++) {
		if (probs[i] > max_p) {
			max_p = probs[i];
			best = i;
		}
	}
	
	return best;
}

void OCRModel::SetMetadata(const String& key, const Value& val) {
	metadata.Add(key, val);
}

Value OCRModel::GetMetadata(const String& key) const {
	int q = metadata.Find(key);
	if (q >= 0) return metadata.GetValue(q);
	return Value();
}

String OCRModel::GetMetadataJson() const {
	return StoreAsJson(metadata);
}

bool OCRModel::SetMetadataJson(const String& json) {
	return LoadFromJson(metadata, json);
}


String OCRModel::GetClassLabel(int cls) const {
	Value classes = GetMetadata("classes");
	if (classes.Is<ValueArray>()) {
		ValueArray arr = classes;
		if (cls >= 0 && cls < arr.GetCount()) return arr[cls].ToString();
	}
	return "";
}

} // namespace OCR

} // namespace Upp
