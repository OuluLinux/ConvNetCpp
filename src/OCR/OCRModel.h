#ifndef _OCR_OCRModel_h_
#define _OCR_OCRModel_h_

#include <ConvNet/ConvNet.h>
#include "OCRTypes.h"
#include "ImageUtils.h"
#include "Normalization.h"

namespace Upp {

namespace OCR {

enum OCRModelType {
	OCR_MODEL_SPLITTER = 0,
	OCR_MODEL_CLASSIFIER = 1
};

class OCRModel {
public:
	OCRModel();
	
	// Serialization
	bool Save(const String& path);
	bool Load(const String& path);
	
	void Serialize(Stream& s);

	// Inference
	Vector<double> Predict(const Image& img);
	int PredictClass(const Image& img);

	// Configuration
	void SetType(OCRModelType type) { model_type = type; }
	OCRModelType GetType() const { return (OCRModelType)model_type; }
	
	void SetArchitecture(const String& json) { architecture_json = json; }
	const String& GetArchitecture() const { return architecture_json; }
	
	ConvNet::Session& GetSession() { return session; }
	void SetSession(const ConvNet::Session& s) { session.CopyFrom(const_cast<ConvNet::Session&>(s)); }
	
	// Metadata
	void SetMetadata(const String& key, const Value& val);
	Value GetMetadata(const String& key) const;
	String GetClassLabel(int cls) const;
	String GetMetadataJson() const;
	bool SetMetadataJson(const String& json);

private:
	ConvNet::Session session;
	int model_type;
	uint32 version;
	String architecture_json;
	ValueMap metadata;
	
	static const uint32 CURRENT_VERSION = 1;
};

} // namespace OCR

} // namespace Upp

#endif
