#ifndef _OCR_NetworkBuilder_h_
#define _OCR_NetworkBuilder_h_

#include <ConvNet/ConvNet.h>

namespace Upp {

namespace OCR {

class NetworkBuilder {
public:
	// Create text splitter network (regression for character boundaries)
	static bool CreateTextSplitter(ConvNet::Session& session);
	
	// Create sliding window text splitter (binary classification)
	static bool CreateTextSplitterWindow(ConvNet::Session& session);
	
	// Create character classifier network
	static bool CreateCharClassifier(ConvNet::Session& session, int class_count = 62);
	
	// Create ResNet-style character classifier
	static bool CreateResNetClassifier(ConvNet::Session& session, int class_count = 62);
	
	// Create custom network from JSON
	static bool CreateCustom(ConvNet::Session& session, const String& json);
};

} // namespace OCR

} // namespace Upp

#endif
