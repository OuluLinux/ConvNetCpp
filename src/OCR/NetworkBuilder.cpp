#include "NetworkBuilder.h"

namespace Upp {

namespace OCR {

bool NetworkBuilder::CreateTextSplitter(ConvNet::Session& session) {
	String json = R"([
    {"type": "input", "input_width": 28, "input_height": 28, "input_depth": 1},
    {"type": "conv", "filter_count": 16, "width": 5, "height": 5, "stride": 1, "pad": 2, "activation": "relu"},
    {"type": "pool", "width": 2, "height": 2, "stride": 2},
    {"type": "conv", "filter_count": 32, "width": 5, "height": 5, "stride": 1, "pad": 2, "activation": "relu"},
    {"type": "pool", "width": 2, "height": 2, "stride": 2},
    {"type": "fc", "neuron_count": 128, "activation": "relu"},
    {"type": "regression", "neuron_count": 1},
    {"type": "sgd", "learning_rate": 0.01, "momentum": 0.9, "batch_size": 32, "l2_decay": 0.0001}
])";
	
	return session.MakeLayers(json);
}

bool NetworkBuilder::CreateTextSplitterWindow(ConvNet::Session& session) {
	String json = R"([
    {"type": "input", "input_width": 3, "input_height": 28, "input_depth": 1},
    {"type": "conv", "filter_count": 16, "width": 3, "height": 3, "pad": 1, "activation": "relu"},
    {"type": "pool", "width": 2, "height": 2, "stride": 2},
    {"type": "conv", "filter_count": 32, "width": 3, "height": 3, "pad": 1, "activation": "relu"},
    {"type": "pool", "width": 2, "height": 2, "stride": 2},
    {"type": "fc", "neuron_count": 64, "activation": "relu"},
    {"type": "softmax", "class_count": 2},
    {"type": "sgd", "learning_rate": 0.01, "momentum": 0.9, "batch_size": 64}
])";
	
	return session.MakeLayers(json);
}

bool NetworkBuilder::CreateCharClassifier(ConvNet::Session& session, int class_count) {
	String json = Format(R"([
    {"type": "input", "input_width": 28, "input_height": 28, "input_depth": 1},
    {"type": "conv", "filter_count": 20, "width": 5, "height": 5, "stride": 1, "pad": 0, "activation": "relu"},
    {"type": "pool", "width": 2, "height": 2, "stride": 2},
    {"type": "conv", "filter_count": 50, "width": 5, "height": 5, "stride": 1, "pad": 0, "activation": "relu"},
    {"type": "pool", "width": 2, "height": 2, "stride": 2},
    {"type": "fc", "neuron_count": 500, "activation": "relu"},
    {"type": "softmax", "class_count": %d},
    {"type": "sgd", "learning_rate": 0.01, "momentum": 0.9, "batch_size": 32, "l2_decay": 0.0001}
])", class_count);
	
	return session.MakeLayers(json);
}

bool NetworkBuilder::CreateResNetClassifier(ConvNet::Session& session, int class_count) {
	String json = Format(R"([
    {"type": "input", "input_width": 32, "input_height": 32, "input_depth": 1},
    {"type": "conv", "filter_count": 32, "width": 3, "height": 3, "pad": 1, "activation": "relu"},
    {"type": "conv", "filter_count": 32, "width": 3, "height": 3, "pad": 1, "activation": "relu"},
    {"type": "pool", "width": 2, "height": 2, "stride": 2},
    {"type": "conv", "filter_count": 64, "width": 3, "height": 3, "pad": 1, "activation": "relu"},
    {"type": "conv", "filter_count": 64, "width": 3, "height": 3, "pad": 1, "activation": "relu"},
    {"type": "pool", "width": 2, "height": 2, "stride": 2},
    {"type": "fc", "neuron_count": 128, "activation": "relu"},
    {"type": "softmax", "class_count": %d},
    {"type": "adam", "learning_rate": 0.001, "batch_size": 32}
])", class_count);
	
	return session.MakeLayers(json);
}

bool NetworkBuilder::CreateCustom(ConvNet::Session& session, const String& json) {
	return session.MakeLayers(json);
}

} // namespace OCR

} // namespace Upp
