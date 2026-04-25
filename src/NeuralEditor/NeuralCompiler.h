#ifndef _NeuralEditor_NeuralCompiler_h_
#define _NeuralEditor_NeuralCompiler_h_

#include <Node/Core/Core.h>
#include <ConvNet/ConvNet.h>

NAMESPACE_UPP

class NeuralCompiler {
public:
	struct Result {
		bool ok = true;
		Vector<String> errors;
		Vector<String> warnings;

		String ToText() const;
	};

	static Result Validate(const Node::Graph& graph);
	static bool BuildSession(const Node::Graph& graph, ::ConvNet::Session& session, String& error_out);
	static bool RunSelfTests(String& report_out);
};

END_UPP_NAMESPACE

#endif
