#ifndef _ProtoVM_ProtoVMCompiler_h_
#define _ProtoVM_ProtoVMCompiler_h_

#include <Node/Core/Core.h>

NAMESPACE_UPP

// ---------------------------------------------------------------------------
// ProtoVMCompiler — static graph validator for electric circuit topology
// ---------------------------------------------------------------------------
// Rules:
//   - Every POWER pin must connect to exactly one SOURCE or WIRE
//   - Every GROUND node must have at least one POWER or WIRE input
//   - SIGNAL / CONTROL lines are directional: one driver, many receivers
//   - Floating pins (unconnected non-optional) are warnings
//   - Short-circuits (POWER directly to GROUND without impedance) are errors

class ProtoVMCompiler {
public:
	struct Result {
		bool ok = true;
		Vector<String> errors;
		Vector<String> warnings;

		String ToText() const;
	};

	static Result Validate(const Node::Graph& graph);
};

END_UPP_NAMESPACE

#endif
