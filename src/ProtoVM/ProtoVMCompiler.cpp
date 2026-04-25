#include "ProtoVMCompiler.h"

NAMESPACE_UPP

String ProtoVMCompiler::Result::ToText() const {
	String out;
	for(const String& e : errors)   out << "ERROR: "   << e << "\n";
	for(const String& w : warnings) out << "WARNING: " << w << "\n";
	if(out.IsEmpty()) out = "OK\n";
	return out;
}

ProtoVMCompiler::Result ProtoVMCompiler::Validate(const Node::Graph& graph) {
	Result r;

	const Node::GraphDoc& doc = graph.GetDoc();

	// Count notable node types
	bool has_source = false;
	bool has_ground = false;

	for(const Node::NodeDoc& nd : doc.nodes) {
		const String& tid = nd.node_type_id;
		if(tid == "electric.source.voltage" || tid == "electric.source.current")
			has_source = true;
		if(tid == "electric.ground")
			has_ground = true;
	}

	// A non-trivial circuit should have a source and a ground
	if(doc.nodes.GetCount() > 1) {
		if(!has_source)
			r.warnings.Add("No voltage or current source in circuit.");
		if(!has_ground)
			r.warnings.Add("No ground reference node in circuit.");
	}

	// Find isolated nodes (no edge references them)
	Index<String> connected;
	for(const Node::EdgeDoc& ed : doc.edges) {
		connected.FindAdd(ed.source_node);
		connected.FindAdd(ed.target_node);
	}
	for(const Node::NodeDoc& nd : doc.nodes) {
		if(nd.pins.GetCount() == 0) continue; // annotation/label node
		if(connected.Find(nd.id) < 0)
			r.warnings.Add("Node '" + nd.id + "' (" + nd.label + ") is isolated — no connections.");
	}

	r.ok = r.errors.IsEmpty();
	return r;
}

END_UPP_NAMESPACE
