#include "NeuralCompiler.h"
#include <cstring>

NAMESPACE_UPP

namespace {

using namespace Node;

const PinDoc* FindPin(const NodeDoc& node, const String& pin_id, PinKind kind) {
	for(const PinDoc& p : node.pins)
		if(p.id == pin_id && p.kind == kind)
			return &p;
	return nullptr;
}

const WidgetSlotDoc* FindSlot(const NodeDoc& node, const String& slot_id) {
	for(const WidgetSlotDoc& s : node.slots)
		if(s.id == slot_id)
			return &s;
	return nullptr;
}

const char* kLayerModulePrefix = "convnet.module.layer.";
const char* kTrainerModulePrefix = "convnet.module.trainer.";

bool ParseBoolText(const String& s, bool& out) {
	String t = ToLower(TrimBoth(s));
	if(t == "true" || t == "1" || t == "yes" || t == "on") {
		out = true;
		return true;
	}
	if(t == "false" || t == "0" || t == "no" || t == "off") {
		out = false;
		return true;
	}
	return false;
}

Value GetSlotValue(const NodeDoc& node, const String& slot_id, const Value& def = Value()) {
	const WidgetSlotDoc* slot = FindSlot(node, slot_id);
	if(!slot)
		return def;
	int value_i = slot->properties.Find("value");
	Value v = value_i >= 0 ? slot->properties.GetValue(value_i) : Value();
	if(v.IsVoid() || IsNull(v))
		return def;
	return v;
}

Value NormalizeSlotValue(const WidgetSlotDoc& slot) {
	int value_i = slot.properties.Find("value");
	Value raw = value_i >= 0 ? slot.properties.GetValue(value_i) : Value();
	if(raw.IsVoid() || IsNull(raw))
		return Value();
	if(IsNumber(raw) || IsValueArray(raw) || IsValueMap(raw))
		return raw;

	String s = TrimBoth(raw.ToString());
	if(s.IsEmpty())
		return raw;

	bool b = false;
	if(ParseBoolText(s, b))
		return b;

	const char* iend = nullptr;
	int iv = ScanInt(~s, &iend);
	if(iend) {
		while(*iend && (byte)*iend <= ' ')
			iend++;
		if(*iend == 0)
			return iv;
	}

	const char* dend = nullptr;
	double dv = ScanDouble(~s, &dend);
	if(dend) {
		while(*dend && (byte)*dend <= ' ')
			dend++;
		if(*dend == 0)
			return dv;
	}

	return raw.ToString();
}

String GetSlotString(const NodeDoc& node, const String& slot_id, const String& def = String()) {
	Value v = GetSlotValue(node, slot_id, def);
	if(v.IsVoid() || IsNull(v))
		return def;
	return v.ToString();
}

int GetSlotInt(const NodeDoc& node, const String& slot_id, int def) {
	String s = TrimBoth(GetSlotString(node, slot_id, IntStr(def)));
	const char* end = nullptr;
	int x = ScanInt(~s, &end);
	if(!end)
		return x;
	while(*end && (byte)*end <= ' ')
		end++;
	return *end ? def : x;
}

bool ParseShape3(const String& shape, int& d, int& w, int& h) {
	String s = shape;
	s.Replace("[", "");
	s.Replace("]", "");
	s.Replace(" ", "");
	Vector<String> parts = Split(s, ',');
	if(parts.GetCount() < 3)
		return false;
	d = StrInt(parts[0]);
	w = StrInt(parts[1]);
	h = StrInt(parts[2]);
	return d > 0 && w > 0 && h > 0;
}

bool IsLegacyLayerNodeType(const String& type_id) {
	return type_id == "convnet.input" ||
	       type_id == "convnet.conv" ||
	       type_id == "convnet.pool" ||
	       type_id == "convnet.dense" ||
	       type_id == "convnet.relu" ||
	       type_id == "convnet.sigmoid" ||
	       type_id == "convnet.tanh";
}

bool IsCompilableLayerNodeType(const String& type_id) {
	return type_id.StartsWith(kLayerModulePrefix) || IsLegacyLayerNodeType(type_id);
}

bool IsCompileNodeType(const String& type_id) {
	return type_id == "convnet.compile";
}

bool IsTrainerNodeType(const String& type_id) {
	return type_id.StartsWith(kTrainerModulePrefix);
}

bool IsTrainNodeType(const String& type_id) {
	return type_id == "convnet.train";
}

int CountByKey(const VectorMap<String, int>& m, const String& k) {
	int i = m.Find(k);
	return i < 0 ? 0 : m[i];
}

bool BuildLayerStackCompileOrder(const Graph& graph,
                                 Vector<const NodeDoc*>& out,
                                 bool& used_stack_topology,
                                 String& error_out) {
	out.Clear();
	used_stack_topology = false;
	error_out.Clear();

	const GraphDoc& doc = graph.GetDoc();
	Vector<const NodeDoc*> layer_nodes;
	VectorMap<String, int> id_to_index;
	for(const NodeDoc& n : doc.nodes) {
		if(!IsCompilableLayerNodeType(n.node_type_id))
			continue;
		int idx = layer_nodes.GetCount();
		layer_nodes.Add(&n);
		id_to_index.Add(n.id, idx);
	}

	if(layer_nodes.IsEmpty())
		return true;

	Vector<Index<int>> adj;
	adj.SetCount(layer_nodes.GetCount());
	Vector<int> indegree;
	indegree.SetCount(layer_nodes.GetCount(), 0);

	int stack_edge_count = 0;
	for(const EdgeDoc& e : doc.edges) {
		int src_i = id_to_index.Find(e.source_node);
		int tgt_i = id_to_index.Find(e.target_node);
		if(src_i < 0 || tgt_i < 0)
			continue;
		const NodeDoc* src_node = layer_nodes[src_i];
		const NodeDoc* tgt_node = layer_nodes[tgt_i];
		const PinDoc* src_pin = FindPin(*src_node, e.source_pin, PinKind::Output);
		const PinDoc* tgt_pin = FindPin(*tgt_node, e.target_pin, PinKind::Input);
		if(!src_pin || !tgt_pin)
			continue;
		if(src_pin->type_name != "LAYER_STACK" || tgt_pin->type_name != "LAYER_STACK")
			continue;
		if(adj[src_i].Find(tgt_i) >= 0)
			continue;
		adj[src_i].Add(tgt_i);
		indegree[tgt_i]++;
		stack_edge_count++;
	}

	if(stack_edge_count == 0) {
		for(const NodeDoc* n : layer_nodes)
			out.Add(n);
		return true;
	}

	used_stack_topology = true;

	Vector<int> ready;
	ready.Reserve(layer_nodes.GetCount());
	for(int i = 0; i < indegree.GetCount(); i++)
		if(indegree[i] == 0)
			ready.Add(i);

	for(int at = 0; at < ready.GetCount(); at++) {
		int src_i = ready[at];
		out.Add(layer_nodes[src_i]);
		for(int next_i : adj[src_i]) {
			indegree[next_i]--;
			if(indegree[next_i] == 0)
				ready.Add(next_i);
		}
	}

	if(out.GetCount() != layer_nodes.GetCount()) {
		error_out = "Layer-stack graph contains cycle(s); cannot determine compile order.";
		return false;
	}
	return true;
}

void FillModuleRow(const NodeDoc& node, ValueMap& row) {
	for(const WidgetSlotDoc& s : node.slots) {
		Value v = NormalizeSlotValue(s);
		if(v.IsVoid() || IsNull(v))
			continue;
		row.GetAdd(s.id) = v;
	}
}

void ApplyTrainerParams(::ConvNet::TrainerBase& trainer, Value& row) {
	auto SetD = [&](const char* key, Function<void(double)> fn) {
		Value v = row.GetAdd(key);
		if(!v.IsNull())
			fn((double)v);
	};
	auto SetI = [&](const char* key, Function<void(int)> fn) {
		Value v = row.GetAdd(key);
		if(!v.IsNull())
			fn((int)v);
	};

	SetD("Beta1", [&](double v) { trainer.SetBeta1(v); });
	SetD("Beta2", [&](double v) { trainer.SetBeta2(v); });
	SetD("l1_decay", [&](double v) { trainer.SetL1Decay(v); });
	SetD("l2_decay", [&](double v) { trainer.SetL2Decay(v); });
	SetD("l1_decay_loss", [&](double v) { trainer.SetL1DecayLoss(v); });
	SetD("l2_decay_loss", [&](double v) { trainer.SetL2DecayLoss(v); });
	SetD("learning_rate", [&](double v) { trainer.SetLearningRate(v); });
	SetI("batch_size", [&](int v) { trainer.SetBatchSize(v); });
	SetD("momentum", [&](double v) { trainer.SetMomentum(v); });
	SetD("eps", [&](double v) { trainer.SetEps(v); });
	SetD("ro", [&](double v) { trainer.SetRo(v); });
}

} // namespace

String NeuralCompiler::Result::ToText() const {
	String out;
	out << (ok ? "Validation: OK\n" : "Validation: FAILED\n");
	if(errors.GetCount()) {
		out << "\nErrors:\n";
		for(const String& e : errors)
			out << " - " << e << "\n";
	}
	if(warnings.GetCount()) {
		out << "\nWarnings:\n";
		for(const String& w : warnings)
			out << " - " << w << "\n";
	}
	return out;
}

NeuralCompiler::Result NeuralCompiler::Validate(const Node::Graph& graph) {
	Result r;

	Vector<Node::ValidationMessage> vm = graph.Validate();
	for(const Node::ValidationMessage& m : vm) {
		if(m.severity == Node::ValidationMessage::ERROR)
			r.errors.Add(m.message);
		else
			r.warnings.Add(m.message);
	}

	const GraphDoc& doc = graph.GetDoc();

	Vector<const NodeDoc*> layer_nodes;
	VectorMap<String, int> layer_node_index;
	Index<String> compile_nodes;
	Index<String> trainer_nodes;
	Index<String> train_nodes;
	for(const NodeDoc& n : doc.nodes) {
		if(IsCompilableLayerNodeType(n.node_type_id)) {
			int idx = layer_nodes.GetCount();
			layer_nodes.Add(&n);
			layer_node_index.Add(n.id, idx);
		}
		if(IsCompileNodeType(n.node_type_id))
			compile_nodes.FindAdd(n.id);
		if(IsTrainerNodeType(n.node_type_id))
			trainer_nodes.FindAdd(n.id);
		if(IsTrainNodeType(n.node_type_id))
			train_nodes.FindAdd(n.id);
	}

	VectorMap<String, int> layer_stack_inputs;
	int layer_stack_sources = 0;
	Vector<int> layer_stack_indegree;
	Vector<Index<int>> layer_stack_adj;
	layer_stack_indegree.SetCount(layer_nodes.GetCount(), 0);
	layer_stack_adj.SetCount(layer_nodes.GetCount());

	VectorMap<String, int> compile_stack_inputs;
	VectorMap<String, int> compile_model_outputs;
	VectorMap<String, int> trainer_model_inputs;
	VectorMap<String, int> trainer_model_outputs;
	VectorMap<String, int> train_model_inputs;

	VectorMap<String, int> model_node_index;
	auto AddModelNode = [&](const String& id) {
		if(model_node_index.Find(id) < 0)
			model_node_index.Add(id, model_node_index.GetCount());
	};
	for(const String& id : compile_nodes) AddModelNode(id);
	for(const String& id : trainer_nodes) AddModelNode(id);
	for(const String& id : train_nodes) AddModelNode(id);
	Vector<Index<int>> model_adj;
	model_adj.SetCount(model_node_index.GetCount());

	for(const EdgeDoc& e : doc.edges) {
		const NodeDoc* src_node = graph.FindNode(e.source_node);
		const NodeDoc* tgt_node = graph.FindNode(e.target_node);
		if(!src_node || !tgt_node)
			continue;

		const PinDoc* src = FindPin(*src_node, e.source_pin, PinKind::Output);
		const PinDoc* tgt = FindPin(*tgt_node, e.target_pin, PinKind::Input);
		if(!src || !tgt)
			continue;

		if(src->type_name != tgt->type_name) {
			r.errors.Add("NeuralFormat mismatch at edge " + e.id + ": "
			           + e.source_node + "." + e.source_pin + " (" + src->type_name + ") -> "
			           + e.target_node + "." + e.target_pin + " (" + tgt->type_name + ")");
		}

		if(src->type_name == "LAYER_STACK" && tgt->type_name == "LAYER_STACK") {
			String target_key = e.target_node + ":" + e.target_pin;
			layer_stack_inputs.GetAdd(target_key, 0)++;
			layer_stack_sources++;

			int src_i = layer_node_index.Find(e.source_node);
			int tgt_i = layer_node_index.Find(e.target_node);
			if(src_i >= 0 && tgt_i >= 0) {
				if(layer_stack_adj[src_i].Find(tgt_i) < 0) {
					layer_stack_adj[src_i].Add(tgt_i);
					layer_stack_indegree[tgt_i]++;
				}
			}

			if(IsCompileNodeType(tgt_node->node_type_id))
				compile_stack_inputs.GetAdd(tgt_node->id, 0)++;
		}
		else if(src->type_name == "LAYER_STACK" || tgt->type_name == "LAYER_STACK") {
			r.errors.Add("Layer-stack flow mismatch at edge " + e.id + ": "
			           + e.source_node + "." + e.source_pin + " -> "
			           + e.target_node + "." + e.target_pin);
		}

		if(src->type_name == "MODEL" && tgt->type_name == "MODEL") {
			if(IsCompileNodeType(src_node->node_type_id))
				compile_model_outputs.GetAdd(src_node->id, 0)++;
			if(IsTrainerNodeType(tgt_node->node_type_id))
				trainer_model_inputs.GetAdd(tgt_node->id, 0)++;
			if(IsTrainerNodeType(src_node->node_type_id))
				trainer_model_outputs.GetAdd(src_node->id, 0)++;
			if(IsTrainNodeType(tgt_node->node_type_id))
				train_model_inputs.GetAdd(tgt_node->id, 0)++;

			int model_src_i = -1;
			int model_tgt_i = -1;
			int model_src_pos = model_node_index.Find(src_node->id);
			int model_tgt_pos = model_node_index.Find(tgt_node->id);
			if(model_src_pos >= 0)
				model_src_i = model_node_index[model_src_pos];
			if(model_tgt_pos >= 0)
				model_tgt_i = model_node_index[model_tgt_pos];
			if(model_src_i >= 0 && model_tgt_i >= 0 && model_adj[model_src_i].Find(model_tgt_i) < 0)
				model_adj[model_src_i].Add(model_tgt_i);
		}
	}

	for(int i = 0; i < layer_stack_inputs.GetCount(); i++) {
		if(layer_stack_inputs[i] > 1) {
			r.errors.Add("Layer-stack input has multiple upstream connections: "
			           + layer_stack_inputs.GetKey(i));
		}
	}
	if(layer_stack_sources == 0)
		r.warnings.Add("No LAYER_STACK edges found; compiler will use node order fallback.");

	String topo_error;
	Vector<const NodeDoc*> compile_order;
	bool used_stack_topology = false;
	if(!BuildLayerStackCompileOrder(graph, compile_order, used_stack_topology, topo_error) && !topo_error.IsEmpty())
		r.errors.Add(topo_error);

	if(used_stack_topology && layer_nodes.GetCount()) {
		Vector<int> roots;
		for(int i = 0; i < layer_nodes.GetCount(); i++) {
			if(layer_stack_indegree[i] == 0)
				roots.Add(i);
		}

		if(roots.GetCount() > 1) {
			String msg = "Layer-stack flow has multiple roots:";
			for(int i : roots)
				msg << " " << layer_nodes[i]->id;
			r.errors.Add(msg);
		}

		if(roots.GetCount()) {
			Index<int> visited;
			Vector<int> queue;
			queue.Add(roots[0]);
			visited.Add(roots[0]);
			for(int at = 0; at < queue.GetCount(); at++) {
				int src_i = queue[at];
				for(int nxt : layer_stack_adj[src_i]) {
					if(visited.Find(nxt) < 0) {
						visited.Add(nxt);
						queue.Add(nxt);
					}
				}
			}

			for(int i = 0; i < layer_nodes.GetCount(); i++) {
				if(visited.Find(i) < 0) {
					r.errors.Add("Layer node is unreachable in LAYER_STACK chain: " + layer_nodes[i]->id);
				}
			}
		}
	}

	if(!layer_nodes.IsEmpty() && compile_nodes.IsEmpty())
		r.warnings.Add("Graph has ConvNet layers but no convnet.compile node.");

	for(const String& id : compile_nodes) {
		int in_count = CountByKey(compile_stack_inputs, id);
		if(in_count == 0)
			r.warnings.Add("Compile node has no LAYER_STACK input: " + id);
		else if(in_count > 1)
			r.errors.Add("Compile node has multiple LAYER_STACK inputs: " + id);

		if(CountByKey(compile_model_outputs, id) == 0)
			r.warnings.Add("Compile node output MODEL is not connected: " + id);
	}

	for(const String& id : trainer_nodes) {
		if(CountByKey(trainer_model_inputs, id) == 0)
			r.warnings.Add("Trainer node has no MODEL input: " + id);
		if(CountByKey(trainer_model_outputs, id) == 0)
			r.warnings.Add("Trainer node has no MODEL output: " + id);
	}

	for(const String& id : train_nodes) {
		if(CountByKey(train_model_inputs, id) == 0)
			r.warnings.Add("Train node has no MODEL input: " + id);
	}

	if(!compile_nodes.IsEmpty() && !train_nodes.IsEmpty() && model_node_index.GetCount()) {
		Index<int> reached;
		Vector<int> queue;
		for(const String& id : compile_nodes) {
			int pos = model_node_index.Find(id);
			int idx = pos < 0 ? -1 : model_node_index[pos];
			if(idx >= 0 && reached.Find(idx) < 0) {
				reached.Add(idx);
				queue.Add(idx);
			}
		}

		for(int at = 0; at < queue.GetCount(); at++) {
			int src_i = queue[at];
			for(int nxt : model_adj[src_i]) {
				if(reached.Find(nxt) < 0) {
					reached.Add(nxt);
					queue.Add(nxt);
				}
			}
		}

		bool has_compile_to_train = false;
		for(const String& id : train_nodes) {
			int pos = model_node_index.Find(id);
			int idx = pos < 0 ? -1 : model_node_index[pos];
			if(idx >= 0 && reached.Find(idx) >= 0) {
				has_compile_to_train = true;
				break;
			}
		}
		if(!has_compile_to_train)
			r.errors.Add("No MODEL path from convnet.compile to convnet.train.");
	}

	r.ok = r.errors.IsEmpty();
	return r;
}

bool NeuralCompiler::BuildSession(const Node::Graph& graph, ::ConvNet::Session& session, String& error_out) {
	session.Clear();
	::ConvNet::RegisterBuiltinSessionModules();

	bool has_any_layer = false;
	bool has_input_layer = false;

	auto CompileLayerNode = [&](const Node::NodeDoc& n) -> bool {
		if(n.node_type_id.StartsWith(kLayerModulePrefix)) {
			String module_type = n.node_type_id.Mid((int)strlen(kLayerModulePrefix));
			if(module_type.IsEmpty()) {
				error_out = "Invalid layer module node type: " + n.node_type_id;
				return false;
			}

			ValueMap row_map;
			row_map.Add("type", module_type);
			FillModuleRow(n, row_map);
			Value row = row_map;
			String module_error;
			int layer_apply = ::ConvNet::SessionModuleRegistry::Get().ApplyLayer(module_type, session, row, module_error);
			if(layer_apply <= 0) {
				error_out = layer_apply < 0 ? module_error : ("Unknown layer module: " + module_type);
				return false;
			}

			has_any_layer = true;
			if(module_type == "input")
				has_input_layer = true;
			return true;
		}

		if(n.node_type_id == "convnet.input") {
			int d = 1, w = 28, h = 28;
			String s = GetSlotString(n, "input_shape_str", "[1,28,28]");
			ParseShape3(s, d, w, h);
			session.AddInputLayer(w, h, d);
			has_any_layer = true;
			has_input_layer = true;
			return true;
		}
		else if(n.node_type_id == "convnet.conv") {
			int k = max(1, GetSlotInt(n, "kernel_size", 3));
			int out = max(1, GetSlotInt(n, "out_channels", 16));
			int stride = max(1, GetSlotInt(n, "stride", 1));
			int pad = max(0, GetSlotInt(n, "padding", 1));
			session.AddConvLayer(k, k, out, 0.0, 1.0, stride, pad, 0.0);
			has_any_layer = true;
			return true;
		}
		else if(n.node_type_id == "convnet.pool") {
			int k = max(1, GetSlotInt(n, "kernel_size", 2));
			int stride = max(1, GetSlotInt(n, "stride", 2));
			int pad = max(0, GetSlotInt(n, "padding", 0));
			session.AddPoolLayer(k, k, stride, pad);
			has_any_layer = true;
			return true;
		}
		else if(n.node_type_id == "convnet.dense") {
			int neurons = max(1, GetSlotInt(n, "num_nodes", 128));
			session.AddFullyConnLayer(neurons);
			has_any_layer = true;
			return true;
		}
		else if(n.node_type_id == "convnet.relu") {
			session.AddReluLayer();
			has_any_layer = true;
			return true;
		}
		else if(n.node_type_id == "convnet.sigmoid") {
			session.AddSigmoidLayer();
			has_any_layer = true;
			return true;
		}
		else if(n.node_type_id == "convnet.tanh") {
			session.AddTanhLayer();
			has_any_layer = true;
			return true;
		}
		return true;
	};

	Vector<const NodeDoc*> layer_compile_order;
	bool used_stack_topology = false;
	if(!BuildLayerStackCompileOrder(graph, layer_compile_order, used_stack_topology, error_out))
		return false;
	for(const NodeDoc* node_ptr : layer_compile_order) {
		if(!node_ptr)
			continue;
		if(!CompileLayerNode(*node_ptr))
			return false;
	}

	for(const Node::NodeDoc& n : graph.GetDoc().nodes) {
		if(!n.node_type_id.StartsWith(kTrainerModulePrefix))
			continue;
		String trainer_type = n.node_type_id.Mid((int)strlen(kTrainerModulePrefix));
		if(trainer_type.IsEmpty()) {
			error_out = "Invalid trainer module node type: " + n.node_type_id;
			return false;
		}

		ValueMap row_map;
		row_map.Add("type", trainer_type);
		FillModuleRow(n, row_map);
		Value row = row_map;
		String module_error;
		int trainer_apply = ::ConvNet::SessionModuleRegistry::Get().ApplyTrainer(trainer_type, session.GetTrainer(), row, module_error);
		if(trainer_apply <= 0) {
			error_out = trainer_apply < 0 ? module_error : ("Unknown trainer module: " + trainer_type);
			return false;
		}

		ApplyTrainerParams(session.GetTrainer(), row);
	}

	if(!has_any_layer) {
		error_out = "No compilable ConvNet layer nodes found in the graph.";
		return false;
	}
	if(!has_input_layer) {
		session.AddInputLayer(28, 28, 1);
	}
	return true;
}

bool NeuralCompiler::RunSelfTests(String& report_out) {
	using namespace Node;

	auto AddTestPin = [](NodeDoc& n, const String& id, PinKind kind, const String& type) {
		PinDoc& p = n.pins.Add();
		p.id = id;
		p.label = id;
		p.kind = kind;
		p.type_name = type;
	};

	auto AddTestLayer = [&](Graph& g, const String& id, const String& type, bool in_stack, bool out_stack) {
		NodeDoc& n = g.GetDoc().nodes.Add();
		n.id = id;
		n.label = id;
		n.node_type_id = type;
		if(in_stack)
			AddTestPin(n, "layer_stack", PinKind::Input, "LAYER_STACK");
		if(out_stack)
			AddTestPin(n, "layer_stack", PinKind::Output, "LAYER_STACK");
	};

	auto AddTestEdge = [&](Graph& g, const String& id,
	                       const String& src_node, const String& src_pin,
	                       const String& tgt_node, const String& tgt_pin) {
		EdgeDoc& e = g.GetDoc().edges.Add();
		e.id = id;
		e.source_node = src_node;
		e.source_pin = src_pin;
		e.target_node = tgt_node;
		e.target_pin = tgt_pin;
	};

	int pass = 0;
	int fail = 0;
	String out;

	auto Test = [&](const String& name, Function<bool(String&)> fn) {
		String detail;
		bool ok = fn(detail);
		if(ok) {
			pass++;
			out << "[PASS] " << name << "\n";
		}
		else {
			fail++;
			out << "[FAIL] " << name;
			if(!detail.IsEmpty())
				out << ": " << detail;
			out << "\n";
		}
	};

	Test("compile-order-linear", [&](String& detail) {
		Graph g;
		AddTestLayer(g, "n_input", "convnet.module.layer.input", false, true);
		AddTestLayer(g, "n_fc", "convnet.module.layer.fc", true, true);
		AddTestLayer(g, "n_relu", "convnet.module.layer.relu", true, true);
		AddTestEdge(g, "e1", "n_input", "layer_stack", "n_fc", "layer_stack");
		AddTestEdge(g, "e2", "n_fc", "layer_stack", "n_relu", "layer_stack");
		g.RebuildIndexPublic();

		Vector<const NodeDoc*> order;
		bool used = false;
		String err;
		if(!BuildLayerStackCompileOrder(g, order, used, err)) {
			detail = err;
			return false;
		}
		if(!used) {
			detail = "expected LAYER_STACK topology mode";
			return false;
		}
		if(order.GetCount() != 3 ||
		   order[0]->id != "n_input" ||
		   order[1]->id != "n_fc" ||
		   order[2]->id != "n_relu") {
			detail = "unexpected order";
			return false;
		}
		return true;
	});

	Test("compile-order-cycle-detect", [&](String& detail) {
		Graph g;
		AddTestLayer(g, "a", "convnet.module.layer.fc", true, true);
		AddTestLayer(g, "b", "convnet.module.layer.relu", true, true);
		AddTestEdge(g, "e1", "a", "layer_stack", "b", "layer_stack");
		AddTestEdge(g, "e2", "b", "layer_stack", "a", "layer_stack");
		g.RebuildIndexPublic();

		Vector<const NodeDoc*> order;
		bool used = false;
		String err;
		if(BuildLayerStackCompileOrder(g, order, used, err)) {
			detail = "expected cycle failure";
			return false;
		}
		if(err.Find("cycle") < 0 && err.Find("Cycle") < 0) {
			detail = "missing cycle diagnostics";
			return false;
		}
		return true;
	});

	Test("compile-order-fallback-node-order", [&](String& detail) {
		Graph g;
		AddTestLayer(g, "a", "convnet.module.layer.input", false, true);
		AddTestLayer(g, "b", "convnet.module.layer.fc", true, true);
		g.RebuildIndexPublic();

		Vector<const NodeDoc*> order;
		bool used = false;
		String err;
		if(!BuildLayerStackCompileOrder(g, order, used, err)) {
			detail = err;
			return false;
		}
		if(used) {
			detail = "expected node-order fallback";
			return false;
		}
		if(order.GetCount() != 2 || order[0]->id != "a" || order[1]->id != "b") {
			detail = "unexpected fallback order";
			return false;
		}
		return true;
	});

	out << "Summary: " << pass << " passed, " << fail << " failed\n";
	report_out = out;
	return fail == 0;
}

END_UPP_NAMESPACE
