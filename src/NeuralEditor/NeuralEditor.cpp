#include "NeuralEditor.h"
#include <Node/Script/Script.h>

NAMESPACE_UPP

namespace {

Color PinColor(const String& type_name) {
	if(type_name == "LAYER_STACK") return Color(105, 185, 255);
	if(type_name == "MODEL")       return Color(175, 100, 255);
	if(type_name == "METRICS")     return Color(255, 165, 110);
	if(type_name == "REPORT")      return Color(235, 185, 115);
	if(type_name == "STRING")      return Color(200, 200, 200);
	return Color(160, 160, 160);
}

void AddPin(Node::NodeDoc& n, const String& id, Node::PinKind kind, const String& type_name) {
	Node::PinDoc& p = n.pins.Add();
	p.id = id;
	p.label = id;
	p.kind = kind;
	p.type_name = type_name;
	p.color = PinColor(type_name);
}

void AddSlot(Node::NodeDoc& n, const String& id, const String& type, const Value& value) {
	Node::WidgetSlotDoc& s = n.slots.Add();
	s.id = id;
	s.type = type;
	s.properties.Add("value", value);
}

String ExtractEdgeIdFromMessage(const String& msg) {
	String marker = " at edge ";
	int at = msg.Find(marker);
	if(at < 0)
		return String();
	String rest = msg.Mid(at + marker.GetCount());
	int end = rest.Find(':');
	if(end < 0)
		end = rest.Find(' ');
	if(end < 0)
		end = rest.GetCount();
	return TrimBoth(rest.Left(end));
}

String ExtractNodeIdFromMessage(const Node::Graph& graph, const String& msg) {
	Vector<String> tokens = Split(msg, ' ');
	for(String tok : tokens) {
		tok = TrimBoth(tok);
		while(tok.GetCount() && (tok[0] == '(' || tok[0] == '[' || tok[0] == '"'))
			tok = tok.Mid(1);
		while(tok.GetCount() && (tok.EndsWith(":") || tok.EndsWith(",") || tok.EndsWith(".") || tok.EndsWith(")") || tok.EndsWith("]") || tok.EndsWith("\"")))
			tok = tok.Left(tok.GetCount() - 1);
		if(tok.IsEmpty())
			continue;
		if(graph.FindNode(tok))
			return tok;
		int dot = tok.Find('.');
		if(dot > 0) {
			String left = tok.Left(dot);
			if(graph.FindNode(left))
				return left;
		}
	}
	return String();
}

struct TemplateDef : Moveable<TemplateDef> {
	String type_id;
	String category;
	String label;
	Node::NodeDoc doc;
};

class NeuralDomain : public INodeWorkbenchDomain {
	Vector<TemplateDef> templates;
	bool palette_ready = false;

	void EnsurePalette() {
		if(palette_ready)
			return;
		palette_ready = true;
		templates.Clear();

		auto Add = [&](const String& type_id, const String& category, const String& label,
		               Function<void(Node::NodeDoc&)> fn) {
			TemplateDef& t = templates.Add();
			t.type_id = type_id;
			t.category = category;
			t.label = label;
			t.doc.node_type_id = type_id;
			t.doc.category = category;
			t.doc.label = label;
			t.doc.fill_clr = Color(42, 46, 56);
			t.doc.line_clr = Color(88, 96, 116);
			t.doc.sz = Sizef(260, 80);
			fn(t.doc);
		};

		::ConvNet::RegisterBuiltinSessionModules();
		const ::ConvNet::SessionModuleRegistry& module_reg = ::ConvNet::SessionModuleRegistry::Get();
		for(const ::ConvNet::SessionLayerModuleSpec& spec : module_reg.GetLayerSpecs()) {
			String type_id = "convnet.module.layer." + spec.type;
			String label = "Layer: " + spec.label;
			Add(type_id, "convnet.layers.modules", label, [=](Node::NodeDoc& n) {
				if(spec.has_layer_stack_input)
					AddPin(n, "layer_stack", Node::PinKind::Input, "LAYER_STACK");
				if(spec.has_layer_stack_output)
					AddPin(n, "layer_stack", Node::PinKind::Output, "LAYER_STACK");
				for(const ::ConvNet::SessionModuleParamSpec& p : spec.params)
					AddSlot(n, p.key, p.widget.IsEmpty() ? String("EditString") : p.widget, p.default_value);
			});
		}

		for(const ::ConvNet::SessionTrainerModuleSpec& spec : module_reg.GetTrainerSpecs()) {
			String type_id = "convnet.module.trainer." + spec.type;
			String label = "Trainer: " + spec.label;
			Add(type_id, "convnet.trainers.modules", label, [=](Node::NodeDoc& n) {
				AddPin(n, "model", Node::PinKind::Input, "MODEL");
				AddPin(n, "model", Node::PinKind::Output, "MODEL");
				for(const ::ConvNet::SessionModuleParamSpec& p : spec.params)
					AddSlot(n, p.key, p.widget.IsEmpty() ? String("EditString") : p.widget, p.default_value);
			});
		}

		Add("convnet.compile", "convnet.build", "Compile Model", [&](Node::NodeDoc& n) {
			AddPin(n, "layer_stack", Node::PinKind::Input, "LAYER_STACK");
			AddPin(n, "model", Node::PinKind::Output, "MODEL");
			AddPin(n, "report", Node::PinKind::Output, "REPORT");
			AddSlot(n, "mode", "DropList", "compile");
		});

		Add("convnet.train", "convnet.runtime", "Train", [&](Node::NodeDoc& n) {
			AddPin(n, "model", Node::PinKind::Input, "MODEL");
			AddPin(n, "metrics", Node::PinKind::Output, "METRICS");
			AddSlot(n, "epochs", "EditIntSpin", 20);
			AddSlot(n, "learning_rate", "EditDoubleSpin", 0.001);
		});
	}

public:
	virtual String GetDomainId() const override   { return "neural"; }
	virtual String GetDomainName() const override { return "Neural"; }
	virtual String GetDomainDesc() const override { return "ConvNet graph compiler domain"; }

	virtual String GetGraphFileFilter() const override {
		return "Neural Graph (*.grf *.nngrf)\t*.grf *.nngrf";
	}
	virtual String GetProjectFileFilter() const override {
		return "Neural Project (*.grfproj *.nnprj)\t*.grfproj *.nnprj";
	}
	virtual String GetSolutionFileFilter() const override {
		return "Neural Solution (*.slnx *.sln *.nnsln)\t*.slnx *.sln *.nnsln";
	}
	virtual String GetExtraExtensions() const override { return ".nngrf|.nnprj|.nnsln|.nnpy"; }

	virtual void OnDomainInit(NodeWorkbenchWindow& host) override {
		EnsurePalette();
		for(const TemplateDef& t : templates) {
			host.GetViewport().RegisterNodeType(
				t.type_id,
				t.label,
				[doc = t.doc]() mutable {
					Node::NodeDoc out;
					out <<= doc;
					return out;
				});
		}

		// Register neural.pyvm node type (Phase 3 Task 02)
		{
			Node::NodeDoc pyvm_doc;
			pyvm_doc.node_type_id = "neural.pyvm";
			pyvm_doc.category     = "neural.script";
			pyvm_doc.label        = "Python Script";
			pyvm_doc.fill_clr     = Color(38, 48, 38);
			pyvm_doc.line_clr     = Color(80, 120, 80);
			pyvm_doc.sz           = Sizef(260, 110);
			AddPin(pyvm_doc, "data_in",  Node::PinKind::Input,  "STRING");
			AddPin(pyvm_doc, "data_out", Node::PinKind::Output, "STRING");
			AddSlot(pyvm_doc, "script_path", "EditString", String());
			AddSlot(pyvm_doc, "entry_fn",    "EditString", String("run"));
			AddSlot(pyvm_doc, "code",        "EditString", String());
			host.GetViewport().RegisterNodeType(
				"neural.pyvm", "Python Script",
				[doc = pyvm_doc]() mutable {
					Node::NodeDoc out;
					out <<= doc;
					return out;
				});
		}
	}

	virtual void BuildPalette(Vector<PaletteItem>& palette_out) override {
		EnsurePalette();
		palette_out.Clear();
		for(const TemplateDef& t : templates) {
			PaletteItem& p = palette_out.Add();
			p.category = t.category;
			p.label = t.label;
			p.type_id = t.type_id;
		}
		// Add python script node
		PaletteItem& py = palette_out.Add();
		py.category = "neural.script";
		py.label    = "Python Script";
		py.type_id  = "neural.pyvm";
	}

	virtual void ValidateGraph(NodeWorkbenchWindow& host,
	                           Vector<WorkbenchDiagnostic>& diag_out) override {
		diag_out.Clear();
		const Node::Graph& graph = host.GetGraph();
		NeuralCompiler::Result r = NeuralCompiler::Validate(graph);

		for(const String& msg : r.errors) {
			WorkbenchDiagnostic& d = diag_out.Add();
			d.severity = DiagSeverity::Error;
			d.message = msg;
			d.source = "neural.validate";
			String edge_id = ExtractEdgeIdFromMessage(msg);
			d.entity_id = !edge_id.IsEmpty() ? edge_id : ExtractNodeIdFromMessage(graph, msg);
		}
		for(const String& msg : r.warnings) {
			WorkbenchDiagnostic& d = diag_out.Add();
			d.severity = DiagSeverity::Warning;
			d.message = msg;
			d.source = "neural.validate";
			String edge_id = ExtractEdgeIdFromMessage(msg);
			d.entity_id = !edge_id.IsEmpty() ? edge_id : ExtractNodeIdFromMessage(graph, msg);
		}

		// Validate neural.pyvm nodes: check script_path exists or inline code present
		IScriptRuntime* rt = host.GetScriptRuntime();
		for(const Node::NodeDoc& n : graph.GetDoc().nodes) {
			if(n.node_type_id != "neural.pyvm") continue;
			String script_path, entry_fn, code;
			for(const Node::WidgetSlotDoc& s : n.slots) {
				int vi = s.properties.Find("value");
				Value v = vi >= 0 ? s.properties.GetValue(vi) : Value();
				if(s.id == "script_path") script_path = v.IsVoid() ? String() : v.ToString();
				if(s.id == "entry_fn")    entry_fn    = v.IsVoid() ? String() : v.ToString();
				if(s.id == "code")        code        = v.IsVoid() ? String() : v.ToString();
			}
			bool has_source = !script_path.IsEmpty() || !code.IsEmpty();
			if(!has_source) {
				WorkbenchDiagnostic& d = diag_out.Add();
				d.severity  = DiagSeverity::Warning;
				d.message   = "Python node has no script_path or inline code: " + n.id;
				d.source    = "neural.pyvm";
				d.entity_id = n.id;
			}
			if(!script_path.IsEmpty() && !FileExists(script_path)) {
				WorkbenchDiagnostic& d = diag_out.Add();
				d.severity  = DiagSeverity::Error;
				d.message   = "script_path not found: " + script_path + " (node: " + n.id + ")";
				d.source    = "neural.pyvm";
				d.entity_id = n.id;
			}
			if(entry_fn.IsEmpty()) {
				WorkbenchDiagnostic& d = diag_out.Add();
				d.severity  = DiagSeverity::Warning;
				d.message   = "Python node entry_fn is empty, will use 'run': " + n.id;
				d.source    = "neural.pyvm";
				d.entity_id = n.id;
			}
			if(!rt) {
				WorkbenchDiagnostic& d = diag_out.Add();
				d.severity  = DiagSeverity::Warning;
				d.message   = "No script runtime registered; Python node will not execute: " + n.id;
				d.source    = "neural.pyvm";
				d.entity_id = n.id;
			}
		}
	}

	virtual bool CompileGraph(NodeWorkbenchWindow& host, String& log_out) override {
		::ConvNet::Session session;
		String err;
		if(!NeuralCompiler::BuildSession(host.GetGraph(), session, err)) {
			log_out = "BuildSession failed: " + err;
			return false;
		}
		log_out = "BuildSession OK.";
		return true;
	}

	virtual bool RunGraph(NodeWorkbenchWindow& host, String& log_out) override {
		::ConvNet::Session session;
		String err;
		if(!NeuralCompiler::BuildSession(host.GetGraph(), session, err)) {
			log_out = "Run failed (build): " + err;
			return false;
		}

		// Read epochs / learning_rate from the first convnet.train node
		int epochs = 1;
		double lr = 0.01;
		const Node::Graph& graph = host.GetGraph();
		for(const Node::NodeDoc& n : graph.GetDoc().nodes) {
			if(n.node_type_id == "convnet.train") {
				for(const Node::WidgetSlotDoc& s : n.slots) {
					if(s.id == "epochs") {
						int vi = s.properties.Find("value");
						if(vi >= 0) epochs = max(1, (int)s.properties.GetValue(vi));
					}
					if(s.id == "learning_rate") {
						int vi = s.properties.Find("value");
						if(vi >= 0) lr = (double)s.properties.GetValue(vi);
					}
				}
				break;
			}
		}
		if(lr > 0.0)
			session.GetTrainer().SetLearningRate(lr);

		// Determine input dimensions from the first (input) layer.
		int in_w = 1, in_h = 1, in_d = 1;
		if(session.GetLayerCount() > 0) {
			::ConvNet::LayerBase& first = session.GetLayer(0);
			in_w = first.input_width  > 0 ? first.input_width  : 1;
			in_h = first.input_height > 0 ? first.input_height : 1;
			in_d = first.input_depth  > 0 ? first.input_depth  : 1;
		}

		String run_log;
		run_log << "Session built. Layers: " << session.GetLayerCount()
		        << "  Input: " << in_w << "x" << in_h << "x" << in_d << "\n";

		// Dummy supervised training: single zero-input sample, class label 0.
		::ConvNet::Volume input_vol(in_w, in_h, in_d, 0.0);
		Vector<double> y;
		y.Add(0.0); // single regression target; works with regression/softmax layers

		double total_loss = 0.0;
		for(int ep = 0; ep < epochs; ep++) {
			session.TrainOnce(input_vol, y);
			double loss = session.GetLossWindow().GetAverage();
			total_loss += loss;
			if(epochs <= 10 || (ep % max(1, epochs / 10)) == 0)
				run_log << "Epoch " << (ep + 1) << "/" << epochs << "  loss=" << Format("%.6f", loss) << "\n";
		}

		run_log << "Avg loss: " << Format("%.6f", epochs > 0 ? total_loss / epochs : 0.0) << "\n";
		run_log << "Done.";
		log_out = run_log;
		return true;
	}

	virtual Vector<String> GetQuickFixes(const WorkbenchDiagnostic& diag) override {
		Vector<String> fixes;
		const String& msg = diag.message;
		if(msg.Find("NeuralFormat mismatch") >= 0 || msg.Find("Layer-stack flow mismatch") >= 0)
			fixes.Add("Remove bad edge");
		if(msg.Find("multiple LAYER_STACK inputs") >= 0)
			fixes.Add("Remove duplicate edge");
		if(msg.Find("multiple upstream connections") >= 0)
			fixes.Add("Remove extra connection");
		if(msg.Find("multiple roots") >= 0)
			fixes.Add("Remove extra root edge");
		return fixes;
	}

	virtual void ApplyQuickFix(NodeWorkbenchWindow& host, const WorkbenchDiagnostic& diag,
	                           int fix_index) override {
		// For edge-removal fixes: entity_id is the edge id extracted from the error message.
		String entity = diag.entity_id;
		if(entity.IsEmpty())
			return;

		Node::Graph& graph = host.GetGraph();
		const Node::GraphDoc& doc = graph.GetDoc();

		// Check if entity_id is an edge id
		for(int i = 0; i < doc.edges.GetCount(); i++) {
			if(doc.edges[i].id == entity) {
				// Remove the edge from the doc
				graph.GetDoc().edges.Remove(i);
				graph.RebuildIndexPublic();
				graph.Invalidate();
				host.GetViewport().SetGraph(graph);
				host.ValidateGraph();
				return;
			}
		}
	}

	virtual void GetTemplates(Vector<TemplateDesc>& out) override {
		auto Add = [&](const char* name, const char* cat, const char* desc) {
			TemplateDesc& t = out.Add();
			t.name = name;
			t.category = cat;
			t.description = desc;
		};
		Add("Classify 2D",    "neural.classification", "Two-class 2D point classifier (input→fc→relu→softmax)");
		Add("Regression 1D",  "neural.regression",     "Scalar regression (input→fc→relu→fc→regression)");
		Add("MNIST Classifier","neural.classification", "Image classifier for 28×28 grayscale (conv→pool→fc→softmax)");
	}

	virtual String GenerateTemplate(int index, const String& dest_dir, String& error_out) override {
		struct NodeEntry : Moveable<NodeEntry> { String id, type; VectorMap<String,Value> slots; };
		struct EdgeEntry : Moveable<EdgeEntry> { String id, src_node, src_pin, tgt_node, tgt_pin; };
		struct TplDef {
			String name;
			Vector<NodeEntry> nodes;
			Vector<EdgeEntry> edges;
		};

		TplDef def;

		// Helper: add a slot override to a NodeEntry
		auto S = [](NodeEntry& n, const char* k, const Value& v) -> NodeEntry& {
			n.slots.GetAdd(k) = v;
			return n;
		};
		// Helper: add an EdgeEntry
		auto AE = [&](const char* id,
		              const char* sn, const char* sp,
		              const char* tn, const char* tp) {
			EdgeEntry& e = def.edges.Add();
			e.id = id; e.src_node = sn; e.src_pin = sp;
			e.tgt_node = tn; e.tgt_pin = tp;
		};
		// Helper: add a NodeEntry and return ref
		auto AN = [&](const char* id, const char* type) -> NodeEntry& {
			NodeEntry& n = def.nodes.Add();
			n.id = id; n.type = type;
			return n;
		};
		(void)S; // suppress unused warning if only one template uses it

		if(index == 0) {
			// --- Classify 2D ---
			def.name = "Classify2D";
			{ NodeEntry& n = AN("n_input",   "convnet.module.layer.input");  S(n,"input_width",1); S(n,"input_height",1); S(n,"input_depth",2); }
			{ NodeEntry& n = AN("n_fc1",     "convnet.module.layer.fc");     S(n,"neuron_count",6); }
			AN("n_relu1",   "convnet.module.layer.relu");
			{ NodeEntry& n = AN("n_fc2",     "convnet.module.layer.fc");     S(n,"neuron_count",2); }
			AN("n_relu2",   "convnet.module.layer.relu");
			{ NodeEntry& n = AN("n_softmax", "convnet.module.layer.softmax"); S(n,"class_count",2); }
			{ NodeEntry& n = AN("n_compile", "convnet.compile");              S(n,"mode",String("compile")); }
			{ NodeEntry& n = AN("n_sgd",     "convnet.module.trainer.sgd");   S(n,"learning_rate",0.01); S(n,"momentum",0.1); S(n,"batch_size",10); S(n,"l2_decay",0.001); }
			{ NodeEntry& n = AN("n_train",   "convnet.train");                S(n,"epochs",40); S(n,"learning_rate",0.01); }
			AE("e1","n_input","layer_stack","n_fc1","layer_stack");
			AE("e2","n_fc1","layer_stack","n_relu1","layer_stack");
			AE("e3","n_relu1","layer_stack","n_fc2","layer_stack");
			AE("e4","n_fc2","layer_stack","n_relu2","layer_stack");
			AE("e5","n_relu2","layer_stack","n_softmax","layer_stack");
			AE("e6","n_softmax","layer_stack","n_compile","layer_stack");
			AE("e7","n_compile","model","n_sgd","model");
			AE("e8","n_sgd","model","n_train","model");
		}
		else if(index == 1) {
			// --- Regression 1D ---
			def.name = "Regression1D";
			{ NodeEntry& n = AN("n_input",      "convnet.module.layer.input");      S(n,"input_width",1); S(n,"input_height",1); S(n,"input_depth",1); }
			{ NodeEntry& n = AN("n_fc1",        "convnet.module.layer.fc");         S(n,"neuron_count",40); }
			AN("n_relu1",      "convnet.module.layer.relu");
			{ NodeEntry& n = AN("n_fc2",        "convnet.module.layer.fc");         S(n,"neuron_count",1); }
			AN("n_regression", "convnet.module.layer.regression");
			{ NodeEntry& n = AN("n_compile",    "convnet.compile");                 S(n,"mode",String("compile")); }
			{ NodeEntry& n = AN("n_sgd",        "convnet.module.trainer.sgd");      S(n,"learning_rate",0.01); S(n,"momentum",0.9); S(n,"batch_size",16); }
			{ NodeEntry& n = AN("n_train",      "convnet.train");                   S(n,"epochs",50); S(n,"learning_rate",0.01); }
			AE("e1","n_input","layer_stack","n_fc1","layer_stack");
			AE("e2","n_fc1","layer_stack","n_relu1","layer_stack");
			AE("e3","n_relu1","layer_stack","n_fc2","layer_stack");
			AE("e4","n_fc2","layer_stack","n_regression","layer_stack");
			AE("e5","n_regression","layer_stack","n_compile","layer_stack");
			AE("e6","n_compile","model","n_sgd","model");
			AE("e7","n_sgd","model","n_train","model");
		}
		else if(index == 2) {
			// --- MNIST Classifier ---
			def.name = "MNISTClassifier";
			{ NodeEntry& n = AN("n_input",   "convnet.module.layer.input");  S(n,"input_width",28); S(n,"input_height",28); S(n,"input_depth",1); }
			{ NodeEntry& n = AN("n_conv1",   "convnet.module.layer.conv");   S(n,"kernel_size",5); S(n,"out_channels",8); S(n,"stride",1); S(n,"padding",2); }
			AN("n_relu1",   "convnet.module.layer.relu");
			{ NodeEntry& n = AN("n_pool1",   "convnet.module.layer.pool");   S(n,"kernel_size",2); S(n,"stride",2); S(n,"padding",0); }
			{ NodeEntry& n = AN("n_fc1",     "convnet.module.layer.fc");     S(n,"neuron_count",64); }
			AN("n_relu2",   "convnet.module.layer.relu");
			{ NodeEntry& n = AN("n_softmax", "convnet.module.layer.softmax"); S(n,"class_count",10); }
			{ NodeEntry& n = AN("n_compile", "convnet.compile");              S(n,"mode",String("compile")); }
			{ NodeEntry& n = AN("n_sgd",     "convnet.module.trainer.sgd");   S(n,"learning_rate",0.01); S(n,"momentum",0.9); S(n,"batch_size",32); S(n,"l2_decay",0.001); }
			{ NodeEntry& n = AN("n_train",   "convnet.train");                S(n,"epochs",20); S(n,"learning_rate",0.01); }
			AE("e1","n_input","layer_stack","n_conv1","layer_stack");
			AE("e2","n_conv1","layer_stack","n_relu1","layer_stack");
			AE("e3","n_relu1","layer_stack","n_pool1","layer_stack");
			AE("e4","n_pool1","layer_stack","n_fc1","layer_stack");
			AE("e5","n_fc1","layer_stack","n_relu2","layer_stack");
			AE("e6","n_relu2","layer_stack","n_softmax","layer_stack");
			AE("e7","n_softmax","layer_stack","n_compile","layer_stack");
			AE("e8","n_compile","model","n_sgd","model");
			AE("e9","n_sgd","model","n_train","model");
		}
		else {
			error_out = "Unknown template index: " + IntStr(index);
			return String();
		}

		// Build Node::Graph from TplDef
		EnsurePalette();
		Node::Graph g;
		Node::GraphDoc& doc = g.GetDoc();

		// Track auto-layout position
		double x = 60, y_base = 60;
		double x_step = 200;
		for(const NodeEntry& ne : def.nodes) {
			Node::NodeDoc& nd = doc.nodes.Add();
			nd.id = ne.id;
			nd.label = ne.id;
			nd.node_type_id = ne.type;
			nd.pos = Pointf(x, y_base);
			nd.sz  = Sizef(220, 80 + ne.slots.GetCount() * 26);
			nd.fill_clr = Color(42, 46, 56);
			nd.line_clr = Color(88, 96, 116);
			x += x_step;

			// Find matching template to copy pins
			for(const TemplateDef& td : templates) {
				if(td.type_id == ne.type) {
					nd.pins <<= td.doc.pins;
					nd.slots <<= td.doc.slots;
					break;
				}
			}

			// Override slot values from TplDef
			for(int si = 0; si < ne.slots.GetCount(); si++) {
				const String& slot_id = ne.slots.GetKey(si);
				const Value&  slot_val = ne.slots[si];
				for(Node::WidgetSlotDoc& s : nd.slots) {
					if(s.id == slot_id) {
						s.properties.GetAdd("value") = slot_val;
						break;
					}
				}
			}
		}

		for(const EdgeEntry& ee : def.edges) {
			Node::EdgeDoc& ed = doc.edges.Add();
			ed.id          = ee.id;
			ed.source_node = ee.src_node;
			ed.source_pin  = ee.src_pin;
			ed.target_node = ee.tgt_node;
			ed.target_pin  = ee.tgt_pin;
			ed.directed    = true;
		}

		g.RebuildIndexPublic();

		// Write files
		String tpl_dir = AppendFileName(dest_dir, def.name);
		if(!DirectoryExists(tpl_dir) && !RealizeDirectory(tpl_dir)) {
			error_out = "Cannot create directory: " + tpl_dir;
			return String();
		}

		String graph_path   = AppendFileName(tpl_dir, "main.nngrf");
		String project_path = AppendFileName(tpl_dir, def.name + ".nnprj");
		String sln_path     = AppendFileName(tpl_dir, def.name + ".nnsln");

		if(!SaveFile(graph_path, Node::SaveEon(g))) {
			error_out = "Failed to save graph: " + graph_path;
			return String();
		}

		WorkbenchProject prj_new;
		prj_new.name = def.name;
		prj_new.graphs.Add(graph_path);
		prj_new.startup_graph = graph_path;
		if(!prj_new.Save(project_path)) {
			error_out = "Failed to save project: " + project_path;
			return String();
		}

		WorkbenchSolution sln_new;
		sln_new.name = def.name;
		sln_new.projects.Add(project_path);
		sln_new.active_project = project_path;
		if(!sln_new.Save(sln_path)) {
			error_out = "Failed to save solution: " + sln_path;
			return String();
		}

		// Update manifest.json — add or replace entry
		String manifest_path = AppendFileName(dest_dir, "manifest.json");
		Value manifest_val;
		if(FileExists(manifest_path))
			manifest_val = ParseJSON(LoadFile(manifest_path));

		ValueMap manifest_map;
		if(IsValueMap(manifest_val))
			manifest_map = manifest_val;
		else {
			manifest_map.GetAdd("version") = 1;
		}

		Value generated_val = manifest_map.GetAdd("generated");
		ValueArray gen_arr;
		if(IsValueArray(generated_val))
			gen_arr = generated_val;

		// Remove existing entry with same name
		ValueArray gen_new;
		for(int i = 0; i < gen_arr.GetCount(); i++) {
			Value entry = gen_arr[i];
			if(IsValueMap(entry) && entry["name"].ToString() != def.name)
				gen_new.Add(entry);
		}

		// Add new entry with domain metadata
		ValueMap entry;
		entry.GetAdd("name")           = def.name;
		entry.GetAdd("domain")         = GetDomainId();
		entry.GetAdd("source_template")= def.name;
		entry.GetAdd("solution")       = sln_path;
		entry.GetAdd("project")        = project_path;
		entry.GetAdd("graph")          = graph_path;
		ValueArray sols, prjs, grfs;
		sols.Add(sln_path);
		prjs.Add(project_path);
		grfs.Add(graph_path);
		entry.GetAdd("solutions")      = sols;
		entry.GetAdd("projects")       = prjs;
		entry.GetAdd("graphs")         = grfs;
		gen_new.Add(entry);

		manifest_map.GetAdd("generated") = gen_new;
		SaveFile(manifest_path, AsJSON(manifest_map, true));

		return sln_path;
	}
};

static DomainRegistry::Entry s_neural_domain_reg([]() -> INodeWorkbenchDomain* {
	return new NeuralDomain();
});

}

NeuralEditorWindow::NeuralEditorWindow() {
	domain.Attach(DomainRegistry::CreateById("neural"));
	if(!domain)
		domain.Attach(new NeuralDomain());
	RegisterDomain(*domain);
	SetScriptRuntime(script_runtime);
}

NeuralEditorWindow::~NeuralEditorWindow() {
	domain.Clear();
}

END_UPP_NAMESPACE
