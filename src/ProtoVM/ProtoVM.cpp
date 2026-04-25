#include "ProtoVM.h"
#include <Node/Script/Script.h>

NAMESPACE_UPP

namespace {

// ---------------------------------------------------------------------------
// Pin / slot helpers
// ---------------------------------------------------------------------------

Color ElectricPinColor(const String& type_name) {
	if(type_name == "POWER")   return Color(255, 220,  80);
	if(type_name == "SIGNAL")  return Color(100, 220, 100);
	if(type_name == "CONTROL") return Color(200, 120, 255);
	if(type_name == "WIRE")    return Color(180, 180, 180);
	if(type_name == "CLOCK")   return Color( 80, 200, 255);
	return Color(160, 160, 160);
}

void AddPin(Node::NodeDoc& n, const String& id, Node::PinKind kind, const String& type_name) {
	Node::PinDoc& p = n.pins.Add();
	p.id        = id;
	p.label     = id;
	p.kind      = kind;
	p.type_name = type_name;
	p.color     = ElectricPinColor(type_name);
}

void AddSlot(Node::NodeDoc& n, const String& id, const String& type, const Value& value) {
	Node::WidgetSlotDoc& s = n.slots.Add();
	s.id   = id;
	s.type = type;
	s.properties.GetAdd("value") = value;
}

// ---------------------------------------------------------------------------
// TemplateDef
// ---------------------------------------------------------------------------

struct TemplateDef : Moveable<TemplateDef> {
	String        type_id;
	String        category;
	String        label;
	Node::NodeDoc doc;
};

// ---------------------------------------------------------------------------
// ProtoDomain — INodeWorkbenchDomain implementation
// ---------------------------------------------------------------------------

class ProtoDomain : public INodeWorkbenchDomain {
	Vector<TemplateDef> templates;
	bool                palette_ready = false;

	void EnsurePalette() {
		if(palette_ready) return;
		palette_ready = true;
		templates.Clear();

		auto Add = [&](const String& type_id, const String& category, const String& label,
		               Function<void(Node::NodeDoc&)> fn) {
			TemplateDef& t = templates.Add();
			t.type_id  = type_id;
			t.category = category;
			t.label    = label;
			t.doc.node_type_id = type_id;
			t.doc.category     = category;
			t.doc.label        = label;
			t.doc.fill_clr     = Color(28, 32, 44);
			t.doc.line_clr     = Color(90, 100, 130);
			t.doc.sz           = Sizef(220, 80);
			fn(t.doc);
		};

		// Sources
		Add("electric.source.voltage", "electric.sources", "Voltage Source", [](Node::NodeDoc& n) {
			n.fill_clr = Color(60, 45, 20);
			n.line_clr = Color(200, 160, 60);
			AddPin(n, "pos", Node::PinKind::Output, "POWER");
			AddPin(n, "neg", Node::PinKind::Output, "WIRE");
			AddSlot(n, "voltage_mv", "EditIntSpin", 5000);
			AddSlot(n, "label",      "EditString",  String("V1"));
		});
		Add("electric.source.current", "electric.sources", "Current Source", [](Node::NodeDoc& n) {
			n.fill_clr = Color(40, 50, 25);
			n.line_clr = Color(160, 200, 80);
			AddPin(n, "pos", Node::PinKind::Output, "POWER");
			AddPin(n, "neg", Node::PinKind::Output, "WIRE");
			AddSlot(n, "current_ua", "EditIntSpin", 1000);
			AddSlot(n, "label",      "EditString",  String("I1"));
		});

		// Ground
		Add("electric.ground", "electric.passive", "Ground", [](Node::NodeDoc& n) {
			n.sz = Sizef(160, 60);
			AddPin(n, "gnd", Node::PinKind::Input, "WIRE");
		});

		// Passive components
		Add("electric.resistor", "electric.passive", "Resistor", [](Node::NodeDoc& n) {
			AddPin(n, "a", Node::PinKind::Input,  "WIRE");
			AddPin(n, "b", Node::PinKind::Output, "WIRE");
			AddSlot(n, "resistance_mohm", "EditIntSpin", 1000); // 1 Ω default
			AddSlot(n, "label",           "EditString",  String("R1"));
		});
		Add("electric.capacitor", "electric.passive", "Capacitor", [](Node::NodeDoc& n) {
			AddPin(n, "a", Node::PinKind::Input,  "WIRE");
			AddPin(n, "b", Node::PinKind::Output, "WIRE");
			AddSlot(n, "capacitance_pf", "EditIntSpin", 100000); // 100 nF
			AddSlot(n, "label",          "EditString",  String("C1"));
		});
		Add("electric.inductor", "electric.passive", "Inductor", [](Node::NodeDoc& n) {
			AddPin(n, "a", Node::PinKind::Input,  "WIRE");
			AddPin(n, "b", Node::PinKind::Output, "WIRE");
			AddSlot(n, "inductance_nh", "EditIntSpin", 1000); // 1 µH
			AddSlot(n, "label",         "EditString",  String("L1"));
		});

		// Semiconductors
		Add("electric.diode", "electric.semiconductors", "Diode", [](Node::NodeDoc& n) {
			n.fill_clr = Color(35, 30, 50);
			n.line_clr = Color(130, 100, 200);
			AddPin(n, "anode",   Node::PinKind::Input,  "WIRE");
			AddPin(n, "cathode", Node::PinKind::Output, "WIRE");
			AddSlot(n, "vf_mv",  "EditIntSpin", 700);
			AddSlot(n, "label",  "EditString",  String("D1"));
		});
		Add("electric.transistor.npn", "electric.semiconductors", "NPN Transistor", [](Node::NodeDoc& n) {
			n.sz = Sizef(220, 100);
			n.fill_clr = Color(35, 30, 50);
			n.line_clr = Color(130, 100, 200);
			AddPin(n, "base",      Node::PinKind::Input,  "CONTROL");
			AddPin(n, "collector", Node::PinKind::Input,  "POWER");
			AddPin(n, "emitter",   Node::PinKind::Output, "WIRE");
			AddSlot(n, "hfe",   "EditIntSpin", 100);
			AddSlot(n, "label", "EditString",  String("Q1"));
		});

		// Logic gates
		Add("electric.logic.inv", "electric.logic", "Inverter (NOT)", [](Node::NodeDoc& n) {
			n.sz = Sizef(180, 70);
			AddPin(n, "in",  Node::PinKind::Input,  "SIGNAL");
			AddPin(n, "out", Node::PinKind::Output, "SIGNAL");
		});
		Add("electric.logic.and2", "electric.logic", "AND Gate", [](Node::NodeDoc& n) {
			n.sz = Sizef(200, 80);
			AddPin(n, "a",   Node::PinKind::Input,  "SIGNAL");
			AddPin(n, "b",   Node::PinKind::Input,  "SIGNAL");
			AddPin(n, "out", Node::PinKind::Output, "SIGNAL");
		});
		Add("electric.logic.or2", "electric.logic", "OR Gate", [](Node::NodeDoc& n) {
			n.sz = Sizef(200, 80);
			AddPin(n, "a",   Node::PinKind::Input,  "SIGNAL");
			AddPin(n, "b",   Node::PinKind::Input,  "SIGNAL");
			AddPin(n, "out", Node::PinKind::Output, "SIGNAL");
		});
		Add("electric.logic.xor2", "electric.logic", "XOR Gate", [](Node::NodeDoc& n) {
			n.sz = Sizef(200, 80);
			AddPin(n, "a",   Node::PinKind::Input,  "SIGNAL");
			AddPin(n, "b",   Node::PinKind::Input,  "SIGNAL");
			AddPin(n, "out", Node::PinKind::Output, "SIGNAL");
		});

		// Clock
		Add("electric.clock", "electric.timing", "Clock Generator", [](Node::NodeDoc& n) {
			n.fill_clr = Color(20, 40, 55);
			n.line_clr = Color(80, 180, 230);
			n.sz = Sizef(220, 80);
			AddPin(n, "clk", Node::PinKind::Output, "CLOCK");
			AddSlot(n, "frequency_hz", "EditIntSpin", 1000000);
			AddSlot(n, "duty_pct",     "EditIntSpin", 50);
			AddSlot(n, "label",        "EditString",  String("CLK1"));
		});

		// Probes
		Add("electric.probe.voltage", "electric.probes", "Voltage Probe", [](Node::NodeDoc& n) {
			n.fill_clr = Color(20, 50, 35);
			n.line_clr = Color(60, 180, 100);
			AddPin(n, "net", Node::PinKind::Input, "WIRE");
			AddPin(n, "gnd", Node::PinKind::Input, "WIRE");
			AddSlot(n, "label", "EditString", String("VP1"));
		});
		Add("electric.probe.logic", "electric.probes", "Logic Probe", [](Node::NodeDoc& n) {
			n.fill_clr = Color(20, 50, 35);
			n.line_clr = Color(60, 180, 100);
			AddPin(n, "sig", Node::PinKind::Input, "SIGNAL");
			AddSlot(n, "label", "EditString", String("LP1"));
		});
	}

public:
	virtual String GetDomainId()   const override { return "electric"; }
	virtual String GetDomainName() const override { return "Electric"; }
	virtual String GetDomainDesc() const override { return "ProtoVM electric circuit schematic domain"; }

	virtual String GetGraphFileFilter() const override {
		return "Electric Graph (*.elgrf)\t*.elgrf";
	}
	virtual String GetProjectFileFilter() const override {
		return "Electric Project (*.elprj)\t*.elprj";
	}
	virtual String GetSolutionFileFilter() const override {
		return "Electric Solution (*.elsln)\t*.elsln";
	}
	virtual String GetExtraExtensions() const override {
		return ".elgrf|.elprj|.elsln";
	}

	virtual void OnDomainInit(NodeWorkbenchWindow& host) override {
		EnsurePalette();
		for(const TemplateDef& t : templates) {
			host.GetViewport().RegisterNodeType(
				t.type_id, t.label,
				[doc = t.doc]() mutable {
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
			p.label    = t.label;
			p.type_id  = t.type_id;
		}
	}

	virtual void ValidateGraph(NodeWorkbenchWindow& host,
	                           Vector<WorkbenchDiagnostic>& diag_out) override {
		diag_out.Clear();
		ProtoVMCompiler::Result r = ProtoVMCompiler::Validate(host.GetGraph());
		for(const String& msg : r.errors) {
			WorkbenchDiagnostic& d = diag_out.Add();
			d.severity = DiagSeverity::Error;
			d.message  = msg;
			d.source   = "electric.validate";
		}
		for(const String& msg : r.warnings) {
			WorkbenchDiagnostic& d = diag_out.Add();
			d.severity = DiagSeverity::Warning;
			d.message  = msg;
			d.source   = "electric.validate";
		}
	}

	virtual bool CompileGraph(NodeWorkbenchWindow& host, String& log_out) override {
		ProtoVMCompiler::Result r = ProtoVMCompiler::Validate(host.GetGraph());
		log_out = r.ToText();
		return r.ok;
	}

	virtual bool RunGraph(NodeWorkbenchWindow& host, String& log_out) override {
		// Phase 4 Task 03 stub — real SPICE/nodal simulation in Phase 5+
		ProtoVMCompiler::Result r = ProtoVMCompiler::Validate(host.GetGraph());
		if(!r.ok) {
			log_out = "Simulation aborted — graph has errors:\n" + r.ToText();
			return false;
		}
		log_out  = "Stub simulation: graph topology OK.\n";
		log_out << "Nodes: " << host.GetGraph().GetDoc().nodes.GetCount()
		        << "  Edges: " << host.GetGraph().GetDoc().edges.GetCount() << "\n";
		log_out << "(Full SPICE simulation not yet implemented.)\n";
		return true;
	}

	virtual void GetTemplates(Vector<TemplateDesc>& out) override {
		auto Desc = [&](const char* name, const char* cat, const char* desc) {
			TemplateDesc& d = out.Add();
			d.name = name; d.category = cat; d.description = desc;
		};
		Desc("RC Low-Pass Filter",    "electric.examples", "R-C low-pass filter with voltage source and voltage probe");
		Desc("LED Driver",             "electric.examples", "NPN transistor LED driver with base resistor");
		Desc("Logic Inverter Chain",   "electric.examples", "Three-stage NOT gate chain with clock source and logic probe");
	}

	virtual String GenerateTemplate(int index, const String& dest_dir, String& error_out) override {
		EnsurePalette();

		struct NodeEntry : Moveable<NodeEntry> {
			String id, type;
			Pointf pos;
			VectorMap<String, Value> slots;
		};
		struct EdgeEntry : Moveable<EdgeEntry> {
			String id, src_node, src_pin, tgt_node, tgt_pin;
		};
		struct TplDef {
			String name;
			Vector<NodeEntry> nodes;
			Vector<EdgeEntry> edges;
		};

		TplDef def;

		auto AN = [&](const char* id, const char* type, Pointf pos) -> NodeEntry& {
			NodeEntry& n = def.nodes.Add();
			n.id = id; n.type = type; n.pos = pos;
			return n;
		};
		auto S = [](NodeEntry& n, const char* slot_id, const Value& val) -> NodeEntry& {
			n.slots.GetAdd(slot_id) = val;
			return n;
		};
		auto AE = [&](const char* id,
		              const char* sn, const char* sp,
		              const char* tn, const char* tp) {
			EdgeEntry& e = def.edges.Add();
			e.id = id; e.src_node = sn; e.src_pin = sp;
			e.tgt_node = tn; e.tgt_pin = tp;
		};
		(void)S;

		if(index == 0) {
			def.name = "RC Low-Pass Filter";
			{ NodeEntry& n = AN("vs1",  "electric.source.voltage", Pointf(60,  200)); S(n,"voltage_mv",5000); S(n,"label",String("V1")); }
			{ NodeEntry& n = AN("r1",   "electric.resistor",       Pointf(280, 200)); S(n,"resistance_mohm",10000); S(n,"label",String("R1")); }
			{ NodeEntry& n = AN("c1",   "electric.capacitor",      Pointf(480, 200)); S(n,"capacitance_pf",100000); S(n,"label",String("C1")); }
			{ NodeEntry& n = AN("gnd1", "electric.ground",         Pointf(300, 360)); (void)n; }
			{ NodeEntry& n = AN("vp1",  "electric.probe.voltage",  Pointf(480, 360)); S(n,"label",String("Vout")); }
			AE("e1","vs1","pos",  "r1","a");
			AE("e2","r1","b",     "c1","a");
			AE("e3","c1","b",     "gnd1","gnd");
			AE("e4","vs1","neg",  "gnd1","gnd");
			AE("e5","c1","a",     "vp1","net");
			AE("e6","gnd1","gnd", "vp1","gnd");
		}
		else if(index == 1) {
			def.name = "LED Driver";
			{ NodeEntry& n = AN("vcc",  "electric.source.voltage",     Pointf(60,  180)); S(n,"voltage_mv",5000); S(n,"label",String("VCC")); }
			{ NodeEntry& n = AN("rb1",  "electric.resistor",           Pointf(250, 300)); S(n,"resistance_mohm",47000); S(n,"label",String("Rb")); }
			{ NodeEntry& n = AN("q1",   "electric.transistor.npn",     Pointf(460, 200)); S(n,"hfe",100); S(n,"label",String("Q1")); }
			{ NodeEntry& n = AN("led1", "electric.diode",              Pointf(460,  60)); S(n,"vf_mv",2000); S(n,"label",String("LED1")); }
			{ NodeEntry& n = AN("rl1",  "electric.resistor",           Pointf(320,  60)); S(n,"resistance_mohm",220000); S(n,"label",String("Rl")); }
			{ NodeEntry& n = AN("gnd1", "electric.ground",             Pointf(460, 380)); (void)n; }
			{ NodeEntry& n = AN("in1",  "electric.probe.logic",        Pointf(60,  300)); S(n,"label",String("IN")); }
			AE("e1","in1","sig",      "rb1","a");
			AE("e2","rb1","b",        "q1","base");
			AE("e3","vcc","pos",      "rl1","a");
			AE("e4","rl1","b",        "led1","anode");
			AE("e5","led1","cathode", "q1","collector");
			AE("e6","q1","emitter",   "gnd1","gnd");
			AE("e7","vcc","neg",      "gnd1","gnd");
		}
		else if(index == 2) {
			def.name = "Logic Inverter Chain";
			{ NodeEntry& n = AN("clk1", "electric.clock",      Pointf( 60, 200)); S(n,"frequency_hz",1000); S(n,"label",String("CLK")); }
			{ NodeEntry& n = AN("inv1", "electric.logic.inv",  Pointf(250, 200)); (void)n; }
			{ NodeEntry& n = AN("inv2", "electric.logic.inv",  Pointf(420, 200)); (void)n; }
			{ NodeEntry& n = AN("inv3", "electric.logic.inv",  Pointf(590, 200)); (void)n; }
			{ NodeEntry& n = AN("lp1",  "electric.probe.logic",Pointf(760, 200)); S(n,"label",String("OUT")); }
			AE("e1","clk1","clk","inv1","in");
			AE("e2","inv1","out","inv2","in");
			AE("e3","inv2","out","inv3","in");
			AE("e4","inv3","out","lp1","sig");
		}
		else {
			error_out = "Unknown template index: " + IntStr(index);
			return String();
		}

		// Build Node::Graph from TplDef
		Node::Graph g;
		Node::GraphDoc& doc = g.GetDoc();

		for(const NodeEntry& ne : def.nodes) {
			Node::NodeDoc& nd = doc.nodes.Add();
			nd.id = ne.id;
			nd.label = ne.id;
			nd.node_type_id = ne.type;
			nd.pos = ne.pos;

			// Copy pins/slots from palette template
			for(const TemplateDef& td : templates) {
				if(td.type_id == ne.type) {
					nd <<= td.doc;
					nd.id  = ne.id;
					nd.pos = ne.pos;
					break;
				}
			}

			// Apply slot overrides
			for(int si = 0; si < ne.slots.GetCount(); si++) {
				const String& slot_id  = ne.slots.GetKey(si);
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

		String graph_path   = AppendFileName(tpl_dir, "main.elgrf");
		String project_path = AppendFileName(tpl_dir, def.name + ".elprj");
		String sln_path     = AppendFileName(tpl_dir, def.name + ".elsln");

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

		// Update manifest.json
		String manifest_path = AppendFileName(dest_dir, "manifest.json");
		Value  manifest_val;
		if(FileExists(manifest_path))
			manifest_val = ParseJSON(LoadFile(manifest_path));

		ValueMap manifest_map;
		if(IsValueMap(manifest_val))
			manifest_map = manifest_val;
		else
			manifest_map.GetAdd("version") = 1;

		Value        generated_val = manifest_map.GetAdd("generated");
		ValueArray   gen_arr;
		if(IsValueArray(generated_val)) gen_arr = generated_val;

		ValueArray gen_new;
		for(int i = 0; i < gen_arr.GetCount(); i++) {
			Value entry = gen_arr[i];
			if(IsValueMap(entry) && entry["name"].ToString() != def.name)
				gen_new.Add(entry);
		}

		ValueMap entry;
		entry.GetAdd("name")           = def.name;
		entry.GetAdd("domain")         = GetDomainId();
		entry.GetAdd("solution")       = sln_path;
		entry.GetAdd("project")        = project_path;
		entry.GetAdd("graph")          = graph_path;
		gen_new.Add(entry);

		manifest_map.GetAdd("generated") = gen_new;
		SaveFile(manifest_path, AsJSON(manifest_map, true));

		return sln_path;
	}
};

static DomainRegistry::Entry s_electric_domain_reg([] { return new ProtoDomain(); });

} // namespace

// ---------------------------------------------------------------------------
// ProtoVMWindow
// ---------------------------------------------------------------------------

ProtoVMWindow::ProtoVMWindow() {
	domain.Attach(DomainRegistry::CreateById("electric"));
	if(!domain)
		domain.Attach(new ProtoDomain());
	RegisterDomain(*domain);
	Title("ProtoVM — Electric Circuit Workbench");
}

ProtoVMWindow::~ProtoVMWindow() {
	domain.Clear();
}

END_UPP_NAMESPACE
