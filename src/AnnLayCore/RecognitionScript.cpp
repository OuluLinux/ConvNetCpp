#include "RecognitionScript.h"

NAMESPACE_UPP

namespace {

struct RecognitionContext {
	const Vector<SlotResult>* results = nullptr;
	VectorMap<String, String>* out_meta = nullptr;
	String* last_output = nullptr;
};

const SlotResult* FindSlot(const RecognitionContext* ctx, const String& slot_id) {
	if(!ctx || !ctx->results)
		return nullptr;
	for(int i = 0; i < ctx->results->GetCount(); i++) {
		const SlotResult& r = (*ctx->results)[i];
		if(r.slot_id == slot_id)
			return &r;
	}
	return nullptr;
}

PyValue Fn_get_label(const Vector<PyValue>& args, void* ud) {
	const RecognitionContext* ctx = (const RecognitionContext*)ud;
	if(args.GetCount() < 1)
		return PyValue(String());
	String slot_id = args[0].ToString();
	const SlotResult* r = FindSlot(ctx, slot_id);
	if(!r)
		return PyValue(String());
	return PyValue(r->top_class);
}

PyValue Fn_get_conf(const Vector<PyValue>& args, void* ud) {
	const RecognitionContext* ctx = (const RecognitionContext*)ud;
	if(args.GetCount() < 1)
		return PyValue(0.0);
	String slot_id = args[0].ToString();
	const SlotResult* r = FindSlot(ctx, slot_id);
	if(!r)
		return PyValue(0.0);
	return PyValue(r->confidence);
}

PyValue Fn_get_text(const Vector<PyValue>& args, void* ud) {
	const RecognitionContext* ctx = (const RecognitionContext*)ud;
	if(args.GetCount() < 1)
		return PyValue(String());
	String slot_id = args[0].ToString();
	const SlotResult* r = FindSlot(ctx, slot_id);
	if(!r)
		return PyValue(String());
	return PyValue(r->raw_text);
}

PyValue Fn_get_bool(const Vector<PyValue>& args, void* ud) {
	const RecognitionContext* ctx = (const RecognitionContext*)ud;
	if(args.GetCount() < 1)
		return PyValue(false);
	String slot_id = args[0].ToString();
	const SlotResult* r = FindSlot(ctx, slot_id);
	if(!r)
		return PyValue(false);

	String cls = ToLower(TrimBoth(r->top_class));
	if(cls == "true" || cls == "1" || cls == "yes" || cls == "on")
		return PyValue(true);
	if(cls == "false" || cls == "0" || cls == "no" || cls == "off")
		return PyValue(false);
	return PyValue(r->class_index == 1);
}

PyValue Fn_get_class_index(const Vector<PyValue>& args, void* ud) {
	const RecognitionContext* ctx = (const RecognitionContext*)ud;
	if(args.GetCount() < 1)
		return PyValue(-1);
	String slot_id = args[0].ToString();
	const SlotResult* r = FindSlot(ctx, slot_id);
	if(!r)
		return PyValue(-1);
	return PyValue((int)r->class_index);
}

PyValue Fn_get_gate_status(const Vector<PyValue>& args, void* ud) {
	const RecognitionContext* ctx = (const RecognitionContext*)ud;
	if(args.GetCount() < 1)
		return PyValue(String());
	String slot_id = args[0].ToString();
	const SlotResult* r = FindSlot(ctx, slot_id);
	if(!r)
		return PyValue(String());
	return PyValue(r->gate_status);
}

PyValue Fn_get_offset_x(const Vector<PyValue>& args, void* ud) {
	const RecognitionContext* ctx = (const RecognitionContext*)ud;
	if(args.GetCount() < 1)
		return PyValue(0.0);
	String slot_id = args[0].ToString();
	const SlotResult* r = FindSlot(ctx, slot_id);
	if(!r)
		return PyValue(0.0);
	return PyValue(r->offset_dx);
}

PyValue Fn_get_offset_y(const Vector<PyValue>& args, void* ud) {
	const RecognitionContext* ctx = (const RecognitionContext*)ud;
	if(args.GetCount() < 1)
		return PyValue(0.0);
	String slot_id = args[0].ToString();
	const SlotResult* r = FindSlot(ctx, slot_id);
	if(!r)
		return PyValue(0.0);
	return PyValue(r->offset_dy);
}

PyValue Fn_set_meta(const Vector<PyValue>& args, void* ud) {
	RecognitionContext* ctx = (RecognitionContext*)ud;
	if(!ctx || !ctx->out_meta || args.GetCount() < 2)
		return PyValue::None();
	String key = args[0].ToString();
	if(key.IsEmpty())
		return PyValue::None();
	String val = args[1].ToString();
	ctx->out_meta->GetAdd(key) = val;
	return PyValue::None();
}

PyValue Fn_print(const Vector<PyValue>& args, void* ud) {
	RecognitionContext* ctx = (RecognitionContext*)ud;
	if(!ctx || !ctx->last_output)
		return PyValue::None();
	for(int i = 0; i < args.GetCount(); i++) {
		if(i > 0) (*ctx->last_output) << " ";
		(*ctx->last_output) << args[i].ToString();
	}
	(*ctx->last_output) << "\n";
	return PyValue::None();
}

} // namespace

void RecognitionScript::RegisterModule(PyVM& vm,
                                       const Vector<SlotResult>& results,
                                       VectorMap<String, String>& out_meta) {
	static RecognitionContext ctx;
	ctx.results = &results;
	ctx.out_meta = &out_meta;
	ctx.last_output = &last_output_;

	PyValue m = PyValue::Dict();
	m.SetItem(PyValue("get_label"), PyValue::Function("get_label", Fn_get_label, &ctx));
	m.SetItem(PyValue("get_conf"),  PyValue::Function("get_conf",  Fn_get_conf,  &ctx));
	m.SetItem(PyValue("get_text"),  PyValue::Function("get_text",  Fn_get_text,  &ctx));
	m.SetItem(PyValue("get_bool"),  PyValue::Function("get_bool",  Fn_get_bool,  &ctx));
	m.SetItem(PyValue("get_class_index"), PyValue::Function("get_class_index", Fn_get_class_index, &ctx));
	m.SetItem(PyValue("get_gate_status"), PyValue::Function("get_gate_status", Fn_get_gate_status, &ctx));
	m.SetItem(PyValue("get_offset_x"),    PyValue::Function("get_offset_x",    Fn_get_offset_x,    &ctx));
	m.SetItem(PyValue("get_offset_y"),    PyValue::Function("get_offset_y",    Fn_get_offset_y,    &ctx));
	m.SetItem(PyValue("set_meta"),  PyValue::Function("set_meta",  Fn_set_meta,  &ctx));
	vm.GetGlobalsRW().SetItem(PyValue("recognition"), m);

	vm.GetGlobalsRW().SetItem(PyValue("print"), PyValue::Function("print", Fn_print, &ctx));
}

bool RecognitionScript::Load(const String& script_path) {
	loaded_ = false;
	last_error_.Clear();

	String src = LoadFile(script_path);
	if(src.IsEmpty()) {
		last_error_ = "Cannot read script: " + script_path;
		return false;
	}

	Tokenizer tokenizer;
	tokenizer.SkipPythonComments(true);
	if(!tokenizer.Process(src, script_path)) {
		last_error_ = "Script tokenize error: " + script_path;
		return false;
	}
	tokenizer.CombineTokens();

	// ByteVM parser expects Python-style implicit line continuation to be
	// flattened inside (), [] and {} expressions.
	Vector<Token> tokens;
	const Vector<Token>& src_tokens = tokenizer.GetTokens();
	tokens.Reserve(src_tokens.GetCount());
	for(int i = 0; i < src_tokens.GetCount(); i++) {
		const Token& tk = src_tokens[i];
		if(tk.IsType(TK_NEWLINE) && tk.bracket_level > 0)
			continue;
		tokens.Add(tk);
	}

	Vector<PyIR> ir;
	try {
		PyCompiler compiler(tokens, script_path);
		compiler.Compile(ir);
	}
	catch(Exc e) {
		last_error_ = "Script compile error: " + e;
		return false;
	}
	catch(...) {
		last_error_ = "Script compile error: unknown exception";
		return false;
	}

	vm_.Clear();
	vm_.InitBuiltins();

	Vector<SlotResult> empty_results;
	VectorMap<String, String> empty_meta;
	RegisterModule(vm_, empty_results, empty_meta);

	try {
		vm_.SetIR(ir);
		vm_.Run();
	}
	catch(Exc e) {
		last_error_ = "Script runtime error: " + e;
		return false;
	}
	catch(...) {
		last_error_ = "Script runtime error: unknown exception";
		return false;
	}

	loaded_ = true;
	return true;
}

VectorMap<String, String> RecognitionScript::Run(const Vector<SlotResult>& raw_results) {
	VectorMap<String, String> out_meta;
	if(!loaded_)
		return out_meta;

	last_error_.Clear();
	last_output_.Clear();
	RegisterModule(vm_, raw_results, out_meta);

	PyValue fn = vm_.GetGlobals().GetItem(PyValue("process_frame"));
	if(fn.IsNone()) {
		last_error_ = "Script must define process_frame()";
		out_meta.Clear();
		return out_meta;
	}
	if(!fn.IsFunction() && !fn.IsBoundMethod()) {
		last_error_ = "process_frame is not callable";
		out_meta.Clear();
		return out_meta;
	}

	try {
		vm_.Call(fn, Vector<PyValue>());
	}
	catch(Exc e) {
		last_error_ = "Script runtime error: " + e;
		out_meta.Clear();
	}
	catch(...) {
		last_error_ = "Script runtime error: unknown exception";
		out_meta.Clear();
	}
	return out_meta;
}

END_UPP_NAMESPACE
