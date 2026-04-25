#include "NeuralScriptRuntime.h"

NAMESPACE_UPP

bool ByteVMScriptRuntime::Init(String& error_out) {
	if(initialized) return true;
	try {
		vm.InitBuiltins();
		// Wire print output to WhenOutput event
		vm.WhenPrint = [this](const String& line) {
			if(WhenOutput)
				WhenOutput(line);
		};
		initialized = true;
		return true;
	} catch(Exc& e) {
		error_out = String("ByteVM init failed: ") + e;
		return false;
	} catch(std::exception& e) {
		error_out = String("ByteVM init failed: ") + e.what();
		return false;
	}
}

void ByteVMScriptRuntime::Reset() {
	vm.Reset();
	vm.Clear();
	initialized = false;
}

ScriptRunResult ByteVMScriptRuntime::RunSource(const String& src,
                                               const String& filename) {
	ScriptRunResult res;
	String init_err;
	if(!initialized && !Init(init_err)) {
		res.error = init_err;
		return res;
	}

	String collected_output;
	auto prev_out = vm.WhenPrint;
	vm.WhenPrint = [&](const String& line) {
		collected_output << line << "\n";
		if(WhenOutput) WhenOutput(line);
	};

	try {
		bool ok = vm.LoadModule("__main__", src, filename);
		res.ok     = ok;
		res.output = collected_output;
		if(ok) {
			PyValue last = vm.GetLastResult();
			if(!last.IsNone())
				res.result = last.ToString();
		} else {
			res.error = "Compile or runtime error in: " + filename;
		}
	} catch(Exc& e) {
		res.ok    = false;
		res.error = String(e);
		res.output = collected_output;
	} catch(std::exception& e) {
		res.ok    = false;
		res.error = e.what();
		res.output = collected_output;
	}

	vm.WhenPrint = prev_out;
	return res;
}

ScriptRunResult ByteVMScriptRuntime::CallFunction(const String& fn_name,
                                                  const Value& args) {
	ScriptRunResult res;
	String init_err;
	if(!initialized && !Init(init_err)) {
		res.error = init_err;
		return res;
	}

	String collected_output;
	auto prev_out = vm.WhenPrint;
	vm.WhenPrint = [&](const String& line) {
		collected_output << line << "\n";
		if(WhenOutput) WhenOutput(line);
	};

	try {
		// Resolve callable from globals
		PyValue globals = vm.GetGlobals();
		PyValue fn_key  = PyValue(fn_name);
		PyValue fn_val  = globals.GetItem(fn_key);

		// Convert Value args to PyValue list
		Vector<PyValue> py_args;
		if(IsValueArray(args)) {
			for(int i = 0; i < args.GetCount(); i++) {
				Value v = args[i];
				if(IsNumber(v))
					py_args.Add(PyValue((double)v));
				else
					py_args.Add(PyValue(v.ToString()));
			}
		}

		PyValue result = vm.Call(fn_val, py_args);
		res.ok     = true;
		res.output = collected_output;
		res.result = result.ToString();
	} catch(Exc& e) {
		res.ok    = false;
		res.error = String(e);
		res.output = collected_output;
	} catch(std::exception& e) {
		res.ok    = false;
		res.error = e.what();
		res.output = collected_output;
	}

	vm.WhenPrint = prev_out;
	return res;
}

END_UPP_NAMESPACE
