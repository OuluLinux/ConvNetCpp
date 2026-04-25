#ifndef _NeuralEditor_NeuralScriptRuntime_h_
#define _NeuralEditor_NeuralScriptRuntime_h_

#include <NodeWorkbench/NodeWorkbench.h>
#include <ByteVM/ByteVM.h>

NAMESPACE_UPP

// ---------------------------------------------------------------------------
// ByteVMScriptRuntime — IScriptRuntime backed by the in-process ByteVM
// ---------------------------------------------------------------------------
class ByteVMScriptRuntime : public IScriptRuntime {
	PyVM vm;
	bool initialized = false;

public:
	virtual String GetBackendId()   const override { return "bytevm"; }
	virtual String GetBackendDesc() const override { return "ByteVM Python 3 subset interpreter"; }

	virtual bool Init(String& error_out) override;
	virtual void Reset() override;
	virtual ScriptRunResult RunSource(const String& src,
	                                  const String& filename) override;
	virtual ScriptRunResult CallFunction(const String& fn_name,
	                                     const Value& args) override;

	// Expose the underlying VM for advanced use (binding providers etc.)
	PyVM& GetVM() { return vm; }
};

END_UPP_NAMESPACE

#endif
