#ifndef _NeuralEditor_NeuralEditor_h_
#define _NeuralEditor_NeuralEditor_h_

#include <NodeWorkbench/NodeWorkbench.h>

#include "NeuralCompiler.h"
#include "NeuralScriptRuntime.h"

NAMESPACE_UPP

class NeuralEditorWindow : public NodeWorkbenchWindow {
public:
	typedef NeuralEditorWindow CLASSNAME;

	NeuralEditorWindow();
	~NeuralEditorWindow();

private:
	One<INodeWorkbenchDomain> domain;
	ByteVMScriptRuntime       script_runtime;
};

END_UPP_NAMESPACE

#endif
