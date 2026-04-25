#ifndef _ProtoVM_ProtoVM_h_
#define _ProtoVM_ProtoVM_h_

#include <NodeWorkbench/NodeWorkbench.h>
#include "ProtoVMCompiler.h"

NAMESPACE_UPP

// ---------------------------------------------------------------------------
// ProtoVMWindow — standalone NodeWorkbench host for the electric domain
// ---------------------------------------------------------------------------

class ProtoVMWindow : public NodeWorkbenchWindow {
public:
	typedef ProtoVMWindow CLASSNAME;

	ProtoVMWindow();
	~ProtoVMWindow();

private:
	One<INodeWorkbenchDomain> domain;
};

END_UPP_NAMESPACE

#endif
