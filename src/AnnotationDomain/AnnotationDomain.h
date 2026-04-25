#ifndef _AnnotationDomain_AnnotationDomain_h_
#define _AnnotationDomain_AnnotationDomain_h_

#include <NodeWorkbench/NodeWorkbench.h>
#include <AnnLayCore/AnnLay.h>
#include <AnnLayCore/AnnSln.h>

NAMESPACE_UPP

// ---------------------------------------------------------------------------
// AnnotationDomainWindow — standalone NodeWorkbench host for annotation domain
// ---------------------------------------------------------------------------

class AnnotationDomainWindow : public NodeWorkbenchWindow {
public:
	typedef AnnotationDomainWindow CLASSNAME;

	AnnotationDomainWindow();
	~AnnotationDomainWindow();

private:
	One<INodeWorkbenchDomain> domain;
};

END_UPP_NAMESPACE

#endif
