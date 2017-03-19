#ifndef _ConvNetCtrl_HeatmapView_h_
#define _ConvNetCtrl_HeatmapView_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class HeatmapView : public Ctrl {
	Session* ses;
	Graph* graph;
	Vector<double> tmp;
	int mode;
	
	enum {MODE_SESSION, MODE_GRAPH};
	
public:
	typedef HeatmapView CLASSNAME;
	HeatmapView();
	
	virtual void Paint(Draw& d);
	void PaintSession(Draw& d);
	void PaintGraph(Draw& d);
	
	void SetSession(Session& ses) {this->ses = &ses; mode = MODE_SESSION;}
	void SetGraph(Graph& g) {this->graph = &g; mode = MODE_GRAPH;}
	
};

}

#endif
