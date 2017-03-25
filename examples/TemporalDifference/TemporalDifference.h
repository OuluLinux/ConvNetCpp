#ifndef _TemporalDifference_TemporalDifference_h
#define _TemporalDifference_TemporalDifference_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <Docking/Docking.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS TemporalDifferenceImg
#define IMAGEFILE <TemporalDifference/TemporalDifference.iml>
#include <Draw/iml_header.h>

class TemporalDifference : public DockWindow {
	GridWorldCtrl gworld;
	Label lbl_reward, lbl_eps;
	SliderCtrl reward, eps;
	Splitter btnsplit;
	Button gofast, gonorm, goslow, reset;
	ButtonOption toggle;
	
	Button reload_btn;
	ParentCtrl agent_ctrl;
	DocEdit agent_edit;
	
	TDAgent agent;
	String t;
	
public:
	typedef TemporalDifference CLASSNAME;
	TemporalDifference();
	~TemporalDifference();
	
	virtual void DockInit();
	
	void Reload();
	void Refresher();
	void Reset(bool init_reward, bool start);
	void ToggleIteration();
	void GridFocus();
	void GridUnfocus();
	void RefreshReward();
	void RefreshEpsilon();
	void SetSpeed(int i);
	
};

#endif
