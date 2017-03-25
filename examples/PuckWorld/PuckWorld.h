#ifndef _PuckWorld_PuckWorld_h
#define _PuckWorld_PuckWorld_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <Docking/Docking.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS PuckWorldImg
#define IMAGEFILE <PuckWorld/PuckWorld.iml>
#include <Draw/iml_header.h>

class PuckWorld;

class PuckWorldAgent : public DQNAgent {
	
protected:
	friend class PuckWorldCtrl;
	
	Vector<double> smooth_reward_history;
	double ppx, ppy, pvx, pvy;
	double rad, BADRAD;
	double tx, ty, tx2, ty2;
	double smooth_reward;
	double reward;
	int action;
	int flott;
	int nflot;
	int t;
	
public:
	PuckWorldAgent();
	
	virtual void SampleNextState(int x, int y, int d, int action, int& next_state, double& reward, bool& reset_episode);
	virtual void Learn();
	
	void Reset();
	void GetState(Vector<double>& slist);
	
	PuckWorld* pworld;
};

struct PuckWorldCtrl : public Ctrl {
	SpinLock		lock;
	PuckWorldAgent*			agent;
	
public:
	typedef PuckWorldCtrl CLASSNAME;
	PuckWorldCtrl();
	
	virtual void Layout();
	virtual void MouseWheel(Point, int zdelta, dword);
	virtual void LeftDown(Point p, dword keyflags);
	virtual void Paint(Draw& w);
	
	void SetAgent(PuckWorldAgent& agent) {this->agent = &agent;}
	
	Callback WhenGridFocus, WhenGridUnfocus;
	
};

class PuckWorld : public DockWindow {
	
protected:
	friend class PuckWorldAgent;
	
	PuckWorldAgent agent;
	PuckWorldCtrl pworld;
	Label lbl_eps;
	SliderCtrl eps;
	Splitter btnsplit;
	Button gofast, gonorm, goslow, reset;
	ButtonOption toggle;
	
	ParentCtrl statusctrl;
	Label status;
	Button load_pretrained;
	
	TrainingGraph reward;
	
	Button reload_btn;
	ParentCtrl agent_ctrl;
	DocEdit agent_edit;
	
	String t;
	
	
public:
	typedef PuckWorld CLASSNAME;
	PuckWorld();
	~PuckWorld();
	
	virtual void DockInit();
	
	void Reset(bool init_reward, bool start);
	void Reload();
	void Refresher();
	void RefreshEpsilon();
	void ToggleIteration();
	void LoadPretrained();
	void RefreshStatus();
	
	void SetSpeed(int i);
	
	
};

#endif
