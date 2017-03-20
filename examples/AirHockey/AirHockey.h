#ifndef _AirHockey_AirHockey_h
#define _AirHockey_AirHockey_h

#include <CtrlLib/CtrlLib.h>
#include <plugin/box2d/Box2D.h>
#include <Painter/Painter.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS AirHockeyImg
#define IMAGEFILE <AirHockey/AirHockey.iml>
#include <Draw/iml_header.h>


#include "World.h"
#include "Player.h"
#include "Object.h"
#include "Contact.h"
#include "AirHockeyGame.h"

namespace GameCtrl {
	
class AirHockeyDQN : public DockWindow {
	
	
protected:
	friend class Player;
	
	HeatmapView network_view;
	GameCtrl::Table2 world;
	Label lbl_eps;
	SliderCtrl eps;
	Splitter btnsplit;
	Button goveryfast, gofast, gonorm, goslow, reset;
	ParentCtrl statusctrl;
	Label status;
	Button load_pretrained;
	Array<TrainingGraph> reward;
	Button reload_btn;
	ParentCtrl agent_ctrl;
	DocEdit agent_edit;
	String t;
	int simspeed;
	
	SpinLock ticking_lock;
	
	
	
public:
	typedef AirHockeyDQN CLASSNAME;
	AirHockeyDQN();
	~AirHockeyDQN();
	
	virtual void DockInit();
	
	void Reset(bool init_reward, bool start);
	void Reload();
	void Refresher();
	void RefreshEpsilon();
	void LoadPretrained();
	void RefreshStatus();
	void PostRefreshStatus() {PostCallback(THISBACK(RefreshStatus));}
	void SetSpeed(int i);
	
};

}

#endif
