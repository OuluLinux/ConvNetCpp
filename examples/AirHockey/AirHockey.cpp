#include "AirHockey.h"
#include "pretrained.brc"
#include <plugin/bz2/bz2.h>

#define IMAGECLASS AirHockeyImg
#define IMAGEFILE <AirHockey/AirHockey.iml>
#include <Draw/iml_source.h>

namespace GameCtrl {

AirHockeyDQN::AirHockeyDQN()
{
	/*Icon(AirHockeyImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("UNFINISHED (!!!) AirHockey");
	
	int h = 512;
	int w = h * 3 / 5;
	
	table.Init();
	
	Add(table.SizePos());
	SetRect(0, 0, w, h);*/

	Title("AirHockey: Deep Q Learning");
	Icon(AirHockeyImg::icon());
	Sizeable().MaximizeBox().MinimizeBox();
	
	world.agents[0].GameCtrl::Player::world = this;
	
	world.Init();
	
	simspeed = 2;
	
	t =		"{\n"
			"\t\"update\":\"qlearn\",\n"
			"\t\"gamma\":0.9,\n"
			"\t\"epsilon\":0.2,\n"
			"\t\"alpha\":0.005,\n"
			"\t\"experience_add_every\":5,\n"
			"\t\"experience_size\":10000,\n"
			"\t\"learning_steps_per_iteration\":5,\n"
			"\t\"tderror_clamp\":1.0,\n"
			"\t\"num_hidden_units\":100,\n"
			"}\n";
	
	reward.SetCount(2);
	
	agent_edit.SetData(t);
	agent_ctrl.Add(agent_edit.HSizePos().VSizePos(0,30));
	agent_ctrl.Add(reload_btn.HSizePos().BottomPos(0,30));
	reload_btn.SetLabel("Reload Agent");
	reload_btn <<= THISBACK(Reload);
	
	statusctrl.Add(status.HSizePos().VSizePos(0,30));
	statusctrl.Add(load_pretrained.HSizePos().BottomPos(0,30));
	load_pretrained.SetLabel("Load a Pretrained Agent");
	load_pretrained <<= THISBACK(LoadPretrained);
	
	Add(btnsplit.HSizePos().TopPos(0,30));
	Add(world.HSizePos().VSizePos(30,30));
	Add(lbl_eps.LeftPos(4,200-4).BottomPos(0,30));
	Add(eps.HSizePos(200,0).BottomPos(0,30));
	eps <<= THISBACK(RefreshEpsilon);
	eps.MinMax(0, +100);
	eps.SetData(20);
	lbl_eps.SetLabel("Exploration epsilon: ");
	btnsplit.Horz();
	btnsplit << reset << goveryfast << gofast << gonorm << goslow;
	goveryfast.SetLabel("Unlimited speed");
	gofast.SetLabel("Go Fast");
	gonorm.SetLabel("Go Normal");
	goslow.SetLabel("Go Slow");
	reset.SetLabel("Reset");
	goveryfast	<<= THISBACK1(SetSpeed, 3);
	gofast	<<= THISBACK1(SetSpeed, 2);
	gonorm	<<= THISBACK1(SetSpeed, 1);
	goslow	<<= THISBACK1(SetSpeed, 0);
	reset	<<= THISBACK2(Reset, true, true);
	
	network_view.SetGraph(world.agents[0].GetGraph());
	
	SetSpeed(1);
	
	PostCallback(THISBACK2(Reset, true, true));
	PostCallback(THISBACK(Reload));
	RefreshEpsilon();
}

AirHockeyDQN::~AirHockeyDQN() {
	
}

void AirHockeyDQN::DockInit() {
	DockLeft(Dockable(agent_ctrl, "Edit Agent").SizeHint(Size(320, 240)));
	DockLeft(Dockable(statusctrl, "Status").SizeHint(Size(320, 240)));
	DockLeft(Dockable(reward[0], "#1 Reward graph").SizeHint(Size(320, 240)));
	DockLeft(Dockable(reward[1], "#2 Reward graph").SizeHint(Size(320, 240)));
	DockBottom(Dockable(network_view, "Network View").SizeHint(Size(640, 120)));
}

void AirHockeyDQN::Refresher() {
	world.Refresh();
	RefreshStatus();
	network_view.Refresh();
}

void AirHockeyDQN::Reset(bool init_reward, bool start) {
	Player& agent = world.agents[0];
	
	agent.do_training = true;
	
	if (init_reward) {
		int states = 152; // count of eyes
		int action_count = 4;
		
		agent.Init(1, states, 1, action_count);
		agent.Reset();
	}
	else {
		// Just reset values
		agent.ResetValues();
	}
}

void AirHockeyDQN::Reload() {
	String param_str = agent_edit.GetData();
	Player& agent = world.agents[0];
	
	//ticking_lock.Enter();
	agent.LoadInitJSON(param_str);
	//ticking_lock.Leave();
}

void AirHockeyDQN::RefreshEpsilon() {
	double d = (double)eps.GetData() / 100.0;
	Player& agent = world.agents[0];
	agent.SetEpsilon(d);
	lbl_eps.SetLabel("Exploration epsilon: " + FormatDoubleFix(d, 2));
}

void AirHockeyDQN::SetSpeed(int i) {
	simspeed = i;
}

void AirHockeyDQN::LoadPretrained() {
	Player& agent1 = world.agents[0];
	Player& agent2 = world.agents[1];
	
	// This is the pre-trained network from original ConvNetJS
	MemReadStream pretrained_mem(pretrained, pretrained_length);
	String json = BZ2Decompress(pretrained_mem);
	
	agent1.do_training = false;
	agent2.do_training = false;
	
	//ticking_lock.Enter();
	agent1.LoadJSON(json);
	agent2.LoadJSON(json);
	//ticking_lock.Leave();
}

void AirHockeyDQN::RefreshStatus() {
	Player& agent = world.agents[0];
	
	String s;
	s << "Experience write pointer: " << agent.GetExperienceWritePointer() << "\n";
	s << "Latest TD error: " << FormatDoubleFix(agent.GetTDError(), 3);
	status.SetLabel(s);
}

}
