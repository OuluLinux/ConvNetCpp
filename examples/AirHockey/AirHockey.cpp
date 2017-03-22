#include "AirHockey.h"
#include "pretrained.brc"
#include <plugin/bz2/bz2.h>

#define IMAGECLASS AirHockeyImg
#define IMAGEFILE <AirHockey/AirHockey.iml>
#include <Draw/iml_source.h>

namespace GameCtrl {

AirHockeyDQN::AirHockeyDQN() {
	Title("AirHockey: Deep Q Learning");
	Icon(AirHockeyImg::icon());
	Sizeable().MaximizeBox().MinimizeBox();
	
	world.agents[0].GameCtrl::Player::world = this;
	world.agents[1].GameCtrl::Player::world = this;
	
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
	reward[0].SetLimit(1000);
	reward[1].SetLimit(1000);
	
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
	btnsplit << reset << goveryfast << gofast << gonorm << show_dbg;
	goveryfast.SetLabel("Go Very Fast");
	gofast.SetLabel("Go Fast");
	gonorm.SetLabel("Go Normal");
	show_dbg.SetLabel("Show Eyes");
	reset.SetLabel("Reset");
	goveryfast	<<= THISBACK1(SetSpeed, 3);
	gofast	<<= THISBACK1(SetSpeed, 2);
	gonorm	<<= THISBACK1(SetSpeed, 1);
	reset	<<= THISBACK2(Reset, true, true);
	show_dbg.Set(true);
	show_dbg <<= THISBACK(SetDrawEyes);
	
	network_view.SetGraph(world.agents[0].GetGraph());
	
	SetSpeed(1);
	
	PostCallback(THISBACK2(Reset, true, true));
	PostCallback(THISBACK(Reload));
	RefreshEpsilon();
	
	SetTimeCallback(-1, THISBACK(Refresher));
}

AirHockeyDQN::~AirHockeyDQN() {
	world.Stop();
}

void AirHockeyDQN::DockInit() {
	DockLeft(Dockable(agent_ctrl, "Edit Agent").SizeHint(Size(320, 240)));
	DockLeft(Dockable(statusctrl, "Status").SizeHint(Size(320, 240)));
	DockLeft(Dockable(reward[0], "Upper player's reward").SizeHint(Size(320, 240)));
	DockLeft(Dockable(reward[1], "Lower player's reward").SizeHint(Size(320, 240)));
	//DockBottom(Dockable(network_view, "Network View").SizeHint(Size(640, 120)));
}

void AirHockeyDQN::Refresher() {
	world.Refresh();
	RefreshStatus();
	network_view.Refresh();
}

void AirHockeyDQN::Reset(bool init_reward, bool start) {
	
	world.agents[0].do_training = true;
	world.agents[1].do_training = true;
	
	if (init_reward) {
		int states = 152; // count of eyes
		int action_count = 4;
		
		world.agents[0].Init(1, states, 1, action_count);
		world.agents[1].Init(1, states, 1, action_count);
		world.agents[0].Reset();
		world.agents[1].Reset();
	}
	else {
		// Just reset values
		world.agents[0].ResetValues();
		world.agents[1].ResetValues();
	}
}

void AirHockeyDQN::Reload() {
	String param_str = agent_edit.GetData();
	
	world.agents[0].LoadInitJSON(param_str);
	world.agents[1].LoadInitJSON(param_str);
	
	world.Start();
}

void AirHockeyDQN::RefreshEpsilon() {
	double d = (double)eps.GetData() / 100.0;
	world.agents[0].SetEpsilon(d);
	world.agents[1].SetEpsilon(d);
	lbl_eps.SetLabel("Exploration epsilon: " + FormatDoubleFix(d, 2));
}

void AirHockeyDQN::SetSpeed(int i) {
	switch (i) {
		case 0: world.SetSpeed(true, 8); break;
		case 1: world.SetSpeed(false); break;
		case 2: world.SetSpeed(true, 16); break;
		case 3: world.SetSpeed(true, 100); break;
	}
	
	String json;
	
	FileOut out1(ConfigFile("best1.json"));
	world.agents[0].StoreJSON(json);
	out1 << json;
	
	FileOut out2(ConfigFile("best2.json"));
	world.agents[1].StoreJSON(json);
	out2 << json;
}

void AirHockeyDQN::SetDrawEyes() {
	bool b = show_dbg.Get();
	world.agents[0].paint_eyes = b;
	world.agents[1].paint_eyes = b;
}

void AirHockeyDQN::AddReward(int id, double value) {
	if (id < reward.GetCount()) {
		reward[id].AddValue(value);
	}
}

void AirHockeyDQN::LoadPretrained() {
	Player& agent1 = world.agents[0];
	Player& agent2 = world.agents[1];
	
	// This is the pre-trained network from original ConvNetJS
	MemReadStream pretrained_mem1(pretrained1, pretrained1_length);
	MemReadStream pretrained_mem2(pretrained2, pretrained2_length);
	String json1 = BZ2Decompress(pretrained_mem1);
	String json2 = BZ2Decompress(pretrained_mem2);
	
	agent1.do_training = false;
	agent2.do_training = false;
	
	world.lock.Enter();
	agent1.LoadJSON(json1);
	agent2.LoadJSON(json2);
	world.lock.Leave();
}

void AirHockeyDQN::RefreshStatus() {
	Player& agent0 = world.agents[0];
	Player& agent1 = world.agents[1];
	
	String s;
	s << "#1 Experience write pointer: " << agent0.GetExperienceWritePointer() << "\n";
	s << "#1 Latest TD error: " << FormatDoubleFix(agent0.GetTDError(), 3) << "\n";
	s << "#2 Experience write pointer: " << agent1.GetExperienceWritePointer() << "\n";
	s << "#2 Latest TD error: " << FormatDoubleFix(agent1.GetTDError(), 3);
	status.SetLabel(s);
}


}
