#include "TemporalDifference.h"

/*
// agent parameter spec to play with (this gets eval()'d on Agent reset)
var spec = {}
spec.update = 'qlearn'; // 'qlearn' or 'sarsa'
spec.gamma = 0.3; // discount factor, [0, 1)
spec.epsilon = 0.5; // initial epsilon for epsilon-greedy policy, [0, 1)
spec.alpha = 0.8; // value function learning rate
spec.lambda = 0; // eligibility trace decay, [0,1). 0 = no eligibility traces
spec.replacing_traces = true; // use replacing or accumulating traces
spec.planN = 50; // number of planning steps per iteration. 0 = no planning

spec.smooth_policy_update = true; // non-standard, updates policy smoothly to follow max_a Q
spec.beta = 0.8; // learning rate for smooth policy update
*/

TemporalDifference::TemporalDifference() {
	Title("GridWorld: temporal difference");
	Icon(TemporalDifferenceImg::icon());
	Sizeable().MaximizeBox().MinimizeBox();
	
	running = false;
	
	
	t =		"{\n"
			"\t\"update\":\"qlearn\",\n"
			"\t\"gamma\":0.3,\n"
			"\t\"epsilon\":0.5,\n"
			"\t\"alpha\":0.8,\n"
			"\t\"lambda\":0,\n"
			"\t\"replacing_traces\":true,\n"
			"\t\"planN\":50,\n"
			"\t\"smooth_policy_update\":true,\n"
			"\t\"beta\":0.8,\n"
			"}\n";
	
	
	agent_edit.SetData(t);
	agent_ctrl.Add(agent_edit.HSizePos().VSizePos(0,30));
	agent_ctrl.Add(reload_btn.HSizePos().BottomPos(0,30));
	reload_btn.SetLabel("Reload Agent");
	reload_btn <<= THISBACK(Reload);
	
	
	Add(btnsplit.HSizePos().TopPos(0,30));
	Add(gworld.HSizePos().VSizePos(30,60));
	Add(lbl_reward.LeftPos(4,200-4).BottomPos(0,30));
	Add(lbl_eps.LeftPos(4,200-4).BottomPos(30,30));
	Add(reward.HSizePos(200,0).BottomPos(0,30));
	Add(eps.HSizePos(200,0).BottomPos(30,30));
	gworld.SetAgent(agent);
	gworld.WhenGridFocus << THISBACK(GridFocus);
	gworld.WhenGridUnfocus << THISBACK(GridUnfocus);
	reward <<= THISBACK(RefreshReward);
	reward.MinMax(-500, +500);
	eps <<= THISBACK(RefreshEpsilon);
	eps.MinMax(0, +100);
	eps.SetData(50);
	lbl_reward.SetLabel("Cell reward: (select a cell)");
	lbl_eps.SetLabel("Exploration epsilon: ");
	btnsplit.Horz();
	btnsplit << reset << toggle << gofast << gonorm << goslow;
	gofast.SetLabel("Go Fast");
	gonorm.SetLabel("Go Normal");
	goslow.SetLabel("Go Slow");
	toggle.SetLabel("Toggle Iteration");
	reset.SetLabel("Reset");
	gofast	<<= THISBACK1(SetSpeed, 2);
	gonorm	<<= THISBACK1(SetSpeed, 1);
	goslow	<<= THISBACK1(SetSpeed, 0);
	toggle	<<= THISBACK(ToggleIteration);
	reset	<<= THISBACK2(Reset, true, true);
	
	SetSpeed(1);
	
	PostCallback(THISBACK2(Reset, true, true));
	PostCallback(THISBACK(Reload));
	Start();
}

TemporalDifference::~TemporalDifference() {
	agent.Stop();
}

void TemporalDifference::DockInit() {
	//AutoHide(DOCK_LEFT, Dockable(agent_ctrl, "Edit Agent").SizeHint(Size(640, 320)));
	DockLeft(Dockable(agent_ctrl, "Edit Agent").SizeHint(Size(320, 240)));
}

void TemporalDifference::Refresher() {
	gworld.Refresh();
	
	if (running) PostCallback(THISBACK(Refresher));
}

void TemporalDifference::Reload() {
	Reset(false, false); // doesn't reset values
	
	String param_str = agent_edit.GetData();
	agent.LoadJSON(param_str);
	
	if (toggle.Get())
		agent.Start();
	
	GridFocus();
}

void TemporalDifference::Start() {
	if (!running) {
		running = true;
		PostCallback(THISBACK(Refresher));
	}
}


void TemporalDifference::Reset(bool init_reward, bool start) {
	agent.Stop();
	
	if (init_reward) {
		agent.Init(10,10,1);
		agent.Reset();
		
		agent.SetReward(3, 3, 0, -1.0);
		agent.SetReward(5, 2, 0, -1.2);
		agent.SetReward(9, 2, 0, +0.3);
		agent.SetReward(5, 4, 0, -1.0);
		agent.SetReward(6, 4, 0, -1.0);
		agent.SetReward(5, 5, 0, +1.0);
		agent.SetReward(6, 5, 0, -1.0);
		agent.SetReward(8, 5, 0, -1.0);
		agent.SetReward(8, 6, 0, -1.0);
		agent.SetReward(3, 7, 0, -1.0);
		agent.SetReward(5, 7, 0, -1.0);
		agent.SetReward(6, 7, 0, -1.0);
		
		// make some cliffs
		for (int q = 0; q < 8; q++) {
			if (q == 4) continue; // make a hole
			agent.SetDisabled(1+q, 2, 0);
		}
		for (int q = 0; q < 6; q++) {
			agent.SetDisabled(4, 2+q, 0);
		}
	}
	else {
		
		// Just reset values
		agent.ResetValues();
		
	}
	
	if (start && toggle.Get())
		agent.Start();
	
}

void TemporalDifference::ToggleIteration() {
	bool b = toggle.Get();
	if (!b)
		agent.Stop();
	else
		agent.Start();
}

void TemporalDifference::SetSpeed(int i) {
	switch (i) {
		case 0: agent.SetIterationDelay(500);	break;
		case 1: agent.SetIterationDelay(100);	break;
		case 2: agent.SetIterationDelay(1);		break;
	}
}

void TemporalDifference::GridFocus() {
	double eps = agent.GetEpsilon();
	this->eps.SetData(eps * 100.0);
	lbl_eps.SetLabel("Exploration epsilon: " + FormatDoubleFix(eps, 2));
	
	int selected = gworld.GetSelected();
	if (selected == -1) return;
	
	double reward = agent.GetReward(selected);
	this->reward.SetData(reward * 100.0);
	lbl_reward.SetLabel("Cell reward: " + FormatDoubleFix(reward, 2));
}

void TemporalDifference::GridUnfocus() {
	lbl_reward.SetLabel("Cell reward: (select a cell)");
}

void TemporalDifference::RefreshReward() {
	int selected = gworld.GetSelected();
	if (selected == -1) return;
	
	double d = (double)reward.GetData() / 100.0;
	
	agent.SetReward(selected, d);
	GridFocus();
}

void TemporalDifference::RefreshEpsilon() {
	double d = (double)eps.GetData() / 100.0;
	agent.SetEpsilon(d);
	GridFocus();
}
