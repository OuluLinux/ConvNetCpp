#include "GridWorld.h"

#define IMAGECLASS GridWorldImg
#define IMAGEFILE <GridWorld/GridWorld.iml>
#include <Draw/iml_source.h>

GridWorld::GridWorld() {
	Title("GridWorld: dynamic programming");
	Icon(GridWorldImg::icon());
	Sizeable().MaximizeBox().MinimizeBox();
	
	Add(btnsplit.HSizePos().TopPos(0,30));
	Add(gworld.HSizePos().VSizePos(30,30));
	Add(lbl_reward.LeftPos(4,200-4).BottomPos(0,30));
	Add(reward.HSizePos(200,0).BottomPos(0,30));
	gworld.SetAgent(agent);
	gworld.WhenGridFocus << THISBACK(GridFocus);
	gworld.WhenGridUnfocus << THISBACK(GridUnfocus);
	reward <<= THISBACK(RefreshReward);
	reward.MinMax(-500, +500);
	lbl_reward.SetLabel("Cell reward: (select a cell)");
	btnsplit.Horz();
	btnsplit << poleval << polup << toggle << reset;
	poleval.SetLabel("Policy Evaluation (once)");
	polup.SetLabel("Policy Update (once)");
	toggle.SetLabel("Toggle Value Iteration");
	reset.SetLabel("Reset");
	poleval	<<= THISBACK(EvaluatePolicy);
	polup	<<= THISBACK(UpdatePolicy);
	toggle	<<= THISBACK(ToggleIteration);
	reset	<<= THISBACK1(Reset, false);
	
	
	PostCallback(THISBACK1(Reset, true));
	
	SetTimeCallback(-40, THISBACK(Refresher));
}

GridWorld::~GridWorld() {
	agent.Stop();
}

void GridWorld::Refresher() {
	gworld.Refresh();
}

void GridWorld::Reset(bool init_reward) {
	agent.Stop();
	
	if (init_reward) {
		agent.Init(10,10);
		agent.Reset();
		
		
		agent.SetIterationDelay(100); // Slow down processing 100ms per iteration
		
		agent.SetGamma(0.9);
		
		agent.SetReward(3, 3, -1.0);
		agent.SetReward(5, 4, -1.0);
		agent.SetReward(6, 4, -1.0);
		agent.SetReward(5, 5, +1.0);
		agent.SetReward(6, 5, -1.0);
		agent.SetReward(8, 5, -1.0);
		agent.SetReward(8, 6, -1.0);
		agent.SetReward(3, 7, -1.0);
		agent.SetReward(5, 7, -1.0);
		agent.SetReward(6, 7, -1.0);
		
		// make some cliffs
		for (int q = 0; q < 8; q++) {
			if (q == 4) continue; // make a hole
			agent.SetDisabled(1+q, 2);
		}
		for (int q = 0; q < 6; q++) {
			agent.SetDisabled(4, 2+q);
		}
	}
	else {
		
		// Just reset values
		agent.ResetValues();
		
	}
	
	if (toggle.Get())
		agent.Start();
	
}

void GridWorld::UpdatePolicy() {
	if (agent.IsRunning()) return;
	agent.UpdatePolicy();
}

void GridWorld::EvaluatePolicy() {
	if (agent.IsRunning()) return;
	agent.EvaluatePolicy();
}

void GridWorld::ToggleIteration() {
	bool b = toggle.Get();
	if (!b)
		agent.Stop();
	else
		agent.Start();
}

void GridWorld::GridFocus() {
	int selected = gworld.GetSelected();
	if (selected == -1) return;
	
	double reward = agent.GetReward(selected);
	this->reward.SetData(reward * 100.0);
	lbl_reward.SetLabel("Cell reward: " + FormatDoubleFix(reward, 2));
}

void GridWorld::GridUnfocus() {
	lbl_reward.SetLabel("Cell reward: (select a cell)");
}

void GridWorld::RefreshReward() {
	int selected = gworld.GetSelected();
	if (selected == -1) return;
	
	double d = (double)reward.GetData() / 100.0;
	
	agent.SetReward(selected, d);
	GridFocus();
}
