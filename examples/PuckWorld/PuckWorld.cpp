#include "PuckWorld.h"
#include "pretrained.brc"
#include <plugin/bz2/bz2.h>

PuckWorldAgent::PuckWorldAgent() {
	nflot = 1000;
}

void PuckWorldAgent::Reset() {
	DQNAgent::Reset();
	
	ppx = Randomf(); // puck x,y
	ppy = Randomf();
	pvx = Randomf() * 0.05 - 0.025; // velocity
	pvy = Randomf() * 0.05 - 0.025;
	tx  = Randomf(); // target
	ty  = Randomf();
	tx2 = Randomf(); // target
	ty2 = Randomf(); // target
	rad = 0.05;
	action = 0;
	t = 0;
	
	BADRAD = 0.25;
	
	smooth_reward = 0.0;
	flott = 0;
	reward = 0;
}

void PuckWorldAgent::SampleNextState(int x, int y, int d, int action, int& next_state, double& reward, bool& reset_episode) {
	
	// world dynamics
	ppx += pvx; // newton
	ppy += pvy;
	pvx *= 0.95; // damping
	pvy *= 0.95;
	
	// agent action influences puck velocity
	double accel = 0.002;
	bool gliding = false;
	if		(action == ACT_LEFT)	pvx -= accel;
	else if (action == ACT_RIGHT)	pvx += accel;
	else if (action == ACT_UP)		pvy -= accel;
	else if (action == ACT_DOWN)	pvy += accel;
	else gliding = true;
	
	// handle boundary conditions and bounce
	if (ppx < rad) {
		pvx *= -0.5; // bounce!
		ppx = rad;
	}
	if (ppx > 1 - rad) {
		pvx *= -0.5;
		ppx = 1 - rad;
	}
	if (ppy < rad) {
		pvy *= -0.5; // bounce!
		ppy = rad;
	}
	if (ppy > 1 - rad) {
		pvy *= -0.5;
		ppy = 1 - rad;
	}
	
	t += 1;
	
	if ((t % 100) == 0) {
		tx = Randomf(); // reset the target location
		ty = Randomf();
	}
	
	// compute distances
	double dx, dy, d1, d2;
	
	dx = ppx - tx;
	dy = ppy - ty;
	d1 = sqrt(dx*dx+dy*dy);
	
	dx = ppx - tx2;
	dy = ppy - ty2;
	d2 = sqrt(dx*dx+dy*dy);
	
	double dxnorm = dx/d2;
	double dynorm = dy/d2;
	double speed = 0.001;
	tx2 += speed * dxnorm;
	ty2 += speed * dynorm;
	
	// compute reward
	double r = -d1; // want to go close to green
	if (d2 < BADRAD) {
		// but if we're too close to red that's bad
		double f = (BADRAD - d2) / BADRAD;
		r -= 2 * f;
		
		// reset if too close
		/*if (f > 0.9) {
			tx2 = Randomf(); // reset the target location
			ty2 = Randomf();
		}*/
	}
	
	if (gliding) r += 0.05; // give bonus for gliding with no force
	
	reward = r;
}

void PuckWorldAgent::Learn() {
	Vector<double> slist;
	GetState(slist);
	action = Act(slist);
	
	int next_state;
	bool reset_episode = false;
    SampleNextState(0,0,0, action, next_state, reward, reset_episode); // run it through environment dynamics
    
	DQNAgent::Learn(reward);
	
	if (smooth_reward == 0.0)
		smooth_reward = reward;
	
	smooth_reward = smooth_reward * 0.999 + reward * 0.001;
	flott += 1;
	
	if (flott == 200) {
		// record smooth reward
		while (smooth_reward_history.GetCount() >= nflot) {
			smooth_reward_history.Remove(0);
		}
		smooth_reward_history.Add(smooth_reward);
		
		pworld->reward.SetLimit(nflot);
		pworld->reward.AddValue(smooth_reward);
		
		flott = 0;
	}
}

void PuckWorldAgent::GetState(Vector<double>& slist) {
	slist.SetCount(8);
	slist[0] = ppx - 0.5;
	slist[1] = ppy - 0.5;
	slist[2] = pvx * 10;
	slist[3] = pvy * 10;
	slist[4] = tx - ppx;
	slist[5] = ty - ppy;
	slist[6] = tx2 - ppx;
	slist[7] = ty2 - ppy;
}














PuckWorld::PuckWorld() {
	Title("PuckWorld: Deep Q Learning");
	Icon(PuckWorldImg::icon());
	Sizeable().MaximizeBox().MinimizeBox();
	
	agent.pworld = this;
	
	
	t =		"{\n"
			"\t\"update\":\"qlearn\",\n"
			"\t\"gamma\":0.9,\n"
			"\t\"epsilon\":0.2,\n"
			"\t\"alpha\":0.01,\n"
			"\t\"experience_add_every\":10,\n"
			"\t\"experience_size\":5000,\n"
			"\t\"learning_steps_per_iteration\":20,\n"
			"\t\"tderror_clamp\":1.0,\n"
			"\t\"num_hidden_units\":100,\n"
			"}\n";
	
	
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
	Add(pworld.HSizePos().VSizePos(30,30));
	Add(lbl_eps.LeftPos(4,200-4).BottomPos(0,30));
	Add(eps.HSizePos(200,0).BottomPos(0,30));
	pworld.SetAgent(agent);
	eps <<= THISBACK(RefreshEpsilon);
	eps.MinMax(0, +100);
	eps.SetData(50);
	lbl_eps.SetLabel("Exploration epsilon: ");
	btnsplit.Horz();
	btnsplit << reset << toggle << gofast << gonorm << goslow;
	gofast.SetLabel("Go Fast");
	gonorm.SetLabel("Go Normal");
	goslow.SetLabel("Go Slow");
	toggle.SetLabel("Toggle Iteration");
	reset.SetLabel("Reset");
	toggle.Set(true);
	gofast	<<= THISBACK1(SetSpeed, 2);
	gonorm	<<= THISBACK1(SetSpeed, 1);
	goslow	<<= THISBACK1(SetSpeed, 0);
	toggle	<<= THISBACK(ToggleIteration);
	reset	<<= THISBACK2(Reset, true, true);
	
	SetSpeed(1);
	
	PostCallback(THISBACK2(Reset, true, true));
	PostCallback(THISBACK(Reload));
	RefreshEpsilon();
	
	SetTimeCallback(-1, THISBACK(Refresher));
}

PuckWorld::~PuckWorld() {
	agent.Stop();
}

void PuckWorld::DockInit() {
	DockLeft(Dockable(agent_ctrl, "Edit Agent").SizeHint(Size(320, 240)));
	DockLeft(Dockable(statusctrl, "Status").SizeHint(Size(320, 240)));
	DockLeft(Dockable(reward, "Average reward graph").SizeHint(Size(320, 240)));
}

void PuckWorld::Refresher() {
	pworld.Refresh();
	RefreshStatus();
}

void PuckWorld::Reset(bool init_reward, bool start) {
	agent.Stop();
	
	if (init_reward) {
		int states = 8; // x,y,vx,vy, puck dx,dy
		int action_count = 5; // left, right, up, down, nothing
		
		agent.Init(1, states, 1, action_count);
		agent.Reset();
		
	}
	else {
		
		// Just reset values
		agent.ResetValues();
	}
	
	if (start && toggle.Get())
		agent.Start();
	
}

void PuckWorld::Reload() {
	Reset(false, false); // doesn't reset values
	
	String param_str = agent_edit.GetData();
	agent.LoadInitJSON(param_str);
	
	if (toggle.Get())
		agent.Start();
}

void PuckWorld::RefreshEpsilon() {
	double d = (double)eps.GetData() / 100.0;
	agent.SetEpsilon(d);
	lbl_eps.SetLabel("Exploration epsilon: " + FormatDoubleFix(d, 2));
}

void PuckWorld::SetSpeed(int i) {
	switch (i) {
		case 0: agent.SetIterationDelay(100);	break;
		case 1: agent.SetIterationDelay(10);	break;
		case 2: agent.SetIterationDelay(0);		break;
	}
}

void PuckWorld::ToggleIteration() {
	bool b = toggle.Get();
	if (!b)
		agent.Stop();
	else
		agent.Start();
}

void PuckWorld::LoadPretrained() {
	
	// This is the pre-trained network from original ConvNetJS
	MemReadStream pretrained_mem(pretrained, pretrained_length);
	String json = BZ2Decompress(pretrained_mem);
	
	agent.Stop();
	agent.LoadJSON(json);
	agent.Start();
}

void PuckWorld::RefreshStatus() {
	String s;
	s << "Experience write pointer: " << agent.GetExperienceWritePointer() << "\n";
	s << "Latest TD error: " << FormatDoubleFix(agent.GetTDError(), 3);
	status.SetLabel(s);
}
