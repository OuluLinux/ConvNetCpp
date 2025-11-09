#include "ConvNet.h"

namespace ConvNet {

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define STOREVAR_(field) map.GetAdd(#field) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVAR_(field) this->field = map.GetValue(map.Find(#field));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}
#define LOADVARDEF_(field, def) {Value tmp = map.GetValue(map.Find(#field)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}
#define LOADVARDEFTEMP(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) field = def; else field = tmp;}

// return Mat but filled with random numbers from gaussian
void RandMat(int n, int d, double mu, double std, Mat& m) {
	m.Init(d, n, 0.0);
	
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mu, std);
	generator.seed(Random(INT_MAX));
	
	for (int i = 0; i < m.GetLength(); i++)
		m.Set(i, distribution(generator));
}

int SampleWeighted(Vector<double>& p) {
	ASSERT(!p.IsEmpty());
	double r = Randomf();
	double c = 0.0;
	for (int i = 0; i < p.GetCount(); i++) {
		c += p[i];
		if (c >= r)
			return i;
	}
	Panic("Invalid input vector");
	return 0;
}

void UpdateMat(Mat& m, double alpha) {
	// updates in place
	for (int i = 0; i < m.GetLength(); i++) {
		double d = m.GetGradient(i);
		if (d != 0.0) {
			m.Add(i, -alpha * d);
			m.SetGradient(i, 0);
		}
	}
}


















Agent::Agent() {
	width = 0;
	height = 0;
	action_count = 0;
	start_state = 0;
	stop_state = -1;
	
	iter_sleep = 0;
	
	running = false;
	stopped = true;
}

Agent::~Agent() {
	Stop();
}

void Agent::Serialize(Stream& s) {
	MatPool::Serialize(s);
	
	s % poldist
	  % reward
	  % value
	  % disable
	  % width % height % length
	  % action_count
	  % start_state % stop_state
	  % iter_sleep
	  % running % stopped;
}

void Agent::Init(int width, int height, int action_count) {
	ASSERT(width > 0 && height > 0);
	ASSERT(stopped);
	
	if (action_count <= 0) {
		action_count =
			(width  > 1 ? 2 : 1) *
			(height > 1 ? 2 : 1);
	}
	
	// hardcoding one gridworld for now
	this->width		= width;
	this->height	= height;
	length = height * width; // number of states
	this->action_count = action_count;
	
	// reset the agent's policy and value function
	value.SetCount(0);
	value.SetCount(length, 0.0);
	poldist.SetCount(0);
	poldist.SetCount(action_count, value); // use cleared value vector as template, no other relation
	
	// specify some rewards
	reward.SetCount(0);
	reward.SetCount(length, 0.0);
	disable.SetCount(0);
	disable.SetCount(length, false);
	
	SetStopState(width / 2, height / 2);
}

bool Agent::LoadInitJSON(const String& json) {
	
	Value js = ParseJSON(json);
	if (js.IsNull()) {
		LOG("JSON parse failed");
		return false;
	}
	
	ValueMap map = js;
	LoadInit(map);
	return true;
}

void Agent::LoadInit(const ValueMap& map) {
	
}

void Agent::Reset() {
	ClearPool();
	
	Agent::Init(width, height, action_count);
	
}

void Agent::ResetValues() {
	value.SetCount(0);
	value.SetCount(length, 0.0);
}

void Agent::Start() {
	if (running) return;
	stopped = false;
	running = true;
	Thread::Start(THISBACK(Run));
}

void Agent::Stop() {
	running = false;
	while (!stopped) Sleep(100);
}

void Agent::Run() {
	while (running) {
		ValueIteration();
		if (iter_sleep)
			Sleep(iter_sleep);
	}
	stopped = true;
}

void Agent::ValueIteration() {
	// calling Learn() is probably enough, but at least here is room for improvements now
	Learn();
}

int Agent::GetPos(int x, int y) const {
	ASSERT(x >= 0 && y >= 0 && x < width && y < height);
	return (width * y) + x;
}

void Agent::GetXY(int state, int& x, int& y) const {
	x = state % width;
	state /= width;
	y = state;
}

void Agent::AllowedActions(int x, int y, Vector<int>& actions) const {
	actions.SetCount(0);
	if (IsDisabled(x,y)) return;
	if (x > 0) { actions.Add(ACT_LEFT); }
	if (y > 0) { actions.Add(ACT_DOWN); }
	if (y < height-1) { actions.Add(ACT_UP); }
	if (x < width-1) { actions.Add(ACT_RIGHT); }
}

void Agent::SetReward(int x, int y, double reward) {
	int ix = (width * y) + x;
	this->reward[ix] = reward;
}

void Agent::SetDisabled(int x, int y, bool disable) {
	int ix = (width * y) + x;
	this->disable[ix] = disable;
	if (disable)
		this->reward[ix] = 0;
}

double Agent::Reward(int s, int a, int ns) {
	// reward of being in s, taking action a, and ending up in ns
	return reward[s];
}

int Agent::GetNextStateDistribution(int x, int y, int a) {
	int ns = GetPos(x,y);
	
	// given (s,a) return distribution over s' (in sparse form)
	if (IsDisabled(ns)) {
		// cliff! oh no!
		Panic("Disabled position. This call shuold be avoided by caller.");
	}
	else if (ns == stop_state) {
		// agent wins! teleport to start
		ns = GetStartState();
		while (IsDisabled(ns)) {
			ns = GetRandomState();
		}
	}
	else {
		// ordinary space
		double  nx = x, ny = y;
		if      (a == ACT_LEFT)		{nx=x-1;}
		else if (a == ACT_DOWN)		{ny=y-1;}
		else if (a == ACT_UP)		{ny=y+1;}
		else if (a == ACT_RIGHT)	{nx=x+1;}
		int ns2 = GetPos(nx, ny);
		if (IsDisabled(ns2)) {
			// actually never mind, this is a wall. reset the agent
			//ns = state;
		}
		else ns = ns2;
	}
	
	// gridworld is deterministic, so return only a single next state
	return ns;
}

void Agent::SampleNextState(int x, int y, int action, int& next_state, double& reward, bool& reset_episode) {
	int state = GetPos(x,y);
	// gridworld is deterministic, so this is easy
	next_state = GetNextStateDistribution(x, y, action);
	reward = this->reward[state]; // observe the raw reward of being in s, taking a, and ending up in ns
	reward -= 0.01; // every step takes a bit of negative reward
	reset_episode = (state == stop_state && next_state == start_state); // episode is over
}


















DPAgent::DPAgent() {
	gamma = 0.75;
	
}

void DPAgent::Reset() {
	Agent::Reset();
	
	// initialize uniform random policy
	Vector<int> poss;
	int state = 0;
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			AllowedActions(x, y, poss);
			double prob = 1.0 / poss.GetCount();
			for (int i = 0; i < poss.GetCount(); i++) {
				poldist[poss[i]][state] = prob;
			}
			state++;
		}
	}
}

int DPAgent::Act(int x, int y) {
	// behave according to the learned policy
	int state = GetPos(x,y);
	Vector<int> poss;
	AllowedActions(x, y, poss);
	Vector<double> ps;
	ps.SetCount(poss.GetCount());
	for (int i = 0; i < poss.GetCount(); i++) {
		int a = poss[i];
		double prob = poldist[a][state];
		ps[i] = prob;
	}
	int maxi = SampleWeighted(ps);
	return poss[maxi];
}

void DPAgent::Learn() {
	// perform a single round of value iteration
	EvaluatePolicy(); // writes policy value
	UpdatePolicy(); // writes policy distribution
}

void DPAgent::EvaluatePolicy() {
	
	// perform a synchronous update of the value function
	value.SetCount(length);
	
	Vector<int> poss;
	int state = 0;
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			
			// integrate over actions in a stochastic policy
			// note that we assume that policy probability mass over allowed actions sums to one
			double v = 0.0;
			AllowedActions(x, y, poss);
			for (int i = 0; i < poss.GetCount(); i++) {
				int a = poss[i];
				double prob = poldist[a][state]; // probability of taking action under policy
				if (prob == 0) { continue; } // no contribution, skip for speed
				int ns = GetNextStateDistribution(x, y, a);
				double rs = Reward(state, a, ns); // reward for s->a->ns transition
				v += prob * (rs + gamma * value[ns]);
			}
			
			value[state] = v;
			state++;
		}
	}
}

void DPAgent::UpdatePolicy() {
	
	Vector<int> poss;
	Vector<double> vs;
	Vector<int> maxpos;
	int state = 0;
	
	// update policy to be greedy w.r.t. learned Value function
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			AllowedActions(x, y, poss);
			if (poss.IsEmpty()) {
				state++;
				continue;
			}
			
			// compute value of taking each allowed action
			int nmax = 0;
			double vmax = -DBL_MAX;
			
			vs.SetCount(poss.GetCount());
			
			for (int i = 0, n = poss.GetCount(); i < n; i++) {
				int a = poss[i];
				int ns = GetNextStateDistribution(x, y, a);
				double rs = Reward(state, a, ns);
				double v = rs + gamma * value[ns];
				vs[i] = v;
				if (v > vmax) { vmax = v; nmax = 1; maxpos.SetCount(1); maxpos[0] = i;}
				else if (v == vmax) { nmax += 1; maxpos.Add(i);}
			}
			
			// update policy smoothly across all argmaxy actions
			for (int i = 0; i < poss.GetCount(); i++) {
				int a = poss[i];
				poldist[a][state] = 0.0;
			}
			
			for (int i = 0; i < maxpos.GetCount(); i++) {
				int a = poss[maxpos[i]];
				poldist[a][state] = 1.0 / nmax;
			}
			
			state++;
		}
	}
}


















TDAgent::TDAgent() {
	update = UPDATE_QLEARN;
	gamma = 0.75;
	epsilon = 0.1;
	alpha = 0.01;
	
	// class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
	smooth_policy_update = false;
	beta = 0.01;
	
	// eligibility traces
	lambda = 0; // TODO: check if double instead of int
	replacing_traces = true;
	
	// optional optimistic initial values
	q_init_val = 0;
	
	planN = 0;
	
	nflot = 100;
	
}

void TDAgent::LoadInit(const ValueMap& map) {
	Agent::LoadInit(map);
	
	String update_str;
	LOADVARDEFTEMP(update_str, update, "");
	update = update_str == "sarsa" ? UPDATE_SARSA : UPDATE_QLEARN;
	
	LOADVARDEF(gamma, gamma, 0.75);
	LOADVARDEF(epsilon, epsilon, 0.1);
	LOADVARDEF(alpha, alpha, 0.01);
	LOADVARDEF(replacing_traces, replacing_traces, true);
	LOADVARDEF(planN, planN, 0);
	LOADVARDEF(smooth_policy_update, smooth_policy_update, false);
	LOADVARDEF(beta, beta, 0.01);
}

void TDAgent::Reset(){
	Agent::Reset();
	
	// reset the agent's policy and value function
	ns = GetNumStates();
	na = GetMaxNumActions();
	
	
	Q.SetCount(na);
	P.SetCount(na);
	e.SetCount(na);
	pq.SetCount(na);
	env_model_s.SetCount(na);
	env_model_r.SetCount(na);
	poldist.SetCount(na);
	for(int i = 0; i < na; i++) {
		
		// This could be done differently by setting first target count and then looping values
		
		// Set size to 0 without potentially unallocating memory
		Q[i].SetCount(0);
		P[i].SetCount(0);
		e[i].SetCount(0);
		pq[i].SetCount(0);
		env_model_s[i].SetCount(0);
		env_model_r[i].SetCount(0);
		poldist[i].SetCount(0);
		
		Q[i].SetCount(ns, q_init_val);
		P[i].SetCount(ns, 0);
		e[i].SetCount(ns, 0);
		pq[i].SetCount(ns, 0);
		env_model_s[i].SetCount(ns, -1);// init to -1 so we can test if we saw the state before
		env_model_r[i].SetCount(ns, 0);
		poldist[i].SetCount(ns, 0);
	}
	
	
	
	// model/planning vars
	sa_seen.SetCount(0);
	
	Vector<int> poss;
	
	// initialize uniform random policy
	int state = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			AllowedActions(x, y, poss);
			for (int i = 0; i < poss.GetCount(); i++) {
				poldist[poss[i]][state] = 1.0 / poss.GetCount();
			}
			state++;
		}
	}
	
	// agent memory, needed for streaming updates
	// (s0,a0,reward0,s1,a1,r1,...)
	reward0 = 0;
	state0 = 0;
	state1 = 0;
	action0 = 0;
	action1 = 0;
	
	current_state = 0;
	nsteps_counter = 0;
	explored = false;
}

void TDAgent::ResetEpisode() {
	// an episode finished
}

int TDAgent::Act(int x, int y) {
	
	// act according to epsilon greedy policy
	Vector<int> poss;
	AllowedActions(x, y, poss);
	ASSERT(!poss.IsEmpty());
	
	int state = GetPos(x,y);
	
	Vector<double> probs;
	
	probs.SetCount(poss.GetCount());
	for (int i = 0; i < poss.GetCount(); i++) {
		probs[i] = poldist[poss[i]][state];
	}
	
	// epsilon greedy policy
	int action;
	if (Randomf() < epsilon) {
		action = poss[Random(poss.GetCount())]; // random available action
		explored = true;
	} else {
		action = poss[SampleWeighted(probs)];
		explored = false;
	}
	
	// shift state memory
	state0 = state1;
	action0 = action1;
	state1 = state;
	action1 = action;
	
	return action;
}

double TDAgent::GetValue(int x, int y) const {
	Vector<int> poss;
	AllowedActions(x, y, poss);
	if (poss.IsEmpty()) return 0.0;
	int state = GetPos(x,y);
	double r;
	for(int i = 0; i < poss.GetCount(); i++) {
		double q = Q[poss[i]][state];
		if (i == 0 || q > r)
			r = q;
	}
	return r;
}

void TDAgent::Learn() {
	int x, y, d;
	GetXY(current_state, x, y);
	
	int action = Act(x,y); // ask agent for an action
	int next_state;
	double reward;
	bool reset_episode = false;
    SampleNextState(x,y, action, next_state, reward, reset_episode); // run it through environment dynamics
    Learn(reward); // allow opportunity for the agent to learn
    current_state = next_state; // evolve environment to next state
    nsteps_counter += 1;
    if (reset_episode) {
		ResetEpisode();
		
		// record the reward achieved
		while (nsteps_history.GetCount() >= nflot) {nsteps_history.Remove(0);}
		nsteps_history.Add(nsteps_counter);
		nsteps_counter = 0;
    }
}

void TDAgent::Learn(double reward1){
	
	// takes reward for previous action, which came from a call to act()
	if (!(reward0 == 0.0)) {
		LearnFromTuple(state0, action0, reward0, state1, action1, lambda);
		if (planN > 0) {
			UpdateModel(state0, action0, reward0, state1);
			Plan();
		}
	}
	reward0 = reward1; // store this for next update
}

void TDAgent::UpdateModel(int state0, int action0, double reward0, int state1) {
	
	// transition (s0,a0) -> (reward0,s1) was observed. Update environment model
	if (env_model_s[action0][state0] == -1) {
		// first time we see this state action
		ActionState& as = sa_seen.Add();
		as.a = action0;
		as.b = state0; // add as seen state
	}
	env_model_s[action0][state0] = state1;
	env_model_r[action0][state0] = reward0;
}

void TDAgent::Plan() {
	
	struct ValSorter {
		bool operator() (const Val& a, const Val& b) const {return a.b < b.b;}
	};
	
	// order the states based on current priority queue information
	Vector<Val> spq;
	for (int i = 0; i < sa_seen.GetCount(); i++) {
		ActionState& sa = sa_seen[i];
		double sap = pq[sa.a][sa.b];
		if (sap > 1e-5) { // gain a bit of efficiency
			Val& v = spq.Add();
			v.a = sa;
			v.b = sap;
		}
	}
	
	// perform the updates
	int nsteps = min(planN, spq.GetCount());
	for (int k = 0; k < nsteps; k++) {
		// random exploration
		//var i = randi(0, sa_seen.length); // pick random prev seen state action
		//var s0a0 = sa_seen[i];
		ActionState& as = spq[k].a;
		int action0 = as.a;
		int state0 = as.b;
		pq[action0][state0] = 0; // erase priority, since we're backing up this state
		double reward0 = env_model_r[action0][state0];
		double state1 = env_model_s[action0][state0];
		int action1 = -1; // not used for Q learning
		if (update == UPDATE_SARSA) {
			
			// generate random action?...
			Vector<int> poss;
			int x,y,d;
			GetXY(state1, x, y);
			AllowedActions(x, y, poss);
			action1 = poss[Random(poss.GetCount())];
		}
		LearnFromTuple(state0, action0, reward0, state1, action1, 0); // note lambda = 0 - shouldnt use eligibility trace here
	}
}

void TDAgent::LearnFromTuple(int state0, int action0, double reward0, int state1, int action1, double lambda) {
	int s0x, s0y, s1x, s1y;
	
	GetXY(state0, s0x, s0y);
	GetXY(state1, s1x, s1y);
	
	Vector<int> poss;
	
	// calculate the target for Q(s,a)
	double target;
	if (update == UPDATE_QLEARN) {
		
		// Q learning target is Q(s0,a0) = reward0 + gamma * max_a Q[s1,a]
		AllowedActions(s1x, s1y, poss);
		double qmax = 0.0;
		for (int i = 0; i < poss.GetCount(); i++) {
			double qval = Q[poss[i]][state1];
			if (i == 0 || qval > qmax) {
				qmax = qval;
			}
		}
		target = reward0 + gamma * qmax;
	}
	else if (update == UPDATE_SARSA) {
		// SARSA target is Q(s0,a0) = reward0 + gamma * Q[s1,a1]
		target = reward0 + gamma * Q[action1][state1];
	}
	
	if (lambda > 0.0) {
		// perform an eligibility trace update
		if(replacing_traces) {
			e[action0][state0] = 1;
		}
		else {
			e[action0][state0] += 1;
		}
		double edecay = lambda * gamma;
		
		Vector<double> state_update;
		state_update.SetCount(length);
		
		
		int state = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				AllowedActions(x, y, poss);
				
				for (int i = 0; i < poss.GetCount(); i++) {
					int action = poss[i];
					double esa = e[action][state];
					double update = alpha * esa * (target - Q[action][state]);
					Q[action][state] += update;
					UpdatePriority(state, action, update);
					e[action][state] *= edecay;
					double u = fabs(update);
					if (u > state_update[state]) {
						state_update[state] = u;
					}
				}
				
				state++;
			}
		}
		
		for (int l = 0; l < length; l++) {
			// save efficiency here
			if (state_update[l] > 1e-5) {
				int x,y;
				GetXY(l,x,y); // so, this has to be done usually anyway, and better do it pre-emptively
				UpdatePolicy(x,y);
			}
		}
		
		if (explored && update == UPDATE_QLEARN) {
			// have to wipe the trace since q learning is off-policy :(
			e.SetCount(action_count);
			for(int i = 0; i < e.GetCount(); i++)
				e[i].SetCount(length);
		}
	} else {
		// simpler and faster update without eligibility trace
		// update Q[sa] towards it with some step size
		double& q = Q[action0][state0];
		double update = alpha * (target - q);
		q += update;
		UpdatePriority(state0, action0, update);
		
		// update the policy to reflect the change (if appropriate)
		int x,y,d;
		GetXY(state0, x, y);
		UpdatePolicy(x, y);
	}
}

void TDAgent::UpdatePriority(int s, int a, double u) {
	// used in planning. Invoked when Q[sa] += update
	// we should find all states that lead to (s,a) and upgrade their priority
	// of being update in the next planning step
	u = fabs(u);
	if (u < 1e-5) { return; } // for efficiency skip small updates
	if (planN == 0) { return; } // there is no planning to be done, skip.
	for (int si = 0; si < ns; si++) {
		// note we are also iterating over impossible actions at all states,
		// but this should be okay because their env_model_s should simply be -1
		// as initialized, so they will never be predicted to point to any state
		// because they will never be observed, and hence never be added to the model
		for (int ai = 0; ai < na; ai++) {
			if (env_model_s[ai][si] == s) {
				// this state leads to s, add it to priority queue
				pq[ai][si] += u;
			}
		}
	}
}

void TDAgent::UpdatePolicy(int x, int y) {
	
	Vector<int> poss;
	AllowedActions(x, y, poss);
	ASSERT(!poss.IsEmpty());
	
	// set policy at s to be the action that achieves max_a Q(s,a)
	// first find the maxy Q values
	int nmax;
	double qmax;
	Vector<double> qs;
	
	int s = GetPos(x,y);
	qs.SetCount(poss.GetCount());
	
	for (int i = 0; i < poss.GetCount(); i++) {
		int a = poss[i];
		double qval = Q[a][s];
		qs[i] = qval;
		if (i == 0 || qval > qmax) { qmax = qval; nmax = 1; }
		else if(qval == qmax) { nmax += 1; }
	}
	
	// now update the policy smoothly towards the argmaxy actions
	double psum = 0.0;
	for (int i = 0; i < poss.GetCount(); i++) {
		int a = poss[i];
		double target = (qs[i] == qmax) ? 1.0/nmax : 0.0;
		if (smooth_policy_update) {
			// slightly hacky :p
			double& pd = poldist[a][s];
			pd += beta * (target - pd);
			psum += pd;
		} else {
			// set hard target
			poldist[a][s] = target;
		}
	}
	if (smooth_policy_update) {
		// renomalize P if we're using smooth policy updates
		for (int i = 0, n = poss.GetCount(); i < n; i++) {
			int a = poss[i];
			poldist[a][s] /= psum;
		}
	}
}



















DQNAgent::DQNAgent() {
	G.SetPool(*this);
	
	gamma = 0.75; // future reward discount factor
	epsilon = 0.1; // for epsilon-greedy policy
	alpha = 0.01; // value function learning rate
	
	experience_add_every = 25; // number of time steps before we add another experience to replay memory
	experience_size = 5000; // size of experience replay
	learning_steps_per_iteration = 10;
	tderror_clamp = 1.0;
	
	num_hidden_units = 100;
	
	expi = 0;
	t = 0;
	reward0 = 0;
	action0 = 0;
	action1 = 0;
	has_reward = false;
	tderror = 0;
	
}

void DQNAgent::Reset() {
	ClearPool();
	
	nh = num_hidden_units; // number of hidden units
	ns = GetNumStates();
	na = GetMaxNumActions();
	
	RandMat(nh, ns, 0, 0.01, net.W1);
	InitMat(net.b1, 1, nh, 0);
	//net.b1 = RandMat(nh, 1, 0, 0.01);
	RandMat(na, nh, 0, 0.01, net.W2);
	InitMat(net.b2, 1, na, 0);
	//net.b2 = RandMat(na, 1, 0, 0.01);
	InitMat(state);
	
	// Original:
	//		var a1mat = G.Add(G.Mul(net.W1, s), net.b1);
	//		var h1mat = G.Tanh(a1mat);
	//		var a2mat = G.Add(G.Mul(net.W2, h1mat), net.b2);
	G.Clear();
	G.Mul(net.W1);
	G.Add(net.b1);
	G.Tanh();
	G.Mul(net.W2);
	G.Add(net.b2);
	
	expi = 0; // where to insert
	
	t = 0;
	
	reward0 = 0;
	action0 = 0;
	action1 = 0;
	has_reward = false;
	
	tderror = 0; // for visualization only...
}

void DQNAgent::LoadInit(const ValueMap& map) {
	LOADVARDEF(gamma, gamma, 0.75); // future reward discount factor
	LOADVARDEF(epsilon, epsilon, 0.1); // for epsilon-greedy policy
	LOADVARDEF(alpha, alpha, 0.01); // value function learning rate
	LOADVARDEF(experience_add_every, experience_add_every, 25); // number of time steps before we add another experience to replay memory
	LOADVARDEF(experience_size, experience_size, 5000); // size of experience replay
	LOADVARDEF(learning_steps_per_iteration, learning_steps_per_iteration, 10);
	LOADVARDEF(tderror_clamp, tderror_clamp, 1.0);
	LOADVARDEF(num_hidden_units, num_hidden_units, 100);
}

int DQNAgent::Act(int x, int y) {
	Panic("Not useful");
	return 0;
}

int DQNAgent::Act(const Vector<double>& slist) {
	
	// convert to a Mat column vector
	Mat& state_mat = Get(state);
	state_mat.Init(width, height, slist);
	
	// epsilon greedy policy
	int action;
	if (Randomf() < epsilon) {
		action = Random(na);
	} else {
		// greedy wrt Q function
		//Mat& amat = ForwardQ(net, state);
		MatId a = G.Forward(state);
		Mat& amat = Get(a);
		action = amat.GetMaxColumn(); // returns index of argmax action
	}
	
	// shift state memory
	state0 = state1;
	action0 = action1;
	state1 = state_mat;
	action1 = action;
	
	return action;
}

void DQNAgent::Learn() {
	Panic("TODO");
}

void DQNAgent::Learn(double reward1, bool force_experience) {
	
	// perform an update on Q function
	if (has_reward && alpha > 0 && state0.GetLength() > 0) {
		
		// learn from this tuple to get a sense of how "surprising" it is to the agent
		tderror = LearnFromTuple(state0, action0, reward0, state1, action1); // a measure of surprise
		
		// decide if we should keep this experience in the replay
		if (t % experience_add_every == 0 || force_experience) {
			if (exp.GetCount() == expi)
				exp.Add();
			ASSERT(state1.GetLength() > 0);
			exp[expi].Set(state0, action0, reward0, state1, action1);
			expi += 1;
			if (expi >= experience_size) { expi = 0; } // roll over when we run out
		}
		t += 1;
		
		if (!exp.IsEmpty()) {
			// sample some additional experience from replay memory and learn from it
			for (int k = 0; k < learning_steps_per_iteration; k++) {
				int ri = Random(exp.GetCount()); // TODO: priority sweeps?
				DQExperience& e = exp[ri];
				LearnFromTuple(e.state0, e.action0, e.reward0, e.state1, e.action1);
			}
		}
	}
	reward0 = reward1; // store for next update
	has_reward = true;
}

double DQNAgent::LearnFromTuple(Mat& s0, int a0, double reward0, Mat& s1, int a1) {
	ASSERT(s0.GetLength() > 0);
	ASSERT(s1.GetLength() > 0);
	// want: Q(s,a) = r + gamma * max_a' Q(s',a')
	
	MatId s0_id = AddTempMat(s0);
	MatId s1_id = AddTempMat(s1);
	
	// compute the target Q value
	MatId t = G.Forward(s1_id);
	Mat& tmat = Get(t);
	double qmax = reward0 + gamma * tmat.Get(tmat.GetMaxColumn());
	
	// now predict
	MatId pred = G.Forward(s0_id);
	Mat& pred_mat = Get(pred);
	
	double tderror = pred_mat.Get(a0) - qmax;
	double clamp = tderror_clamp;
	if (fabs(tderror) > clamp) {  // huber loss to robustify
		if (tderror > clamp)
			tderror = +clamp;
		else
			tderror = -clamp;
	}
	pred_mat.SetGradient(a0, tderror);
	G.Backward(); // compute gradients on net params
	
	// update net
	UpdateMat(Get(net.W1), alpha);
	UpdateMat(Get(net.b1), alpha);
	UpdateMat(Get(net.W2), alpha);
	UpdateMat(Get(net.b2), alpha);
	
	ClearTempMat();
	
	return tderror;
}

void DQNAgent::Evaluate(const Vector<double>& in, Vector<double>& out) {
	
	// convert to a Mat column vector
	Mat& state_mat = Get(state);
	state_mat.Init(width, height, in);
	
	// greedy wrt Q function
	//Mat& amat = ForwardQ(net, state);
	MatId a = G.Forward(state);
	Mat& mat = Get(a);
	
	out.SetCount(mat.GetLength());
	for(int i = 0; i < mat.GetLength(); i++)
		out[i] = mat.Get(i);
	
}

void DQNAgent::Evaluate(double* in, double* out) {
	
	// convert to a Mat column vector
	Mat& state_mat = Get(state);
	state_mat.Init(width, height, in);
	
	// greedy wrt Q function
	//Mat& amat = ForwardQ(net, state);
	MatId a = G.Forward(state);
	Mat& mat = Get(a);
	
	for(int i = 0; i < mat.GetLength(); i++)
		out[i] = mat.Get(i);
	
}

void DQNAgent::Learn(const Vector<double>& in, const Vector<double>& out) {
	// convert to a Mat column vector
	Mat& state_mat = Get(state);
	state_mat.Init(width, height, in);
	
	// now predict
	MatId pred = G.Forward(state);
	Mat& pred_mat = Get(pred);
	
	for(int i = 0; i < out.GetCount(); i++) {
		double tderror = pred_mat.Get(i) - out[i];
		double clamp = tderror_clamp;
		if (fabs(tderror) > clamp) {  // huber loss to robustify
			if (tderror > clamp)
				tderror = +clamp;
			else
				tderror = -clamp;
		}
		pred_mat.SetGradient(i, tderror);
	}
	G.Backward(); // compute gradients on net params
	
	// update net
	UpdateMat(Get(net.W1), alpha);
	UpdateMat(Get(net.b1), alpha);
	UpdateMat(Get(net.W2), alpha);
	UpdateMat(Get(net.b2), alpha);
	
	ClearTempMat();
	
}

void DQNAgent::Learn(double* in, double* out) {
	// convert to a Mat column vector
	Mat& state_mat = Get(state);
	state_mat.Init(width, height, in);
	
	// now predict
	MatId pred = G.Forward(state);
	Mat& pred_mat = Get(pred);
	
	for(int i = 0; i < action_count; i++) {
		double tderror = pred_mat.Get(i) - out[i];
		double clamp = tderror_clamp;
		if (fabs(tderror) > clamp) {  // huber loss to robustify
			if (tderror > clamp)
				tderror = +clamp;
			else
				tderror = -clamp;
		}
		pred_mat.SetGradient(i, tderror);
	}
	G.Backward(); // compute gradients on net params
	
	// update net
	UpdateMat(Get(net.W1), alpha);
	UpdateMat(Get(net.b1), alpha);
	UpdateMat(Get(net.W2), alpha);
	UpdateMat(Get(net.b2), alpha);
	
	ClearTempMat();
	
}

}
