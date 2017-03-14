#ifndef _ConvNet_Agent_h_
#define _ConvNet_Agent_h_

namespace ConvNet {


// return Mat but filled with random numbers from gaussian
inline Volume RandVolume(int n, int d, double mu, double std) {
	//var m = new Mat(n, d);
	//fillRandn(m,mu,std);
	////fillRand(m,-std,std); // kind of :P
	Panic("TODO");
	//return m;
}









inline double SampleWeighted(Vector<double>& p) {
	ASSERT(!p.IsEmpty());
	double r = Randomf();
	double c = 0.0;
	for (int i = 0, n = p.GetCount(); i < n; i++) {
		c += p[i];
		if (c >= r)
			return i;
	}
	Panic("wtf");
}


class AgentEnvironment {
	
	
public:
	AgentEnvironment();
	
	Vector<int> AllowedActions(int s);
	double Reward(int s, int a, int ns);
	int NextStateDistribution(int s, int a);
	int GetNumStates();
	int GetMaxNumActions();
	
};

class Agent {
	
protected:
	AgentEnvironment env;
	
public:
	Agent();
	
};










class DPAgent : public Agent {
	
	Vector<double> V; // state value function
	Vector<double> P; // policy distribution \pi(s,a)
	double gamma; // future reward discount factor
	int ns; // num of states
	int na; // num of actions
	
public:
	DPAgent() {
		gamma = 0.75;
		Reset();
	}
	
	void SetGamma(double d) {gamma = d;}
	
	void Reset() {
		// reset the agent's policy and value function
		ns = env.GetNumStates();
		na = env.GetMaxNumActions();
		V.SetCount(ns, 0);
		P.SetCount(ns * na, 0);
		
		// initialize uniform random policy
		for (int s = 0; s < ns; s++) {
			Vector<int> poss = env.AllowedActions(s);
			for (int i = 0, n = poss.GetCount(); i < n; i++) {
				P[poss[i]*ns+s] = 1.0 / poss.GetCount();
			}
		}
	}
	
	double Act(int s) {
		// behave according to the learned policy
		Vector<int> poss = env.AllowedActions(s);
		Vector<double> ps;
		for (int i = 0, n = poss.GetCount(); i < n; i++) {
			int a = poss[i];
			double prob = P[a*ns+s];
			ps.Add(prob);
		}
		int maxi = SampleWeighted(ps);
		return poss[maxi];
	}
	
	void Learn() {
		// perform a single round of value iteration
		EvaluatePolicy(); // writes V
		UpdatePolicy(); // writes P
	}
	
	void EvaluatePolicy() {
		// perform a synchronous update of the value function
		V.SetCount(ns, 0.0);
		for (int s = 0; s < ns; s++) {
			// integrate over actions in a stochastic policy
			// note that we assume that policy probability mass over allowed actions sums to one
			double v = 0.0;
			Vector<int> poss = env.AllowedActions(s);
			for (int i = 0, n = poss.GetCount(); i < n; i++) {
				int a = poss[i];
				double prob = P[a*ns+s]; // probability of taking action under policy
				if (prob == 0) { continue; } // no contribution, skip for speed
				int ns = env.NextStateDistribution(s, a);
				double rs = env.Reward(s, a, ns); // reward for s->a->ns transition
				v += prob * (rs + gamma * V[ns]);
			}
			V[s] = v;
		}
	}
	
	void UpdatePolicy() {
		// update policy to be greedy w.r.t. learned Value function
		for (int s = 0; s < ns; s++) {
			Vector<int> poss = env.AllowedActions(s);
			
			// compute value of taking each allowed action
			double vmax;
			int nmax;
			Vector<double> vs;
			
			vs.SetCount(poss.GetCount());
			for (int i = 0, n = poss.GetCount(); i < n; i++) {
				int a = poss[i];
				int ns = env.NextStateDistribution(s, a);
				double rs = env.Reward(s, a, ns);
				double v = rs + gamma * V[ns];
				vs[i] = v;
				if (i == 0 || v > vmax) { vmax = v; nmax = 1; }
				else if(v == vmax) { nmax += 1; }
			}
			// update policy smoothly across all argmaxy actions
			for (int i = 0, n = poss.GetCount(); i < n; i++) {
				int a = poss[i];
				P[a*ns+s] = (vs[i] == vmax) ? 1.0 / nmax : 0.0;
			}
		}
	}
};














class TDAgent : public Agent {
	
	enum {UPDATE_QLEARN, UPDATE_SARSA};
	
	Vector<double> Q;	// state action value function
	Vector<double> P;	// policy distribution \pi(s,a)
	Vector<double> e;	// eligibility trace
	Vector<double> pq;
	Vector<double> env_model_r;	// environment model (s,a) -> (s',r)
	Vector<int> env_model_s;	// environment model (s,a) -> (s',r)
	Vector<int> sa_seen;
	double gamma;	// future reward discount factor
	double epsilon;	// for epsilon-greedy policy
	double alpha;	// value function learning rate
	double beta;	// learning rate for policy, if smooth updates are on
	double r0;
	int update;		// qlearn | sarsa
	int lambda;		// eligibility trace decay. 0 = no eligibility traces used
	int q_init_val;
	int planN;		// number of planning steps per learning iteration (0 = no planning)
	int s0, s1, a0, a1;
	int ns, na;
	bool smooth_policy_update;
	bool replacing_traces;
	bool explored;
	
public:
	TDAgent() {
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
		
		Reset();
	}
	
	void Reset(){
		// reset the agent's policy and value function
		ns = env.GetNumStates();
		na = env.GetMaxNumActions();
		
		Q.SetCount(0);
		P.SetCount(0);
		e.SetCount(0);
		env_model_s.SetCount(0);
		env_model_r.SetCount(0);
		pq.SetCount(0);
		
		int count = ns * na;
		Q.SetCount(count, q_init_val);
		P.SetCount(count, 0);
		e.SetCount(count, 0);
		env_model_s.SetCount(count, -1);// init to -1 so we can test if we saw the state before
		env_model_r.SetCount(count, 0);
		pq.SetCount(count, 0);
		
		
		// model/planning vars
		sa_seen.SetCount(0);
		
		// initialize uniform random policy
		for (int s = 0; s < ns; s++) {
			Vector<int> poss = env.AllowedActions(s);
			for (int i = 0, n = poss.GetCount(); i < n; i++) {
				P[poss[i]*ns+s] = 1.0 / poss.GetCount();
			}
		}
		
		// agent memory, needed for streaming updates
		// (s0,a0,r0,s1,a1,r1,...)
		r0 = 0;
		s0 = 0;
		s1 = 0;
		a0 = 0;
		a1 = 0;
		
		explored = false;
	}
	
	void ResetEpisode() {
		// an episode finished
	}
	
	double Act(int s){
		// act according to epsilon greedy policy
		Vector<int> poss = env.AllowedActions(s);
		Vector<double> probs;
		for (int i = 0, n = poss.GetCount(); i < n; i++) {
			probs.Add(P[poss[i]*ns+s]);
		}
		
		// epsilon greedy policy
		double a;
		if (Randomf() < epsilon) {
			a = poss[Random(poss.GetCount())]; // random available action
			explored = true;
		} else {
			a = poss[SampleWeighted(probs)];
			explored = false;
		}
		
		// shift state memory
		s0 = s1;
		a0 = a1;
		s1 = s;
		a1 = a;
		return a;
	}
	
	void Learn(double r1){
		// takes reward for previous action, which came from a call to act()
		if (!(r0 == 0.0)) {
			LearnFromTuple(s0, a0, r0, s1, a1, lambda);
			if (planN > 0) {
				UpdateModel(s0, a0, r0, s1);
				Plan();
			}
		}
		r0 = r1; // store this for next update
	}
	
	void UpdateModel(int s0, int a0, double r0, int s1) {
		// transition (s0,a0) -> (r0,s1) was observed. Update environment model
		int sa = a0 * ns + s0;
		if (env_model_s[sa] == -1) {
			// first time we see this state action
			sa_seen.Add(a0 * ns + s0); // add as seen state
		}
		env_model_s[sa] = s1;
		env_model_r[sa] = r0;
	}
	
	void Plan() {
		typedef Tuple2<int, double> Val;
		struct ValSorter {
			bool operator() (const Val& a, const Val& b) const {return a.b < b.b;}
		};
		
		// order the states based on current priority queue information
		Vector<Val> spq;
		for (int i = 0, n = sa_seen.GetCount(); i < n; i++) {
			int sa = sa_seen[i];
			double sap = pq[sa];
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
			int s0a0 = spq[k].a;
			pq[s0a0] = 0; // erase priority, since we're backing up this state
			int s0 = s0a0 % ns;
			int a0 = s0a0 / ns;
			double r0 = env_model_r[s0a0];
			double s1 = env_model_s[s0a0];
			int a1 = -1; // not used for Q learning
			if (update == UPDATE_SARSA) {
				// generate random action?...
				Vector<int> poss = env.AllowedActions(s1);
				a1 = poss[Random(poss.GetCount())];
			}
			LearnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
		}
	}
	
	void LearnFromTuple(int s0, int a0, double r0, int s1, int a1, double lambda) {
		int sa = a0 * ns + s0;
		
		// calculate the target for Q(s,a)
		double target;
		if (update == UPDATE_QLEARN) {
			// Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
			Vector<int> poss = env.AllowedActions(s1);
			double qmax = 0.0;
			for (int i = 0, n = poss.GetCount(); i < n; i++) {
				int s1a = poss[i] * ns + s1;
				double qval = Q[s1a];
				if (i == 0 || qval > qmax) {
					qmax = qval;
				}
			}
			target = r0 + gamma * qmax;
		}
		else if (update == UPDATE_SARSA) {
			// SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
			int s1a1 = a1 * ns + s1;
			target = r0 + gamma * Q[s1a1];
		}
		
		if (lambda > 0.0) {
			// perform an eligibility trace update
			if(replacing_traces) {
				e[sa] = 1;
			}
			else {
				e[sa] += 1;
			}
			double edecay = lambda * gamma;
			
			Vector<double> state_update;
			state_update.SetCount(ns);
			
			for (int s = 0; s < ns; s++) {
				Vector<int> poss = env.AllowedActions(s);
				for (int i = 0; i < poss.GetCount(); i++) {
					int a = poss[i];
					int saloop = a * ns + s;
					double esa = e[saloop];
					double update = alpha * esa * (target - Q[saloop]);
					Q[saloop] += update;
					UpdatePriority(s, a, update);
					e[saloop] *= edecay;
					double u = fabs(update);
					if (u > state_update[s]) {
						state_update[s] = u;
					}
				}
			}
			for (int s = 0; s < ns; s++) {
				if (state_update[s] > 1e-5) { // save efficiency here
					UpdatePolicy(s);
				}
			}
			if (explored && update == UPDATE_QLEARN) {
				// have to wipe the trace since q learning is off-policy :(
				e.SetCount(0);
				e.SetCount(ns * na);
			}
		} else {
			// simpler and faster update without eligibility trace
			// update Q[sa] towards it with some step size
			double update = alpha * (target - Q[sa]);
			Q[sa] += update;
			UpdatePriority(s0, a0, update);
			
			// update the policy to reflect the change (if appropriate)
			UpdatePolicy(s0);
		}
	}
	
	void UpdatePriority(int s, int a, double u) {
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
				int siai = ai * ns + si;
				if (env_model_s[siai] == s) {
					// this state leads to s, add it to priority queue
					pq[siai] += u;
				}
			}
		}
	}
	
	void UpdatePolicy(int s) {
		Vector<int> poss = env.AllowedActions(s);
		
		// set policy at s to be the action that achieves max_a Q(s,a)
		// first find the maxy Q values
		int nmax;
		double qmax;
		Vector<double> qs;
		
		for (int i = 0, n = poss.GetCount(); i < n; i++) {
			int a = poss[i];
			double qval = Q[a*ns+s];
			qs.Add(qval);
			if (i == 0 || qval > qmax) { qmax = qval; nmax = 1; }
			else if(qval == qmax) { nmax += 1; }
		}
		// now update the policy smoothly towards the argmaxy actions
		double psum = 0.0;
		for (int i = 0, n = poss.GetCount(); i < n; i++) {
			int a = poss[i];
			double target = (qs[i] == qmax) ? 1.0/nmax : 0.0;
			int ix = a*ns+s;
			if (smooth_policy_update) {
				// slightly hacky :p
				P[ix] += beta * (target - P[ix]);
				psum += P[ix];
			} else {
				// set hard target
				P[ix] = target;
			}
		}
		if (smooth_policy_update) {
			// renomalize P if we're using smooth policy updates
			for (int i = 0, n = poss.GetCount(); i < n; i++) {
				int a = poss[i];
				P[a*ns+s] /= psum;
			}
		}
	}
};














struct DQNet {
	Volume W1;
	Volume b1;
	Volume W2;
	Volume b2;
};

inline void UpdateMat(Volume& m, double alpha) {
	// updates in place
	//for (int i = 0, n = m.n * m.d; i < n; i++) {
	Panic("chk");
	for (int i = 0, n = m.GetLength(); i < n; i++) {
		double d = m.GetGradient(i);
		if (d != 0) {
			m.Add(i, -alpha * d);
			m.SetGradient(i, 0);
		}
	}
}

inline void UpdateNet(DQNet& net, double alpha) {
	/*for (var p in net) {
		if (net.hasOwnProperty(p)){
			UpdateMat(net[p], alpha);
		}
	}*/
	UpdateMat(net.W1, alpha);
	UpdateMat(net.b1, alpha);
	UpdateMat(net.W2, alpha);
	UpdateMat(net.b2, alpha);
	Panic("TODO");
}

struct DQExperience : Moveable<DQExperience> {
	Volume s0, s1;
	int a0, a1;
	double r0;
	
	void Set(Volume& s0, int a0, double r0, Volume& s1, int a1) {
		this->s0 = s0;
		this->a0 = a0;
		this->r0 = r0;
		this->s1 = s1;
		this->a1 = a1;
	}
};

class DQNAgent : public Agent {
	
	DQNet net;
	Graph G;
	Graph lastG;
	Vector<DQExperience> exp; // experience
	double gamma, epsilon, alpha, tderror_clamp;
	int experience_add_every, experience_size;
	int learning_steps_per_iteration;
	int num_hidden_units;
	int expi;
	int nh;
	int ns;
	int na;
	int t;
	int tderror;
	
	Volume s0, s1;
	int a0, a1;
	double r0;
	
public:
	DQNAgent() {
		gamma = 0.75; // future reward discount factor
		epsilon = 0.1; // for epsilon-greedy policy
		alpha = 0.01; // value function learning rate
		
		experience_add_every = 25; // number of time steps before we add another experience to replay memory
		experience_size = 5000; // size of experience replay
		learning_steps_per_iteration = 10;
		tderror_clamp = 1.0;
		
		num_hidden_units = 100;
		
		env = env;
		Reset();
	}
	
	
	void Reset() {
		nh = num_hidden_units; // number of hidden units
		ns = env.GetNumStates();
		na = env.GetMaxNumActions();
		
		// nets are hardcoded for now as key (str) -> Mat
		// not proud of  better solution is to have a whole Net object
		// on top of Mats, but for now sticking with this
		
		net.W1 = RandVolume(nh, ns, 0, 0.01);
		net.b1.Init(nh, 1, 0, 0.01);
		net.W2 = RandVolume(na, nh, 0, 0.01);
		net.b2.Init(na, 1, 0, 0.01);
		
		expi = 0; // where to insert
		
		t = 0;
		
		r0 = 0;
		a0 = 0;
		a1 = 0;
		//s0.Clear();
		//s1.Clear();
		
		tderror = 0; // for visualization only...
	}
	
	void toJSON() {
		// save function
		/*var j = {};
		j.nh = nh;
		j.ns = ns;
		j.na = na;
		j.net = R.netToJSON(net);*/
		Panic("TODO");
	}
	
	void fromJSON(const ValueMap& j) {
		// load function
		/*nh = j.nh;
		ns = j.ns;
		na = j.na;
		net = R.netFromJSON(j.net);*/
		Panic("TODO");
	}
	
	Volume& ForwardQ(DQNet& net, Volume& s, bool needs_backprop) {
		//var G = new Graph(needs_backprop);
		/*
		var a1mat = G.Add(G.Mul(net.W1, s), net.b1);
		var h1mat = G.Tanh(a1mat);
		var a2mat = G.Add(G.Mul(net.W2, h1mat), net.b2);
		lastG = G; // back this up. Kind of hacky isn't it
		*/
		Panic("TODO");
		//return a2mat;
	}
	
	int Act(const Vector<int>& slist) {
		// convert to a Mat column vector
		Volume s(ns, 1);
		Panic("TODO");//s.SetFrom(slist);
		
		
		// epsilon greedy policy
		int a;
		if (Randomf() < epsilon) {
			a = Random(na);
		} else {
			// greedy wrt Q function
			Volume& amat = ForwardQ(net, s, false);
			a = amat.GetMaxColumn(); // returns index of argmax action
		}
		
		// shift state memory
		s0 = s1;
		a0 = a1;
		s1 = s;
		a1 = a;
		
		return a;
	}
	
	void Learn(double r1) {
		// perform an update on Q function
		if (!(r0 == 0.0) && alpha > 0) {
			
			// learn from this tuple to get a sense of how "surprising" it is to the agent
			tderror = LearnFromTuple(s0, a0, r0, s1, a1); // a measure of surprise
			
			// decide if we should keep this experience in the replay
			if (t % experience_add_every == 0) {
				exp[expi].Set(s0, a0, r0, s1, a1);
				expi += 1;
				if (expi > experience_size) { expi = 0; } // roll over when we run out
			}
			t += 1;
			
			// sample some additional experience from replay memory and learn from it
			for (int k = 0; k < learning_steps_per_iteration; k++) {
				int ri = Random(exp.GetCount()); // TODO: priority sweeps?
				DQExperience& e = exp[ri];
				LearnFromTuple(e.s0, e.a0, e.r0, e.s1, e.a1);
			}
		}
		r0 = r1; // store for next update
	}
	
	double LearnFromTuple(Volume& s0, int a0, double r0, Volume& s1, int a1) {
		// want: Q(s,a) = r + gamma * max_a' Q(s',a')
		
		// compute the target Q value
		Volume& tmat = ForwardQ(net, s1, false);
		double qmax = r0 + gamma * tmat.Get(tmat.GetMaxColumn());
		
		// now predict
		Volume& pred = ForwardQ(net, s0, true);
		
		double tderror = pred.Get(a0) - qmax;
		double clamp = tderror_clamp;
		if (fabs(tderror) > clamp) {  // huber loss to robustify
			if (tderror > clamp) tderror = clamp;
			if (tderror < -clamp) tderror = -clamp;
		}
		pred.SetGradient(a0, tderror);
		lastG.Backward(); // compute gradients on net params
		
		// update net
		UpdateNet(net, alpha);
		return tderror;
	}
	
};


}

#endif
