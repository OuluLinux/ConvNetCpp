#ifndef _ConvNet_Agent_h_
#define _ConvNet_Agent_h_

#include "Recurrent.h"

namespace ConvNet {

enum {ACT_LEFT, ACT_UP, ACT_RIGHT, ACT_DOWN, ACT_IDLE};

class Agent {
	
protected:
	friend class GridWorldCtrl;
	
	Vector<Vector<double> > poldist; // policy distribution per action
	Vector<double> reward;
	Vector<double> value; // state value function
	Vector<bool> disable;
	int width, height, length;
	int action_count;
	int start_state, stop_state;
	int iter_sleep;
	bool running, stopped;
	
public:
	typedef Agent CLASSNAME;
	Agent();
	virtual ~Agent();
	
	virtual void Reset();
	virtual double Reward(int s, int a, int ns);
	virtual void EvaluatePolicy() {Panic("not implemented");}
	virtual void UpdatePolicy() {Panic("not implemented");}
	virtual void Learn() = 0;
	virtual int Act(int x, int y) = 0;
	virtual double GetValue(int x, int y) const {return value[GetPos(x, y)];}
	virtual void Load(const ValueMap& map);
	virtual void Store(ValueMap& map);
	virtual void LoadInit(const ValueMap& map);
	virtual void SampleNextState(int x, int y, int action, int& next_state, double& reward, bool& reset_episode);
	
	void Start();
	void Stop();
	void Run();
	void ValueIteration();
	void Init(int width, int height, int action_count=0);
	void ResetValues();
	void AllowedActions(int x, int y, Vector<int>& actions) const;
	int  Act(int state);
	bool LoadJSON(const String& json);
	bool LoadInitJSON(const String& json);
	bool StoreJSON(String& json);
	
	double GetReward(int s) const {return reward[s];}
	int GetNumStates() {return length;}
	int GetMaxNumActions() {return action_count;}
	int GetPos(int x, int y) const;
	int GetStartState() { return start_state; }
	int GetStopState() { return stop_state; }
	int GetRandomState() { return Random(length); }
	int GetWidth() const {return width;}
	int GetHeight() const {return height;}
	int GetNextStateDistribution(int x, int y, int a);
	void GetXY(int state, int& x, int& y) const;
	bool IsDisabled(int x, int y) const {return disable[GetPos(x,y)];}
	bool IsDisabled(int i) const {return disable[i];}
	bool IsRunning() const {return running;}
	
	void SetStartState(int x, int y) {start_state = GetPos(x,y);}
	void SetStopState(int x, int y) {stop_state = GetPos(x,y);}
	void SetIterationDelay(int ms) {iter_sleep = ms;}
	void SetReward(int x, int y, double reward);
	void SetReward(int s, double reward) {this->reward[s] = reward;}
	void SetDisabled(int x, int y, bool disable=true);
};










class DPAgent : public Agent {
	
protected:
	friend class GridWorldCtrl;
	
	double gamma; // future reward discount factor
	
public:
	DPAgent();
	
	
	void SetGamma(double d) {gamma = d;}
	
	virtual void Reset();
	virtual int Act(int x, int y);
	virtual void Learn();
	virtual void EvaluatePolicy();
	virtual void UpdatePolicy();
	
};








class TDAgent : public Agent {
	
	enum {UPDATE_QLEARN, UPDATE_SARSA};
	
	typedef Tuple2<int, int> ActionState;
	typedef Tuple2<ActionState, double> Val;
	
	Vector<ActionState> sa_seen;
	Vector<Vector<double> > Q;	// state action value function
	Vector<Vector<double> > P;	// policy distribution \pi(s,a)
	Vector<Vector<double> > e;	// eligibility trace
	Vector<Vector<double> > pq;
	Vector<Vector<double> > env_model_r;	// environment model (s,a) -> (s',r)
	Vector<Vector<int> >env_model_s;	// environment model (s,a) -> (s',r)
	Vector<int> nsteps_history;
	double gamma;	// future reward discount factor
	double epsilon;	// for epsilon-greedy policy
	double alpha;	// value function learning rate
	double beta;	// learning rate for policy, if smooth updates are on
	double reward0;
	int update;		// qlearn | sarsa
	int lambda;		// eligibility trace decay. 0 = no eligibility traces used
	int q_init_val;
	int planN;		// number of planning steps per learning iteration (0 = no planning)
	int state0, state1, action0, action1;
	int ns, na;
	int current_state;
	int nsteps_counter;
	int nflot;
	bool smooth_policy_update;
	bool replacing_traces;
	bool explored;
	
public:
	
	TDAgent();
	
	virtual void Reset();
	virtual void Learn();
	virtual int Act(int x, int y);
	virtual double GetValue(int x, int y) const;
	virtual void LoadInit(const ValueMap& map);
	
	double GetEpsilon() const {return epsilon;}
	
	void SetEpsilon(double e) {epsilon = e;}
	
	void ResetEpisode();
	void Learn(double reward1);
	void UpdateModel(int state0, int action0, double reward0, int state1);
	void Plan();
	void LearnFromTuple(int state0, int action0, double reward0, int state1, int action1, double lambda);
	void UpdatePriority(int s, int a, double u);
	void UpdatePolicy(int x, int y);
	
};














struct DQNet {
	Mat W1;
	Mat b1;
	Mat W2;
	Mat b2;
	
	void Load(const ValueMap& map);
	void Store(ValueMap& map);
	void Serialize(Stream& s) {s % W1 % b1 % W2 % b2;}
};



struct DQExperience : Moveable<DQExperience> {
	Mat state0, state1;
	int action0, action1;
	double reward0;
	
	void Set(Mat& state0, int action0, double reward0, Mat& state1, int action1) {
		this->state0 = state0;
		this->action0 = action0;
		this->reward0 = reward0;
		this->state1 = state1;
		this->action1 = action1;
	}
	void Serialize(Stream& s) {s % state0 % state1 % action0 % action1 % reward0;}
};

class DQNAgent : public Agent {
	
	DQNet net;
	Graph G;
	Vector<DQExperience> exp; // experience
	double gamma, epsilon, alpha, tderror_clamp;
	double tderror;
	int experience_add_every, experience_size;
	int learning_steps_per_iteration;
	int num_hidden_units;
	int expi;
	int nh;
	int ns;
	int na;
	int t;
	bool has_reward;
	
	Mat state;
	Mat state0, state1;
	int action0, action1;
	double reward0;
	
public:

	DQNAgent();
	
	virtual void Learn();
	virtual int Act(int x, int y);
	virtual void Load(const ValueMap& map);
	virtual void Store(ValueMap& map);
	virtual void LoadInit(const ValueMap& map);
	virtual void StoreInit(ValueMap& map);
	virtual void Reset();
	
	int GetExperienceWritePointer() const {return expi;}
	double GetTDError() const {return tderror;}
	double GetEpsilon() const {return epsilon;}
	Graph& GetGraph() {return G;}
	int GetExperienceCount() const {return exp.GetCount();}
	void ClearExperience() {exp.Clear();}
	
	void SetEpsilon(double e) {epsilon = e;}
	
	int Act(const Vector<double>& slist);
	void Learn(double reward1);
	double LearnFromTuple(Mat& s0, int a0, double reward0, Mat& s1, int a1);
	
	void Serialize(Stream& s) {
		if (s.IsLoading()) {
			ValueMap map;
			s % map;
			Load(map);
		}
		else if (s.IsStoring()) {
			ValueMap map;
			Store(map);
			s % map;
		}
		s % exp % gamma % epsilon % alpha % tderror_clamp % tderror % expi % t;
	}
};








struct SDQExperienceItem : Moveable<SDQExperienceItem> {
	Mat state;
	int action;
	
	void Set(Mat& state, int action) {
		this->state = state;
		this->action = action;
	}
	void Serialize(Stream& s) {s % state % state;}
};

struct SDQExperience : Moveable<SDQExperience> {
	Vector<SDQExperienceItem> exp; // experience sequence
	double reward;
	
	void operator=(const SDQExperience& se) {
		exp <<= se.exp;
		reward = se.reward;
	}
};


class SDQNAgent : public Agent {
	
	DQNet net;
	Graph G;
	Vector<SDQExperience> exp, tmp_exp; // experience
	double gamma, epsilon, alpha, tderror_clamp;
	double tderror;
	int selected_exp;
	int experience_add_every, experience_size;
	int learning_steps_per_iteration;
	int num_hidden_units;
	int expi;
	int nh;
	int ns;
	int na;
	int t;
	bool has_reward;
	
	/*Mat state;
	Mat state0, state1;
	int action0, action1;
	double reward0;*/
	
public:

	SDQNAgent();
	
	virtual void Learn();
	virtual int Act(int x, int y);
	virtual void Load(const ValueMap& map);
	virtual void Store(ValueMap& map);
	virtual void LoadInit(const ValueMap& map);
	virtual void StoreInit(ValueMap& map);
	virtual void Reset();
	
	int GetExperienceWritePointer() const {return expi;}
	double GetTDError() const {return tderror;}
	Graph& GetGraph() {return G;}
	
	void SetEpsilon(double e) {epsilon = e;}
	void SetSequenceCount(int i);
	
	int Act(const Vector<double>& slist);
	void Learn(int seq_id, double reward);
	double LearnFromTuple(Mat& s0, int a0, double reward0, Mat& s1, int a1);
	void BeginSequence(int i);
	
	void Serialize(Stream& s) {
		if (s.IsLoading()) {
			ValueMap map;
			s % map;
			Load(map);
		}
		else if (s.IsStoring()) {
			ValueMap map;
			Store(map);
			s % map;
		}
	}
};











void	RandMat(int n, int d, double mu, double std, Mat& out);
int		SampleWeighted(Vector<double>& p);
void	UpdateMat(Mat& m, double alpha);
void	UpdateNet(DQNet& net, double alpha);

}

#endif
