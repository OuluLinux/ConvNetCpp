#ifndef _ConvNet_Brain_h_
#define _ConvNet_Brain_h_

#include "Session.h"

namespace ConvNet {

class Experience : Moveable<Experience> {
	
	
public:
	Experience();
	Experience(const Experience& src) {*this = src;}
	Experience& operator=(const Experience& src);
	Experience& Set(const Vector<double>& state0, int action0, double reward0, const Vector<double>& state1);
	
	Vector<double> state0;
    int action0;
    double reward0;
    Vector<double> state1;
    
};

struct ActionValue : Moveable<ActionValue> {
	int action;
	double value;
};

class Brain : public Session {
	Vector<double> random_action_distribution;
	
	int temporal_window;
	int net_inputs;
	int num_states;
	int num_actions;
	int window_size;
	Vector<Vector<double> > state_window;
	Vector<double> action_window;
	Vector<double> reward_window;
	Vector<Vector<double> > net_window;
	
	
	Vector<Experience> experience;
	bool learning;
	int age, forward_passes;
	double epsilon, latest_reward;
	Vector<double> last_input_array;
	Window average_reward_window, average_loss_window;
	
	// Temp vars
	Vector<double> net_input;
	Vector<double> action1ofk;
	
public:
	typedef Brain CLASSNAME;
	Brain();
	
	void Init(int num_states, int num_actions, Vector<double>* random_action_distribution=NULL);
	void Reset() {Init(num_states, num_actions);}
	
	virtual const Vector<double>& GetLastInput() const {return last_input_array;}
	
	int GetRandomAction();
	ActionValue GetPolicy(const Vector<double>& weights);
	void GetNetInput(const Vector<double>& xt, Vector<double>& w);
	int Forward(const Vector<double>& input_array);
	void Backward(double reward);
	
	// TODO: fix ambiguity between Brain::GetAverageReward and Session::GetRewardAverage
	double GetLatestReward() const {return latest_reward;}
	double GetAverageReward() const {return average_reward_window.GetAverage();}
	double GetAverageRewardWindowSize() const {return average_reward_window.GetCount();}
	double GetAverageLoss() const {return average_loss_window.GetAverage();}
	double GetAverageLossWindowSize() const {return average_loss_window.GetCount();}
	int GetExperienceCount() const {return experience.GetCount();}
	double GetEpsilon() const {return epsilon;}
	int GetAge() const {return age;}
	
	virtual double GetLossAverage() const {return average_loss_window.GetAverage();}
	virtual double GetRewardAverage() const {return average_reward_window.GetAverage();}
	
	void SetLearning(bool b) {learning = b;}
	
	int experience_size;
	double start_learn_threshold;
	double gamma;
	int learning_steps_total;
	int learning_steps_burnin;
	double epsilon_min;
	double epsilon_test_time;

	String ToString() const;
	
};

}

#endif
