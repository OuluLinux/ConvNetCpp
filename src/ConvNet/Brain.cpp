#include "ConvNet.h"

namespace ConvNet {
	







Experience::Experience() {
	action0 = -1;
	reward0 = 0;
}

Experience& Experience::operator=(const Experience& src) {
	HeaplessCopy(this->state0, src.state0);
	this->action0 = src.action0;
	this->reward0 = src.reward0;
	HeaplessCopy(this->state1, src.state1);
	return *this;
}

Experience& Experience::Set(const Vector<double>& state0, int action0, double reward0, const Vector<double>& state1) {
	HeaplessCopy(this->state0, state0);
	this->action0 = action0;
	this->reward0 = reward0;
	HeaplessCopy(this->state1, state1);
	return *this;
}

















Brain::Brain() {
	num_states = 0;
	num_actions = 0;
	age = 0;
}

void Brain::Init(int num_states, int num_actions, Vector<double>* random_action_distribution, int learning_steps_total, int random_beginning_steps) {
	this->random_action_distribution.Clear();
	state_window.Clear();
	action_window.Clear();
	reward_window.Clear();
	net_window.Clear();
	experience.Clear();
	last_input_array.Clear();
	
	// in number of time steps, of temporal memory
	// the ACTUAL input to the net will be (x,a) temporal_window times, and followed by current x
	// so to have no information from previous time step going into value function, set to 0.
	temporal_window =  1;
	
	// size of experience replay memory
	experience_size = 30000;
	
	// number of examples in experience replay memory before we begin learning
	start_learn_threshold = floor(min(experience_size*0.1, 1000.0));
	
	// gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
	gamma = 0.8;
	
	// number of steps we will learn for
	this->learning_steps_total = learning_steps_total;
	
	// how many steps of the above to perform only random actions (in the beginning)?
	this->learning_steps_burnin = random_beginning_steps;
	
	// what epsilon value do we bottom out on? 0.0 => purely deterministic policy at end
	epsilon_min = 0.05;
	
	// what epsilon to use at test time? (i.e. when learning is disabled)
	epsilon_test_time = 0.01;
	
	// advanced feature. Sometimes a random action should be biased towards some values
	// for example in flappy bird, we may want to choose to not flap more often
	if (random_action_distribution) {
		
		// this better sum to 1 by the way, and be of length num_actions
		this->random_action_distribution <<= *random_action_distribution;
		
		if (random_action_distribution->GetCount() != num_actions) {
			throw ArgumentException("TROUBLE. random_action_distribution should be same length as num_actions.");
		}
		
		double s = 0.0;
		for(int k=0; k < random_action_distribution->GetCount(); k++) {
			s += (*random_action_distribution)[k];
		}
		if (fabs(s - 1.0) > 0.0001) {
			throw ArgumentException("TROUBLE. random_action_distribution should sum to 1!");
		}
	}
	
	// states that go into neural net to predict optimal action look as
	// x0,a0,x1,a1,x2,a2,...xt
	// this variable controls the size of that temporal window. Actions are
	// encoded as 1-of-k hot vectors
	net_inputs = num_states * temporal_window + num_actions * temporal_window + num_states;
	this->num_states = num_states;
	this->num_actions = num_actions;
	window_size = max(temporal_window, 2); // must be at least 2, but if we want more context even more
	state_window.SetCount(window_size);
	action_window.SetCount(window_size);
	reward_window.SetCount(window_size);
	net_window.SetCount(window_size);
	
	
	// This is just manually same what parsed JSON would make.
	// To understand: check convnetjs example and then check JSON parser in Session.
	{
		Session::Clear();
		
		// this is an advanced usage feature, because size of the input to the network, and number of
		// actions must check out. This is not very pretty Object Oriented programming but I can't see
		// a way out of it :(
		// create a very simple neural net by default
		AddInputLayer(1, 1, net_inputs);
		
		// allow user to specify this via the option, for convenience
		int hidden_layer_sizes = 2;
		for (int k = 0; k < hidden_layer_sizes; k++) {
			double l1_decay_mul = 0.0;
			double l2_decay_mul = 1.0;
			double bias_pref = 0.1;
			AddFullyConnLayer(50, l1_decay_mul, l2_decay_mul, bias_pref);
			AddReluLayer();
		}
		AddFullyConnLayer(num_actions);
		AddRegressionLayer(); // value function output
		
		// and finally we need a Temporal Difference Learning trainer!
		trainer.SetType(TRAINER_SGD);
		trainer.learning_rate = 0.01;
		trainer.momentum = 0.0;
		trainer.batch_size = 64;
		trainer.l2_decay = 0.01;
	}
	
	
	// various housekeeping variables
	age = 0; // incremented every backward()
	forward_passes = 0; // incremented every forward()
	epsilon = 1.0; // controls exploration exploitation tradeoff. Should be annealed over time
	latest_reward = 0;
	
	average_reward_window.Init(1000, 10);
	average_loss_window.Init(1000, 10);
	SetWindowSize(1000, 10);
	
	learning = true;
}

int Brain::GetRandomAction() {
	// a bit of a helper function. It returns a random action
	// we are abstracting this away because in future we may want to
	// do more sophisticated things. For example some actions could be more
	// or less likely at "rest"/default state.
	if(random_action_distribution.IsEmpty()) {
		return Random(num_actions);
	} else {
		// okay, lets do some fancier sampling:
		double p = Randomf();
		double cumprob = 0.0;
		for (int k=0; k < num_actions; k++) {
			cumprob += random_action_distribution[k];
			if (p < cumprob)
				return k;
		}
	}
	NEVER();
	return 0;
}

ActionValue Brain::GetPolicy(const Vector<double>& weights) {
	// compute the value of doing any action in this state
	// and return the argmax action and its value
	ASSERTEXC(weights.GetCount() == net_inputs);
	svol.Set(weights);
	Volume& action_values = net.Forward(svol);
	int maxk = 0;
	double maxval = action_values.Get(0);
	for(int k=1; k < num_actions; k++) {
		double d = action_values.Get(k);
		if (d > maxval) {
			maxk = k;
			maxval = d;
		}
	}
	ActionValue av;
	av.action = maxk;
	av.value = maxval;
	return av;
}

void Brain::GetNetInput(const Vector<double>& xt, Vector<double>& w) {
	// return s = (x,a,x,a,x,a,xt) state vector.
	// It's a concatenation of last window_size (x,a) pairs and current state x
	w.SetCount(0);
	w.Append(xt); // start with current state
	
	// and now go backwards and append states and actions from history temporal_window times
	int n = window_size;
	for (int k = 0; k < temporal_window; k++) {
		
		// state
		w.Append(state_window[n-1-k]);
		
		// action, encoded as 1-of-k indicator vector. We scale it up a bit because
		// we dont want weight regularization to undervalue this information, as it only exists once
		action1ofk.SetCount(0);
		action1ofk.SetCount(num_actions, 0.0);
		action1ofk[action_window[n-1-k]] = 1.0*num_states;
		w.Append(action1ofk);
	}
}

int Brain::Forward(const Vector<double>& input_array) {
	
	// compute forward (behavior) pass given the input neuron signals from body
	forward_passes += 1;
	last_input_array <<= input_array; // back this up
	
	// create network input
	int action;
	net_input.SetCount(0);
	if(forward_passes > temporal_window) {
		
		// we have enough to actually do something reasonable
		GetNetInput(input_array, net_input);
		ASSERT(net_input.GetCount() == net_inputs);
		
		if(learning) {
			// compute epsilon for the epsilon-greedy policy
			epsilon = min(1.0, max(epsilon_min, 1.0 - (double)(age - learning_steps_burnin)/(learning_steps_total - learning_steps_burnin)));
		} else {
			epsilon = epsilon_test_time; // use test-time value
		}
		double rf = Randomf();
		if (rf < epsilon) {
			// choose a random action with epsilon probability
			action = GetRandomAction();
		} else {
			// otherwise use our policy to make decision
			ActionValue maxact = GetPolicy(net_input);
			action = maxact.action;
		}
	} else {
		// pathological case that happens first few iterations
		// before we accumulate window_size inputs
		action = GetRandomAction();
	}
	
	// remember the state and action we took for backward pass
	net_window.Remove(0);
	net_window.Add() <<= net_input;
	state_window.Remove(0);
	state_window.Add() <<= input_array;
	action_window.Remove(0);
	action_window.Add(action);
	
	return action;
}

void Brain::Backward(double reward) {
	latest_reward = reward;
	average_reward_window.Add(reward);
	reward_window.Remove(0);
	reward_window.Add(reward);
	
	if (!learning)
		return;
	
	Enter();
	
	// various book-keeping
	age += 1;
	
	// it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
	// (given that an appropriate number of state measurements already exist, of course)
	if (forward_passes > temporal_window + 1) {
		//Experience e;
		Experience& e = (experience.GetCount() < experience_size) ? experience.Add() : experience[Random(experience_size)];
		int n = window_size;
		HeaplessCopy(e.state0, net_window[n-2]);
		e.action0 = action_window[n-2];
		e.reward0 = reward_window[n-2];
		HeaplessCopy(e.state1, net_window[n-1]);
	}
	
	// learn based on experience, once we have some samples to go on
	// this is where the magic happens...
	if (experience.GetCount() > start_learn_threshold) {
		double avcost = 0.0;
		for(int k = 0; k < trainer.batch_size; k++) {
			int re = Random(experience.GetCount());
			Experience& e = experience[re];
			ASSERTEXC(e.state0.GetCount() == net_inputs);
			x.Set(e.state0);
			ActionValue maxact = GetPolicy(e.state1);
			double r = e.reward0 + gamma * maxact.value;
			if (!IsFin(r)) r = 0;
			trainer.Train(x, e.action0, r);
			double loss = trainer.GetLoss();
			avcost += loss;
		}
		avcost = avcost / trainer.batch_size;
		average_loss_window.Add(avcost);
	}
	
	Leave();
}

String Brain::ToString() const {
	// basic information
	String t = "";
	t << "experience replay size: " << experience.GetCount() << "\n";
	t << "exploration epsilon: " << epsilon << "\n";
	t << "age: " << age << "\n";
	t << "average Q-learning loss: " << average_loss_window.GetAverage() << "\n";
	t << "smooth-ish reward: " << average_reward_window.GetAverage();
	return t;
}

}
