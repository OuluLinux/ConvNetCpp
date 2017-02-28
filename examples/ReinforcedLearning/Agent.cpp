#include "ReinforcedLearning.h"

Agent::Agent() {
	p.x = 50;
	p.y = 50;
	op = p;
	angle = 0;
	oangle = 0;
	
	actions.Add(Pointf(1,	1));
	actions.Add(Pointf(0.8,	1));
	actions.Add(Pointf(1,	0.8));
	actions.Add(Pointf(0.5,	0));
	actions.Add(Pointf(0,	0.5));
	
	// properties
	rad = 20;
	for (int k = 0; k < 9; k++) {
		eyes.Add().Init((k-3)*0.25);
	}
	
	// braaain
	brain.Init(eyes.GetCount() * 3, actions.GetCount());
	
	reward_bonus = 0.0;
	digestion_signal = 0.0;
	
	// outputs on world
	rot1 = 0.0; // rotation speed of 1st wheel
	rot2 = 0.0; // rotation speed of 2nd wheel
	
	prevactionix = -1;
	simspeed = 2;
	actionix = -1;
}

void Agent::Forward() {
	// in forward pass the agent simply behaves in the environment
	// create input to brain
	int num_eyes = eyes.GetCount();
	Vector<double> input_array;
	input_array.SetCount(num_eyes * 3, 0);
	for (int i=0; i < num_eyes; i++) {
		Eye& e = eyes[i];
		input_array[i*3] = 1.0;
		input_array[i*3+1] = 1.0;
		input_array[i*3+2] = 1.0;
		if(e.sensed_type != -1) {
			// sensed_type is 0 for wall, 1 for food and 2 for poison.
			// lets do a 1-of-k encoding into the input array
			input_array[i*3 + e.sensed_type] = e.sensed_proximity/e.max_range; // normalize to [0,1]
		}
	}
	
	// get action from brain
	actionix = brain.Forward(input_array);
	Pointf action = actions[actionix];
	
	// demultiplex into behavior variables
	rot1 = action.x;
	rot2 = action.y;
	
}

void Agent::Backward() {
	// in backward pass agent learns.
	// compute reward
	double proximity_reward = 0.0;
	int num_eyes = eyes.GetCount();
	for(int i=0; i < num_eyes; i++) {
		Eye& e = eyes[i];
		// agents dont like to see walls, especially up close
		proximity_reward += e.sensed_type == 0 ? e.sensed_proximity/e.max_range : 1.0;
	}
	proximity_reward = proximity_reward/num_eyes;
	proximity_reward = min(1.0, proximity_reward * 2.0);
	
	// agents like to go straight forward
	double forward_reward = 0.0;
	if (actionix == 0 && proximity_reward > 0.75)
		forward_reward = 0.1 * proximity_reward;
	
	// agents like to eat good things
	double digestion_reward = digestion_signal;
	digestion_signal = 0.0;
	
	double reward = proximity_reward + forward_reward + digestion_reward;
	
	// pass to brain for learning
	brain.Backward(reward);
}
