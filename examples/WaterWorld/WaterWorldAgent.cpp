#include "WaterWorld.h"


WaterWorldAgent::WaterWorldAgent() {
	nflot = 1000;
	iter = 0;
	
	// positional information
	p.x = 300;
	p.y = 300;
	v.x = 0;
	v.y = 0;
	op = p;
	
	actions.Add(ACT_LEFT);
	actions.Add(ACT_RIGHT);
	actions.Add(ACT_UP);
	actions.Add(ACT_DOWN);
	
	// properties
	rad = 10;
	for (int k = 0; k < 30; k++) {
		eyes.Add().Init(k*0.21);
	}
	
	digestion_signal = 0.0;
	
	// outputs on world
	action = 0;

	smooth_reward = 0.0;
	do_training = true;
}

void WaterWorldAgent::Forward() {
	// in forward pass the agent simply behaves in the environment
	// create input to brain
	int num_eyes = eyes.GetCount();
	int ne = num_eyes * 5;
	Vector<double> input_array;
	input_array.SetCount(num_eyes * 5 + 2, 0);
	for (int i = 0; i < num_eyes; i++) {
		Eye& e = eyes[i];
		input_array[i*5] = 1.0;
		input_array[i*5+1] = 1.0;
		input_array[i*5+2] = 1.0;
		input_array[i*5+3] = e.vx; // velocity information of the sensed target
		input_array[i*5+4] = e.vy;
		if(e.sensed_type != -1) {
			// sensed_type is 0 for wall, 1 for food and 2 for poison.
			// lets do a 1-of-k encoding into the input array
			input_array[i*5 + e.sensed_type] = e.sensed_proximity/e.max_range; // normalize to [0,1]
		}
	}
	
	// proprioception and orientation
    input_array[ne+0] = v.x;
    input_array[ne+1] = v.y;

    action = actions[Act(input_array)];
}

void WaterWorldAgent::Backward() {
	reward = digestion_signal;
	
	// pass to brain for learning
	if (do_training)
		DQNAgent::Learn(reward);
	
	smooth_reward += reward;
	
	if (iter % 50 == 0) {
		while (smooth_reward_history.GetCount() >= nflot) {
			smooth_reward_history.Remove(0);
		}
		smooth_reward_history.Add(smooth_reward);
		
		world->reward.SetLimit(nflot);
		world->reward.AddValue(smooth_reward);
		
		iter = 0;
	}
	iter++;
}

void WaterWorldAgent::Reset() {
	DQNAgent::Reset();
	
	rad = 20;
	action = 0;
	
	smooth_reward = 0.0;
	reward = 0;
}

