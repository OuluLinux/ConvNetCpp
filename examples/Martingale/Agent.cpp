#include "ReinforcedLearning.h"

RLAgent::RLAgent() {
	pos = 0;
	failcount = 0;
	multiplier = 1;
	prevdir = 0;
	collected_reward = 0;
	doublelen = 0;
	
	posv.SetCount(max_martingale, 0);
	negv.SetCount(max_martingale, 0);
	
	// braaain
	brain.Init(input_count, ACT_COUNT, NULL, 100000, 3000);
	
	reward_bonus = 0.0;
	digestion_signal = 0.0;
	
	simspeed = 2;
	actionix = -1;
}

void RLAgent::Forward() {
	// in forward pass the agent simply behaves in the environment
	// create input to brain
	Vector<double> input_array;
	input_array.SetCount(input_count, 0);
	
	Vector<double>& buf = world->buffer;
	double next = buf[pos];
	for(int i = 0; i < input_length; i++) {
		int j = max(0, pos - i - 1);
		double cur = buf[j];
		double diff = next - cur;
		input_array[i] = diff;
		next = cur;
	}
	for(int i = 0; i < max_martingale; i++) {
		input_array[input_length + i] = posv[i];
		input_array[input_length + max_martingale + i] = negv[i];
	}
	input_array[input_array.GetCount() - 1] = collected_reward > 0 ? 1.0 : 0.0;
	
	// get action from brain
	actionix = brain.Forward(input_array);
	
	pos++;
	if (pos >= buf.GetCount()) pos = 0;
}

void RLAgent::Backward() {
	if (!pos) return;
	
	double cur = world->buffer[pos];
	double prev = world->buffer[pos-1];
	double diff = cur - prev;
	
	
	// in backward pass agent learns.
	// compute reward
	double reward;
	
	
	if (doublelen >= max_martingale)
		actionix = ACT_COLLECT;
	switch (actionix) {
		
	case ACT_DOUBLE:
		reward = 0;
		collected_reward += diff * multiplier;
		if (diff < 0) {
			negv[doublelen] = 1.0;
		} else {
			posv[doublelen] = 1.0;
		}
		multiplier *= 2;
		doublelen++;
		break;
	
	case ACT_COLLECT:
		multiplier = 1;
		failcount = 0;
		reward = collected_reward;
		collected_reward = 0;
		doublelen = 0;
		for(int i = 0; i < posv.GetCount(); i++) {
			posv[i] = 0;
			negv[i] = 0;
		}
	}
	
	// pass to brain for learning
	brain.Backward(reward);
}
