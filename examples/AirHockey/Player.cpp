#include "AirHockey.h"

namespace GameCtrl {
using namespace Upp;

Player::Player() {
	nflot = 1000;
	iter = 0;
	
	// positional information
	actions.Add(ACT_LEFT);
	actions.Add(ACT_RIGHT);
	actions.Add(ACT_UP);
	actions.Add(ACT_DOWN);
	
	// properties
	for (int k = 0; k < 30; k++) {
		eyes.Add().Init(k*0.21);
	}
	
	// outputs on world
	action = 0;

	smooth_reward = 0.0;
	do_training = true;
	
	id = -1;
	puck = NULL;
	paint_eyes = true;
}

void Player::Paint(WorldDraw& wdraw, Draw& draw) {
	Color fill_color = Blue();
	Color border_color = Black();
	
	if (paint_eyes) {
		
		// draw agents sight
		Color player_clr(150,150,255);
		Color puck_clr(255,150,150);
		
		double aspect = wdraw.GetAspect();
		double radius = GetRadius();
		Pointf center = GetPosition();
		int r = int(aspect * radius * 2.0);
		Point p = wdraw.ToScreen(center.x , center.y);
		
		for(int j = 0; j < eyes.GetCount(); j++) {
			Eye& e = eyes[j];
			double sr = e.sensed_proximity;
			Color line_clr;
			if(e.sensed_type == -1 || e.sensed_type == 0)
				line_clr = Black(); // wall or nothing
			else if (e.sensed_type == 1)
				line_clr = player_clr; // players
			else if(e.sensed_type == 2)
				line_clr = puck_clr; // puck
			double angle = e.angle;
			Pointf b(
				center.x + sr * sin(angle),
				center.y + sr * cos(angle));
			Point p2 = wdraw.ToScreen(b.x, b.y);
			
			draw.DrawLine(p, p2, 1, line_clr);
		}
	}
	
	PaintCircle(wdraw, draw, fill_color, border_color);
}

void Player::Process() {
	Puck& puck = *this->puck;
	
	Pointf puck_pos = puck.GetPosition();
	Pointf puck_speed = puck.GetSpeed();
	Pointf pos = GetPosition();
	
	Forward();
	
	Pointf force(0,0);
	double force_value = 1000.0;
	if (action == ACT_LEFT) {
		force.x -= force_value;
	}
	else if (action == ACT_UP) {
		force.y += force_value;
	}
	else if (action == ACT_RIGHT) {
		force.x += force_value;
	}
	else if (action == ACT_DOWN) {
		force.y -= force_value;
	}
	
	ApplyForceToCenter(force);
	
}

void Player::Forward() {
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
			double d = e.sensed_proximity/e.max_range;
			input_array[i*5 + e.sensed_type] = d; // normalize to [0,1]
		}
	}
	
	// proprioception and orientation
	Pointf v = GetSpeed();
    input_array[ne+0] = v.x;
    input_array[ne+1] = v.y;

    action = actions[Act(input_array)];
}

void Player::Backward() {
	double reward = game_score;
	game_score = 0;
	
	// pass to brain for learning
	if (do_training)
		DQNAgent::Learn(reward);
	
	smooth_reward += reward;
	
	if (iter % 50 == 0) {
		while (smooth_reward_history.GetCount() >= nflot) {
			smooth_reward_history.Remove(0);
		}
		smooth_reward_history.Add(smooth_reward);
		
		world->AddReward(id, smooth_reward);
		
		iter = 0;
	}
	iter++;
}

void Player::Reset() {
	DQNAgent::Reset();
	
	action = 0;
	
	smooth_reward = 0.0;
	reward = 0;
	game_score = 0;
}

}
