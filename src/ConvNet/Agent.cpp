#include "ConvNet.h"

namespace ConvNet {

AgentEnvironment::AgentEnvironment() {
	
}

Vector<int> AgentEnvironment::AllowedActions(int s) {
	
	return Vector<int>();
}

double AgentEnvironment::Reward(int s, int a, int ns) {
	
	
	return 0;
}

int AgentEnvironment::NextStateDistribution(int s, int a) {
	
	
	return 0;
}

int AgentEnvironment::GetNumStates() {
	
	
	return 0;
}

int AgentEnvironment::GetMaxNumActions() {
	
	
	return 0;
}





Agent::Agent() {
	
}

}
