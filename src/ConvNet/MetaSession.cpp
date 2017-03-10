#include "ConvNet.h"

namespace ConvNet {

MetaSession::MetaSession() {
	iter = 0; // iteration counter, goes from 0 -> num_epochs * num_training_data
	data.Add();
}

MetaSession::~MetaSession() {
	
}

void MetaSession::ClearSessions() {
	session.Clear();
	
	
}

void MetaSession::Store(ValueMap& map) const {
	Panic("TODO");
}

void MetaSession::Load(const ValueMap& map) {
	Panic("TODO");
}

}
