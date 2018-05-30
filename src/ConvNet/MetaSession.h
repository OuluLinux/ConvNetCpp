#ifndef _ConvNet_MetaSession_h_
#define _ConvNet_MetaSession_h_

#include "Session.h"

namespace ConvNet {

class MetaSession {
	
protected:
	Array<Session> session;
	Array<SessionData> data;
	
	Volume tmp_in, tmp_out;
	int iter, total_iter;
	
public:
	typedef MetaSession CLASSNAME;
	MetaSession();
	~MetaSession();
	
	
	void ClearSessions();
	void Step();
	
	
	Array<Session>& GetSessions() {return session;}
	SessionData& GetSessionData(int i) {return data[i];}
	const Session& GetSession(int i) const {return session[i];}
	Session& GetSession(int i) {return session[i];}
	int GetSessionCount() const {return session.GetCount();}
	int GetIteration() const {return iter;}
	int GetIterationsTotal() const {return total_iter;}
	
	
	
	// callback functions
	
	// called during training of a fold periodically
	Callback1<int> WhenStepInterval;
	
	// called when a fold is finished, while evaluating a batch
	Callback WhenFinishFold;
	
	// called when a batch of candidates has finished evaluating
	Callback WhenFinishBatch;

};


}

#endif
