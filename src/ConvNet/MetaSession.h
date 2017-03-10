#ifndef _ConvNet_MetaSession_h_
#define _ConvNet_MetaSession_h_

#include "Session.h"

namespace ConvNet {

class MetaSession {
	
protected:
	Array<Session> session;
	Array<SessionData> data;
	
	int iter;
	
public:
	typedef MetaSession CLASSNAME;
	MetaSession();
	~MetaSession();
	
	
	void ClearSessions();
	
	virtual void Store(ValueMap& map) const;
	virtual void Load(const ValueMap& map);
	
	Array<Session>& GetSessions() {return session;}
	SessionData& GetSessionData(int i) {return data[i];}
	int GetSessionCount() const {return session.GetCount();}
	const Session& GetSession(int i) const {return session[i];}
	
};


}

#endif
