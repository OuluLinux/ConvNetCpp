#ifndef _FrameRecognizer_SessionStatsCtrl_h_
#define _FrameRecognizer_SessionStatsCtrl_h_

#include <CtrlLib/CtrlLib.h>

NAMESPACE_UPP

struct SessionStats : Moveable<SessionStats> {
	int hands_detected = 0;
	int recommendations = 0;
	int engine_errors = 0;
	int frames_processed = 0;
	double avg_confidence = 0.0;
	String session_start;
};

class SessionStatsCtrl : public ParentCtrl {
public:
	typedef SessionStatsCtrl CLASSNAME;

	SessionStatsCtrl();
	void Update(const SessionStats& s);
	void Reset();

private:
	Label lbl_start_;
	Label lbl_frames_;
	Label lbl_hands_;
	Label lbl_recs_;
	Label lbl_errors_;
	Label lbl_conf_;
};

END_UPP_NAMESPACE

#endif
