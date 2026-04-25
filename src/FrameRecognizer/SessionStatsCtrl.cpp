#include "SessionStatsCtrl.h"

NAMESPACE_UPP

SessionStatsCtrl::SessionStatsCtrl()
{
	Add(lbl_start_.LeftPos(8, 420).TopPos(8, 20));
	Add(lbl_frames_.LeftPos(8, 420).TopPos(32, 20));
	Add(lbl_hands_.LeftPos(8, 420).TopPos(56, 20));
	Add(lbl_recs_.LeftPos(8, 420).TopPos(80, 20));
	Add(lbl_errors_.LeftPos(8, 420).TopPos(104, 20));
	Add(lbl_conf_.LeftPos(8, 420).TopPos(128, 20));
	Reset();
}

void SessionStatsCtrl::Update(const SessionStats& s)
{
	lbl_start_.SetLabel("Session start: " + s.session_start);
	lbl_frames_.SetLabel("Frames processed: " + AsString(s.frames_processed));
	lbl_hands_.SetLabel("Hands detected: " + AsString(s.hands_detected));
	lbl_recs_.SetLabel("Recommendations: " + AsString(s.recommendations));
	lbl_errors_.SetLabel("Engine errors: " + AsString(s.engine_errors));
	lbl_conf_.SetLabel(Format("Avg confidence: %.3f", s.avg_confidence));
}

void SessionStatsCtrl::Reset()
{
	SessionStats s;
	Time t = GetSysTime();
	s.session_start = Format("%04d-%02d-%02d %02d:%02d:%02d", t.year, t.month, t.day, t.hour, t.minute, t.second);
	Update(s);
}

END_UPP_NAMESPACE
