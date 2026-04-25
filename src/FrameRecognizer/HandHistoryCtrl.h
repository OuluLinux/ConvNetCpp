#ifndef _FrameRecognizer_HandHistoryCtrl_h_
#define _FrameRecognizer_HandHistoryCtrl_h_

#include <CtrlLib/CtrlLib.h>

NAMESPACE_UPP

struct HandHistoryEntry : Moveable<HandHistoryEntry> {
	String timestamp;
	String dealer;
	String board;
	String hero_cards;
	String action;
	double confidence = 0.0;
};

class HandHistoryCtrl : public ParentCtrl {
public:
	typedef HandHistoryCtrl CLASSNAME;

	HandHistoryCtrl();
	void AddEntry(const HandHistoryEntry& e);
	void Clear();

private:
	ArrayCtrl list_;
};

END_UPP_NAMESPACE

#endif
