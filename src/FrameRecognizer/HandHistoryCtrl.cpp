#include "HandHistoryCtrl.h"

NAMESPACE_UPP

HandHistoryCtrl::HandHistoryCtrl()
{
	list_.AddColumn("Time", 60);
	list_.AddColumn("Dealer", 52);
	list_.AddColumn("Board", 110);
	list_.AddColumn("Hero", 64);
	list_.AddColumn("Action", 64);
	list_.AddColumn("Conf", 44);
	list_.EvenRowColor();
	Add(list_.SizePos());
}

void HandHistoryCtrl::AddEntry(const HandHistoryEntry& e)
{
	list_.Add(e.timestamp, e.dealer, e.board, e.hero_cards, e.action, Format("%.2f", e.confidence));
	if(list_.GetCount() > 0)
		list_.SetCursor(list_.GetCount() - 1);
}

void HandHistoryCtrl::Clear()
{
	list_.Clear();
}

END_UPP_NAMESPACE
