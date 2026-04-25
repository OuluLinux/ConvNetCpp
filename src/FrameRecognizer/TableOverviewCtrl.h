#ifndef _FrameRecognizer_TableOverviewCtrl_h_
#define _FrameRecognizer_TableOverviewCtrl_h_

#include <CtrlLib/CtrlLib.h>

NAMESPACE_UPP

class TableOverviewCtrl : public Ctrl {
public:
	typedef TableOverviewCtrl CLASSNAME;

	void SetFrame(const Image& frame, const Vector<Rect>& table_rects, const Vector<String>& table_names);

	virtual void Paint(Draw& w) override;
	virtual void Layout() override;

private:
	Image          frame_;
	Vector<Rect>   rects_;
	Vector<String> names_;

	struct PackedItem : Moveable<PackedItem> {
		int   rect_idx = -1;
		Point pos;
	};

	int                pack_scale_ = 0;
	Vector<PackedItem> packed_;

	void Repack(Size widget_sz);
};

END_UPP_NAMESPACE

#endif
