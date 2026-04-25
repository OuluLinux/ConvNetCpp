#ifndef _FrameRecognizer_SlotThresholdCtrl_h_
#define _FrameRecognizer_SlotThresholdCtrl_h_

#include <CtrlLib/CtrlLib.h>

NAMESPACE_UPP

class SlotThresholdCtrl : public ParentCtrl {
public:
	typedef SlotThresholdCtrl CLASSNAME;

	SlotThresholdCtrl();
	virtual ~SlotThresholdCtrl();

	void SetSlotNames(const Vector<String>& names);
	VectorMap<String, double> GetThresholds() const;
	int GetSlotCount() const { return rows_.GetCount(); }
	virtual void Jsonize(JsonIO& jo) override;

private:
	struct SlotRow : Moveable<SlotRow> {
		String slot_id;
		Label* lbl_name = nullptr;
		SliderCtrl* slider = nullptr;
		Label* lbl_val = nullptr;
	};

	ScrollBar scroll_;
	Vector<SlotRow> rows_;

	void ClearRows();
	void OnAnySlider();
	void OnScroll();
	void LayoutRows();

	virtual void Layout() override;
};

END_UPP_NAMESPACE

#endif
