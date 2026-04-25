#include "SlotThresholdCtrl.h"

#include <algorithm>

NAMESPACE_UPP

namespace {

const int kRowH = 24;
const int kDefaultThreshold = 30;

}

SlotThresholdCtrl::SlotThresholdCtrl()
{
	Add(scroll_.RightPos(0, 16).VSizePos());
	scroll_.WhenScroll = THISBACK(OnScroll);
}

SlotThresholdCtrl::~SlotThresholdCtrl()
{
	ClearRows();
}

void SlotThresholdCtrl::ClearRows()
{
	for(auto& row : rows_) {
		if(row.lbl_name) {
			row.lbl_name->Remove();
			delete row.lbl_name;
		}
		if(row.slider) {
			row.slider->Remove();
			delete row.slider;
		}
		if(row.lbl_val) {
			row.lbl_val->Remove();
			delete row.lbl_val;
		}
	}
	rows_.Clear();
}

void SlotThresholdCtrl::SetSlotNames(const Vector<String>& names)
{
	ClearRows();

	Vector<String> sorted = clone(names);
	Sort(sorted);

	for(const String& name : sorted) {
		if(name.IsEmpty())
			continue;
		SlotRow row;
		row.slot_id = name;
		row.lbl_name = new Label();
		row.slider = new SliderCtrl();
		row.lbl_val = new Label();

		row.lbl_name->SetLabel(name);
		row.slider->MinMax(0, 100);
		row.slider->SetData(kDefaultThreshold);
		row.slider->WhenAction = THISBACK(OnAnySlider);
		row.lbl_val->SetLabel("0.30");

		Add(*row.lbl_name);
		Add(*row.slider);
		Add(*row.lbl_val);
		rows_.Add(pick(row));
	}

	OnAnySlider();
	LayoutRows();
}

VectorMap<String, double> SlotThresholdCtrl::GetThresholds() const
{
	VectorMap<String, double> out;
	for(const auto& row : rows_) {
		if(!row.slider)
			continue;
		int iv = (int)row.slider->GetData();
		if(iv != kDefaultThreshold)
			out.GetAdd(row.slot_id) = iv / 100.0;
	}
	return out;
}

void SlotThresholdCtrl::Jsonize(JsonIO& jo)
{
	ValueMap m;
	if(jo.IsStoring()) {
		for(const auto& row : rows_) {
			if(!row.slider)
				continue;
			m.Add(row.slot_id, (int)row.slider->GetData());
		}
	}
	jo("thresholds", m);
	if(jo.IsLoading()) {
		for(auto& row : rows_) {
			int k = m.Find(row.slot_id);
			if(k >= 0 && row.slider)
				row.slider->SetData(minmax((int)m[k], 0, 100));
		}
		OnAnySlider();
	}
}

void SlotThresholdCtrl::OnAnySlider()
{
	for(auto& row : rows_) {
		if(row.slider && row.lbl_val) {
			double v = ((int)row.slider->GetData()) / 100.0;
			row.lbl_val->SetLabel(Format("%.2f", v));
		}
	}
}

void SlotThresholdCtrl::OnScroll()
{
	LayoutRows();
}

void SlotThresholdCtrl::LayoutRows()
{
	Size sz = GetSize();
	int visible_h = max(0, sz.cy);
	int content_h = rows_.GetCount() * kRowH;
	scroll_.SetTotal(content_h);
	scroll_.SetPage(visible_h);

	int y0 = -scroll_.Get();
	for(int i = 0; i < rows_.GetCount(); i++) {
		int y = y0 + i * kRowH;
		auto& row = rows_[i];
		if(row.lbl_name)
			row.lbl_name->SetRect(4, y + 3, 140, 18);
		if(row.slider)
			row.slider->SetRect(148, y + 2, max(40, sz.cx - 148 - 58 - 18), 20);
		if(row.lbl_val)
			row.lbl_val->SetRect(max(4, sz.cx - 54 - 18), y + 3, 36, 18);
	}
}

void SlotThresholdCtrl::Layout()
{
	scroll_.SetRect(max(0, GetSize().cx - 16), 0, 16, max(0, GetSize().cy));
	LayoutRows();
}

END_UPP_NAMESPACE
