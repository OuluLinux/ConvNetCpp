#include "CopyAnnotationsDialog.h"

CopyAnnotationsDialog::CopyAnnotationsDialog(const Dataset& ds, int current_img_idx, const Vector<Category>& cats) 
	: dataset(ds), categories(cats) 
{
	Title("Copy Annotations");
	SetRect(0, 0, 400, 450);

	Add(lbl_target.SetLabel("Target Image:").LeftPos(10, 100).TopPos(10, 20));
	Add(dl_target.HSizePos(10, 10).TopPos(30, 20));
	
	for(int i = 0; i < dataset.images.GetCount(); i++) {
		if(i == current_img_idx) continue;
		dl_target.Add(i, Format("[%d] %s", i, dataset.images[i].file_name));
	}
	if(dl_target.GetCount() > 0) dl_target.SetIndex(0);

	Add(cb_filter.SetLabel("Filter by Category").LeftPos(10, 200).TopPos(60, 20));
	cb_filter << THISBACK(OnFilterToggle);

	Add(list_categories.HSizePos(10, 10).VSizePos(90, 50));
	list_categories.AddColumn("Copy?").Ctrls<Option>();
	list_categories.AddColumn("Category");
	
	for(int i = 0; i < categories.GetCount(); i++) {
		list_categories.Add(true, categories[i].name);
	}
	
	list_categories.Disable();

	Add(btn_copy.SetLabel("Copy").RightPos(100, 80).BottomPos(10, 25));
	Add(btn_cancel.SetLabel("Cancel").RightPos(10, 80).BottomPos(10, 25));

	btn_copy << [=] { Break(IDOK); };
	btn_cancel << [=] { Break(IDCANCEL); };
}

void CopyAnnotationsDialog::OnFilterToggle() {
	if(~cb_filter) list_categories.Enable();
	else list_categories.Disable();
}

int CopyAnnotationsDialog::GetTargetImageIndex() const {
	return dl_target.GetData();
}

Vector<int> CopyAnnotationsDialog::GetSelectedCategoryIds() const {
	Vector<int> ids;
	for(int i = 0; i < list_categories.GetCount(); i++) {
		if(list_categories.Get(i, 0)) {
			ids.Add(categories[i].id);
		}
	}
	return ids;
}
