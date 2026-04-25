#include "CategorySettingsDialog.h"

CategorySettingsDialog::CategorySettingsDialog(Category& cat, bool _is_new) 
	: category(cat), is_new(_is_new) 
{
	Title(is_new ? "Create Category" : "Edit Category");
	SetRect(0, 0, 450, 500);

	Add(lbl_name.SetLabel("Name:").LeftPos(10, 100).TopPos(10, 20));
	Add(edit_name.HSizePos(10, 10).TopPos(30, 20));
	edit_name.SetData(category.name);

	Add(lbl_super.SetLabel("Supercategory:").LeftPos(10, 150).TopPos(60, 20));
	Add(edit_super.HSizePos(10, 130).TopPos(80, 20));
	edit_super.SetData(category.supercategory);

	Add(lbl_color.SetLabel("Color:").RightPos(10, 100).TopPos(60, 20));
	Add(cp_color.RightPos(10, 100).TopPos(80, 20));
	cp_color.SetData(category.color);

	Add(lbl_keypoints.SetLabel("Keypoints Definition:").LeftPos(10, 200).TopPos(110, 20));
	Add(list_keypoints.HSizePos(10, 10).VSizePos(135, 80));
	list_keypoints.AddColumn("Label").Edit(edit_kp_label);
	list_keypoints.AddColumn("Connects To").Edit(edit_kp_connects);
	list_keypoints.Appending().Removing();
	
	for(int i = 0; i < category.keypoint_labels.GetCount(); i++)
		list_keypoints.Add(category.keypoint_labels[i], category.keypoint_connects_to[i]);

	Add(btn_add_kp.SetLabel("Add Keypoint").LeftPos(10, 120).BottomPos(45, 25));
	Add(btn_remove_kp.SetLabel("Remove selected").LeftPos(140, 120).BottomPos(45, 25));
	
	Add(btn_ok.SetLabel("OK").RightPos(100, 80).BottomPos(10, 25));
	Add(btn_cancel.SetLabel("Cancel").RightPos(10, 80).BottomPos(10, 25));

	btn_add_kp << [=] { list_keypoints.Add("new_keypoint", ""); };
	btn_remove_kp << [=] { if(list_keypoints.IsCursor()) list_keypoints.DoRemove(); };
	
	btn_ok << THISBACK(OnOK);
	btn_cancel << THISBACK(OnCancel);
}

void CategorySettingsDialog::OnOK() {
	String name = ~edit_name;
	if(name.IsEmpty()) {
		Exclamation("Category name cannot be empty");
		return;
	}
	
	category.name = name;
	category.supercategory = ~edit_super;
	category.color = ~cp_color;
	
	category.keypoint_labels.Clear();
	category.keypoint_connects_to.Clear();
	for(int i = 0; i < list_keypoints.GetCount(); i++) {
		category.keypoint_labels.Add(list_keypoints.Get(i, 0).ToString());
		category.keypoint_connects_to.Add(list_keypoints.Get(i, 1).ToString());
	}
	
	Break(IDOK);
}

void CategorySettingsDialog::OnCancel() {
	Break(IDCANCEL);
}
