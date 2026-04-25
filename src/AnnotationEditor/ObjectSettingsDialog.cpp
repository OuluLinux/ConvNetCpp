#include "ObjectSettingsDialog.h"

ObjectSettingsDialog::ObjectSettingsDialog(AnnotationObject& obj, const Vector<Category>& cats) 
	: object(obj), categories(cats) 
{
	Title(Format("Settings for Object %d", obj.id));
	SetRect(0, 0, 450, 500);

	Add(lbl_name.SetLabel("Name:").LeftPos(10, 100).TopPos(10, 20));
	Add(edit_name.HSizePos(10, 10).TopPos(30, 20));
	edit_name.SetData(object.name);

	Add(lbl_category.SetLabel("Category:").LeftPos(10, 100).TopPos(60, 20));
	Add(dl_category.HSizePos(10, 130).TopPos(80, 20));
	for(int i = 0; i < categories.GetCount(); i++)
		dl_category.Add(categories[i].id, categories[i].name);
	int cat_cursor = -1;
	for(int i = 0; i < categories.GetCount(); i++) {
		if(categories[i].id == object.category_id) {
			cat_cursor = i;
			break;
		}
	}
	if(cat_cursor >= 0) dl_category.SetIndex(cat_cursor);
	else if(dl_category.GetCount() > 0) dl_category.SetIndex(0);

	Add(lbl_color.SetLabel("Color:").RightPos(10, 100).TopPos(60, 20));
	Add(cp_color.RightPos(10, 100).TopPos(80, 20));
	cp_color.SetData(object.color);

	Add(lbl_state.SetLabel("Label Visibility State:").LeftPos(10, 200).TopPos(110, 20));
	Add(dl_state.HSizePos(10, 10).TopPos(130, 20));
	dl_state.Add(0, "Not labeled");
	dl_state.Add(1, "Labeled not visible");
	dl_state.Add(2, "Labeled visible");
	dl_state.SetData(object.label_visibility_state);

	Add(lbl_metadata.SetLabel("Metadata:").LeftPos(10, 100).TopPos(165, 20));
	Add(list_metadata.HSizePos(10, 10).VSizePos(190, 80));
	list_metadata.AddColumn("Key").Edit(edit_name); // Reusing edit_name for simple string edit
	list_metadata.AddColumn("Value").Edit(edit_name);
	list_metadata.Appending().Removing();
	
	for(int i = 0; i < object.metadata.GetCount(); i++)
		list_metadata.Add(object.metadata.GetKey(i), object.metadata[i]);

	Add(btn_add_meta.SetLabel("Add Row").LeftPos(10, 80).BottomPos(45, 25));
	Add(btn_remove_meta.SetLabel("Remove Row").LeftPos(100, 100).BottomPos(45, 25));
	
	Add(btn_ok.SetLabel("OK").RightPos(100, 80).BottomPos(10, 25));
	Add(btn_cancel.SetLabel("Cancel").RightPos(10, 80).BottomPos(10, 25));

	btn_add_meta << [=] { list_metadata.Add("", ""); };
	btn_remove_meta << [=] { if(list_metadata.IsCursor()) list_metadata.DoRemove(); };
	
	btn_ok << THISBACK(OnOK);
	btn_cancel << THISBACK(OnCancel);
}

void ObjectSettingsDialog::OnOK() {
	object.name = ~edit_name;
	int cidx = dl_category.GetIndex();
	if(cidx >= 0 && cidx < categories.GetCount())
		object.category_id = categories[cidx].id;
	object.color = ~cp_color;
	object.label_visibility_state = ~dl_state;
	
	object.metadata.Clear();
	for(int i = 0; i < list_metadata.GetCount(); i++)
		object.metadata.Add(list_metadata.Get(i, 0).ToString(),
		                    list_metadata.Get(i, 1).ToString());
	
	Break(IDOK);
}

void ObjectSettingsDialog::OnCancel() {
	Break(IDCANCEL);
}
