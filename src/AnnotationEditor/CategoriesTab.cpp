#include "CategoriesTab.h"

NAMESPACE_UPP

CategoryCard::CategoryCard(Category& c, int count) : category(c), usage_count(count) {
	SetFrame(FieldFrame());
	Add(lbl_name.SetLabel(category.name).HSizePos(30, 30).TopPos(5, 25));
	lbl_name.SetFont(StdFont().Bold());
	Add(lbl_stats.SetLabel(Format("%d objects use this category", usage_count)).HSizePos(5, 5).TopPos(30, 20));
	lbl_stats.SetFont(StdFont(9).Italic());
	Add(btn_more.SetLabel("...").RightPos(5, 20).TopPos(5, 20));
	btn_more << [=] {
		MenuBar bar;
		bar.Add("Edit", [=] { WhenEdit(category.id); });
		bar.Add("Delete", [=] { WhenDelete(category.id); });
		bar.Execute();
	};
}

void CategoryCard::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	w.DrawEllipse(5, 8, 15, 15, category.color, 1, Black());
}

CategoriesTab::CategoriesTab() {
	Add(title.SetLabel("Categories").LeftPos(10, 200).TopPos(10, 30));
	title.SetFont(StdFont(20).Bold());
	Add(subtitle.SetLabel("loaded categories").LeftPos(10, 200).TopPos(40, 20));
	subtitle.SetFont(StdFont().Italic());
	Add(btn_create.SetLabel("Create").LeftPos(10, 80).TopPos(70, 25));
	Add(btn_refresh.SetLabel("Refresh").LeftPos(100, 80).TopPos(70, 25));
	Add(scroll_view.HSizePos(10, 10).VSizePos(110, 10));
	scroll_view.SetFrame(FieldFrame());
	scroll_view.Add(grid_area.HSizePos());
	scroll_view.AddFrame(sb);
	sb.WhenScroll = [=] { grid_area.TopPos(-sb, grid_area.GetSize().cy); };
	btn_create << THISBACK(OnCreate);
	btn_refresh << THISBACK(RefreshView);
}

void CategoriesTab::SetCategories(Vector<Category>& c) {
	categories = &c;
	RefreshView();
}

void CategoriesTab::SetDatasets(const Vector<Dataset>& d) {
	datasets = &d;
}

int CategoriesTab::GetUsageCount(int cat_id) {
	int count = 0;
	if(!datasets)
		return 0;
	for(const auto& ds : *datasets)
		for(const auto& img : ds.images)
			for(const auto& obj : img.annotations)
				if(obj.category_id == cat_id)
					count++;
	return count;
}

void CategoriesTab::RefreshView() {
	for(int i = 0; i < cards.GetCount(); i++)
		cards[i].Remove();
	cards.Clear();
	if(!categories)
		return;
	int margin = 8;
	int card_h = 60;
	int view_cx = max(100, scroll_view.GetSize().cx - sb.GetSize().cx);
	int card_w = view_cx - 2 * margin;
	int x = margin;
	int y = margin;
	for(int i = 0; i < categories->GetCount(); i++) {
		auto& element = cards.Add(new CategoryCard((*categories)[i], GetUsageCount((*categories)[i].id)));
		element.WhenEdit = THISBACK(OnEdit);
		element.WhenDelete = THISBACK(OnDelete);
		grid_area.Add(element.LeftPos(x, card_w).TopPos(y, card_h));
		y += card_h + margin;
	}
	int total_h = y + margin;
	grid_area.SetRect(0, 0, view_cx, total_h);
	sb.SetTotal(total_h);
	sb.SetPage(scroll_view.GetSize().cy);
}

void CategoriesTab::OnCreate() {
	if(!categories)
		return;
	Category cat;
	static int next_id = 10;
	cat.id = next_id++;
	cat.color = Color(rand() % 200, rand() % 200, rand() % 200);
	CategorySettingsDialog dlg(cat, true);
	if(dlg.Run() == IDOK) {
		categories->Add(pick(cat));
		RefreshView();
		WhenChange();
		WhenLog("Created category " + categories->Top().name);
	}
}

void CategoriesTab::OnEdit(int id) {
	if(!categories)
		return;
	for(int i = 0; i < categories->GetCount(); i++) {
		if((*categories)[i].id != id)
			continue;
		CategorySettingsDialog dlg((*categories)[i], false);
		if(dlg.Run() == IDOK) {
			RefreshView();
			WhenChange();
			WhenLog("Updated category " + (*categories)[i].name);
		}
		return;
	}
}

void CategoriesTab::OnDelete(int id) {
	if(!categories)
		return;
	int count = GetUsageCount(id);
	if(count > 0) {
		PromptOK("Category is in use (" + AsString(count) + " objects). Delete them first.");
		WhenLog("Category " + AsString(id) + " is in use");
		return;
	}
	if(!PromptOKCancel("Delete category?"))
		return;
	for(int i = 0; i < categories->GetCount(); i++) {
		if((*categories)[i].id != id)
			continue;
		WhenLog("Deleted category " + (*categories)[i].name);
		categories->Remove(i);
		RefreshView();
		WhenChange();
		return;
	}
}

void CategoriesTab::Layout() {
	RefreshView();
}

END_UPP_NAMESPACE
