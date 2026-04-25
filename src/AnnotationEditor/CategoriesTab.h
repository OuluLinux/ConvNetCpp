#ifndef _AnnotationEditor_CategoriesTab_h_
#define _AnnotationEditor_CategoriesTab_h_

#include "AnnotationEditorCommon.h"

NAMESPACE_UPP
struct CategoryCard : public ParentCtrl {
	Category& category; int usage_count; Label lbl_name, lbl_stats; Button btn_more;
	CategoryCard(Category& c, int count);
	virtual void Paint(Draw& w) override;
	Event<int> WhenEdit, WhenDelete;
};

class CategoriesTab : public ParentCtrl {
public:
	typedef CategoriesTab CLASSNAME;
	CategoriesTab();
	void SetCategories(Vector<Category>& c);
	void SetDatasets(const Vector<Dataset>& d);
	int GetUsageCount(int cat_id);
	void RefreshView();
	void OnCreate();
	void OnEdit(int id);
	void OnDelete(int id);
	Event<> WhenChange; Event<String> WhenLog;
private:
	Label title, subtitle; Button btn_create, btn_refresh; ParentCtrl scroll_view, grid_area; ScrollBar sb; Array<CategoryCard> cards; Vector<Category>* categories = nullptr; const Vector<Dataset>* datasets = nullptr;
	virtual void Layout() override;
};

END_UPP_NAMESPACE

#endif
