#include "Dataset.h"

NAMESPACE_UPP

Category::Category(const Category& c) {
	id = c.id;
	name = c.name;
	supercategory = c.supercategory;
	color = c.color;
	keypoint_labels <<= c.keypoint_labels;
	keypoint_connects_to <<= c.keypoint_connects_to;
}

Category& Category::operator=(const Category& c) {
	id = c.id;
	name = c.name;
	supercategory = c.supercategory;
	color = c.color;
	keypoint_labels <<= c.keypoint_labels;
	keypoint_connects_to <<= c.keypoint_connects_to;
	return *this;
}

void Category::Jsonize(JsonIO& jio) {
	jio("id", id)("name", name)("supercategory", supercategory)("color", color)
	   ("keypoint_labels", keypoint_labels)("keypoint_connects_to", keypoint_connects_to);
}

void KeypointInstance::Jsonize(JsonIO& jio) {
	jio("id", id)("label", label)("x", x)("y", y)("visibility_state", visibility_state);
}

void AnnotationObject::UpdateBBox() {
	double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
	bool has_data = false;
	for(const auto& poly : polygons) {
		for(const auto& p : poly) {
			min_x = min(min_x, p.x); max_x = max(max_x, p.x);
			min_y = min(min_y, p.y); max_y = max(max_y, p.y);
			has_data = true;
		}
	}
	for(const auto& kp : keypoints) {
		min_x = min(min_x, kp.x); max_x = max(max_x, kp.x);
		min_y = min(min_y, kp.y); max_y = max(max_y, kp.y);
		has_data = true;
	}
	if(has_data) bbox = Rectf(min_x, min_y, max_x, max_y);
	else bbox = Rectf(0, 0, 0, 0);
}

AnnotationObject::AnnotationObject(const AnnotationObject& o) {
	id = o.id;
	category_id = o.category_id;
	name = o.name;
	slot_id = o.slot_id;
	slot_id = o.slot_id;
	color = o.color;
	visible = o.visible;
	label_visibility_state = o.label_visibility_state;
	polygons <<= o.polygons;
	keypoints <<= o.keypoints;
	metadata <<= o.metadata;
	score = o.score;
	accepted = o.accepted;
	rejected = o.rejected;
	bbox = o.bbox;
}

AnnotationObject& AnnotationObject::operator=(const AnnotationObject& o) {
	id = o.id;
	category_id = o.category_id;
	name = o.name;
	slot_id = o.slot_id;
	slot_id = o.slot_id;
	color = o.color;
	visible = o.visible;
	label_visibility_state = o.label_visibility_state;
	polygons <<= o.polygons;
	keypoints <<= o.keypoints;
	metadata <<= o.metadata;
	score = o.score;
	accepted = o.accepted;
	rejected = o.rejected;
	bbox = o.bbox;
	return *this;
}

void AnnotationObject::BBoxToPolygon(double x, double y, double w, double h) {
	Vector<Pointf> pts;
	pts.Add(Pointf(x, y));     pts.Add(Pointf(x+w, y));
	pts.Add(Pointf(x+w, y+h)); pts.Add(Pointf(x, y+h));
	polygons.Add(pick(pts));
}

void AnnotationObject::Jsonize(JsonIO& jio) {
	jio("id", id)("category_id", category_id)("name", name)("slot_id", slot_id)("color", color)
	   ("visible", visible)("label_visibility_state", label_visibility_state)
	   ("polygons", polygons)("keypoints", keypoints)
	   .Var("metadata", metadata, [](JsonIO& j, VectorMap<String,String>& m){ StringMap(j, m); })
	   ("score", score)("accepted", accepted)("rejected", rejected);
	if(jio.IsLoading()) {
		if(polygons.IsEmpty() && keypoints.IsEmpty()) {
			const Value& bv = jio.Get("bbox");
			if(IsValueArray(bv) && bv.GetCount() == 4) {
				double x = bv[0], y = bv[1], w = bv[2], h = bv[3];
				if(w > 0 && h > 0) BBoxToPolygon(x, y, w, h);
			}
		}
		UpdateBBox();
	}
}

void ImageEntry::SyncLegacyToImageMetadata() {
	image_metadata.GetAdd("table_currency_unit") = table_currency_unit;
	image_metadata.GetAdd("dealer") = AsString(dealer);
	image_metadata.GetAdd("pot_total") = AsString(pot_total);
	image_metadata.GetAdd("round_pot") = AsString(round_pot);
	image_metadata.GetAdd("side_pot") = AsString(side_pot);
	image_metadata.GetAdd("hero_cards") = hero_cards;
	image_metadata.GetAdd("board_cards") = board_cards;
	for(int i = 0; i < player_in_table.GetCount(); i++)
		image_metadata.GetAdd(Format("seat%d_in_table", i + 1)) = player_in_table[i] ? "true" : "false";
	for(int i = 0; i < player_in_game.GetCount(); i++)
		image_metadata.GetAdd(Format("seat%d_in_game", i + 1)) = player_in_game[i] ? "true" : "false";
}

void ImageEntry::SyncImageMetadataToLegacy() {
	auto set_s = [&](const String& key, String& dst) {
		int q = image_metadata.Find(key);
		if(q >= 0) dst = image_metadata[q];
	};
	auto set_i = [&](const String& key, int& dst) {
		int q = image_metadata.Find(key);
		if(q >= 0) dst = ParseIntMeta(image_metadata[q], dst);
	};
	auto set_d = [&](const String& key, double& dst) {
		int q = image_metadata.Find(key);
		if(q >= 0) dst = ParseDoubleMeta(image_metadata[q], dst);
	};
	set_s("table_currency_unit", table_currency_unit);
	set_i("dealer", dealer);
	set_d("pot_total", pot_total);
	set_d("round_pot", round_pot);
	set_d("side_pot", side_pot);
	set_s("hero_cards", hero_cards);
	set_s("board_cards", board_cards);
	for(int i = 0; i < player_in_table.GetCount(); i++) {
		String k = Format("seat%d_in_table", i + 1);
		int q = image_metadata.Find(k);
		if(q >= 0) player_in_table[i] = ParseBoolMeta(image_metadata[q], player_in_table[i] != 0) ? 1 : 0;
	}
	for(int i = 0; i < player_in_game.GetCount(); i++) {
		String k = Format("seat%d_in_game", i + 1);
		int q = image_metadata.Find(k);
		if(q >= 0) player_in_game[i] = ParseBoolMeta(image_metadata[q], player_in_game[i] != 0) ? 1 : 0;
	}
}

void ImageEntry::EnsureMetadataDefaults() {
	const int players = 8;
	Vector<int> in_table;
	in_table.SetCount(players, 1);
	for(int i = 0; i < min(players, player_in_table.GetCount()); i++)
		in_table[i] = player_in_table[i] ? 1 : 0;
	player_in_table = pick(in_table);

	Vector<int> in_game;
	in_game.SetCount(players, 0);
	for(int i = 0; i < min(players, player_in_game.GetCount()); i++)
		in_game[i] = player_in_game[i] ? 1 : 0;
	player_in_game = pick(in_game);

	if(table_currency_unit.IsEmpty())
		table_currency_unit = "BB";
	if(!image_metadata.IsEmpty())
		SyncImageMetadataToLegacy();
	SyncLegacyToImageMetadata();
}

ImageEntry::ImageEntry(const ImageEntry& e) {
	file_path = e.file_path;
	file_name = e.file_name;
	width = e.width;
	height = e.height;
	has_annotations = e.has_annotations;
	reviewed = e.reviewed;
	priority_score = e.priority_score;
	annotations <<= e.annotations;
	suggestions <<= e.suggestions;
	rejected_suggestions <<= e.rejected_suggestions;
	mlui_scripts <<= e.mlui_scripts;
	table_currency_unit = e.table_currency_unit;
	dealer = e.dealer;
	pot_total = e.pot_total;
	round_pot = e.round_pot;
	side_pot = e.side_pot;
	player_in_table <<= e.player_in_table;
	player_in_game <<= e.player_in_game;
	hero_cards = e.hero_cards;
	board_cards = e.board_cards;
	image_metadata <<= e.image_metadata;
	EnsureMetadataDefaults();
}

ImageEntry& ImageEntry::operator=(const ImageEntry& e) {
	file_path = e.file_path;
	file_name = e.file_name;
	width = e.width;
	height = e.height;
	has_annotations = e.has_annotations;
	reviewed = e.reviewed;
	priority_score = e.priority_score;
	annotations <<= e.annotations;
	suggestions <<= e.suggestions;
	rejected_suggestions <<= e.rejected_suggestions;
	mlui_scripts <<= e.mlui_scripts;
	table_currency_unit = e.table_currency_unit;
	dealer = e.dealer;
	pot_total = e.pot_total;
	round_pot = e.round_pot;
	side_pot = e.side_pot;
	player_in_table <<= e.player_in_table;
	player_in_game <<= e.player_in_game;
	hero_cards = e.hero_cards;
	board_cards = e.board_cards;
	image_metadata <<= e.image_metadata;
	EnsureMetadataDefaults();
	return *this;
}

void ImageEntry::Jsonize(JsonIO& jio) {
	Vector<String> image_meta_keys;
	Vector<String> image_meta_values;
	if(jio.IsStoring()) {
		image_meta_keys <<= image_metadata.GetKeys();
		image_meta_values <<= image_metadata.GetValues();
	}
	jio("file_path", file_path)("file_name", file_name)("width", width)("height", height)
	   ("has_annotations", has_annotations)("reviewed", reviewed)("priority_score", priority_score)
	   ("annotations", annotations)("suggestions", suggestions)
	   ("rejected_suggestions", rejected_suggestions)
	   ("mlui_scripts", mlui_scripts)
	   ("table_currency_unit", table_currency_unit)
	   ("dealer", dealer)
	   ("pot_total", pot_total)
	   ("round_pot", round_pot)
	   ("side_pot", side_pot)
	   ("player_in_table", player_in_table)
	   ("player_in_game", player_in_game)
	   ("hero_cards", hero_cards)
	   ("board_cards", board_cards)
	   ("image_meta_keys", image_meta_keys)
	   ("image_meta_values", image_meta_values);
	if(jio.IsLoading()) {
		image_metadata.Clear();
		for(int i = 0; i < min(image_meta_keys.GetCount(), image_meta_values.GetCount()); i++) {
			if(!image_meta_keys[i].IsEmpty())
				image_metadata.Add(image_meta_keys[i], image_meta_values[i]);
		}
		EnsureMetadataDefaults();
	}
}

Dataset::Dataset(const Dataset& d) {
	name = d.name;
	folder_path = d.folder_path;
	default_categories <<= d.default_categories;
	created_by = d.created_by;
	image_count = d.image_count;
	images <<= d.images;
	mlui_script_path = d.mlui_script_path;
}

Dataset& Dataset::operator=(const Dataset& d) {
	name = d.name;
	folder_path = d.folder_path;
	default_categories <<= d.default_categories;
	created_by = d.created_by;
	image_count = d.image_count;
	images <<= d.images;
	mlui_script_path = d.mlui_script_path;
	return *this;
}

void Dataset::Jsonize(JsonIO& jio) {
	jio("name", name)("folder_path", folder_path)("default_categories", default_categories)
	   ("created_by", created_by)("image_count", image_count)("images", images)
	   ("mlui_script_path", mlui_script_path);
}

void Project::Jsonize(JsonIO& jio) {
	jio("format", format)("version", version)("last_saved", last_saved)("is_autosave", is_autosave)
	   ("autosave_source_project_path", autosave_source_project_path)
	   ("categories", categories)("datasets", datasets)
	   ("last_dataset_index", last_dataset_index)("last_image_index", last_image_index)
	   ("last_mlui_script_path", last_mlui_script_path);
}

END_UPP_NAMESPACE
