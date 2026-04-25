#include <plugin/png/png.h>
#include <plugin/zip/zip.h>

#include "AnnLib.h"

NAMESPACE_UPP

namespace {

String ResolveRelativePath(const String& base_file, const String& path)
{
	if(path.IsEmpty())
		return String();
	if(IsFullPath(path))
		return path;
	return AppendFileName(GetFileDirectory(base_file), path);
}

String NormalizeProjectPath(const String& path)
{
	if(path.IsEmpty())
		return path;
	return GetFileExt(path).IsEmpty() ? path + ".annlib" : path;
}

bool ReadIntArray(const Value& v, int expected, Vector<int>& out)
{
	if(!v.Is<ValueArray>())
		return false;
	ValueArray a = v;
	if(a.GetCount() < expected)
		return false;
	out.Clear();
	for(int i = 0; i < expected; i++)
		out.Add((int)a[i]);
	return true;
}

String RelativizeToProject(const String& project_path, const String& abs_path)
{
	if(project_path.IsEmpty() || !IsFullPath(abs_path))
		return abs_path;
	String project_dir = GetFileDirectory(project_path);
	if(abs_path.StartsWith(project_dir)) {
		String rel = abs_path.Mid(project_dir.GetCount());
		while(rel.StartsWith("/") || rel.StartsWith("\\"))
			rel = rel.Mid(1);
		if(!rel.IsEmpty())
			return "./" + rel;
	}
	return abs_path;
}

AnnLib DeepCopyAnnLib(const AnnLib& src)
{
	AnnLib out;
	out.SetProjectPath(src.GetProjectPath());
	out.SetDataDir(src.GetDataDir());
	out.SetCreatedAt(src.GetCreatedAt());
	out.SetLabelTaxonomy(clone(src.GetLabelTaxonomy()));
	out.SetVideoSources(clone(src.GetVideoSources()));

	AnnDataset& dst = out.GetDataset();
	const AnnDataset& in = src.GetDataset();
	dst.images = clone(in.images);
	dst.categories = clone(in.categories);
	dst.annotations = clone(in.annotations);
	return out;
}

}

void AnnLib::Clear()
{
	project_path.Clear();
	data_dir.Clear();
	created_at = Format(GetSysTime());
	label_taxonomy = GetInitialAnnotationTaxonomy();
	video_sources.Clear();
	dataset = AnnDataset();
	EnsureInitialTaxonomy();
}

void AnnLib::EnsureInitialTaxonomy()
{
	if(label_taxonomy.IsEmpty())
		label_taxonomy = GetInitialAnnotationTaxonomy();

	if(dataset.categories.IsEmpty()) {
		dataset.categories = GetInitialAnnotationCategories();
		return;
	}

	for(int i = 0; i < label_taxonomy.GetCount(); i++) {
		bool found = false;
		for(int j = 0; j < dataset.categories.GetCount(); j++) {
			if(dataset.categories[j].name == label_taxonomy[i]) {
				found = true;
				break;
			}
		}
		if(!found) {
			AnnCategory& c = dataset.categories.Add();
			c.id = dataset.categories.GetCount();
			c.name = label_taxonomy[i];
			c.supercategory = "gui";
		}
	}
}

String AnnLib::GetResolvedDataDir() const
{
	if(data_dir.IsEmpty()) {
		if(project_path.IsEmpty())
			return String();
		return AppendFileName(GetFileDirectory(project_path), GetFileTitle(project_path) + "_data");
	}
	return ResolveRelativePath(project_path, data_dir);
}

String AnnLib::GetImagesDir() const
{
	String dir = GetResolvedDataDir();
	return dir.IsEmpty() ? String() : AppendFileName(dir, "images");
}

int AnnLib::NextImageId() const
{
	int max_id = 0;
	for(const AnnImage& img : dataset.images)
		max_id = max(max_id, img.id);
	return max_id + 1;
}

int AnnLib::NextAnnotationId() const
{
	int max_id = 0;
	for(const AnnObject& obj : dataset.annotations)
		max_id = max(max_id, obj.id);
	return max_id + 1;
}

bool AnnLib::LoadProjectJson(const String& annlib_path, const Value& root)
{
	if(!root.Is<ValueMap>())
		return false;
	ValueMap m = root;
	project_path = annlib_path;
	data_dir = m["data_dir"];
	created_at = m["created_at"];
	label_taxonomy.Clear();
	if(m["label_taxonomy"].Is<ValueArray>()) {
		ValueArray arr = m["label_taxonomy"];
		for(int i = 0; i < arr.GetCount(); i++)
			label_taxonomy.Add(arr[i].ToString());
	}

	video_sources.Clear();
	if(m["video_sources"].Is<ValueArray>()) {
		ValueArray arr = m["video_sources"];
		for(int i = 0; i < arr.GetCount(); i++) {
			if(!arr[i].Is<ValueMap>())
				continue;
			ValueMap vs = arr[i];
			AnnVideoSource& out = video_sources.Add();
			out.label = vs["label"];
			out.host = vs["host"];
			if(out.host.IsEmpty())
				out.host = "127.0.0.1";
			out.port = IsNull(vs["port"]) ? 18082 : (int)vs["port"];
			Vector<int> rect;
			if(ReadIntArray(vs["window_rect"], 4, rect)) {
				out.window_rect.x = rect[0];
				out.window_rect.y = rect[1];
				out.window_rect.w = rect[2];
				out.window_rect.h = rect[3];
			}
		}
	}

	EnsureInitialTaxonomy();
	String annotations_path = AppendFileName(GetResolvedDataDir(), "annotations.json");
	if(FileExists(annotations_path))
		return LoadAnnotations(annotations_path);
	return true;
}

bool AnnLib::LoadAnnotations(const String& annotations_path)
{
	dataset.images.Clear();
	dataset.annotations.Clear();
	Value root = ParseJSON(LoadFile(annotations_path));
	if(!root.Is<ValueMap>())
		return false;
	ValueMap m = root;

	if(m["categories"].Is<ValueArray>()) {
		dataset.categories.Clear();
		ValueArray arr = m["categories"];
		for(int i = 0; i < arr.GetCount(); i++) {
			if(!arr[i].Is<ValueMap>())
				continue;
			ValueMap c = arr[i];
			AnnCategory& out = dataset.categories.Add();
			out.id = IsNull(c["id"]) ? (i + 1) : (int)c["id"];
			out.name = c["name"];
			out.supercategory = c["supercategory"];
			if(out.supercategory.IsEmpty())
				out.supercategory = "gui";
		}
	}

	if(m["images"].Is<ValueArray>()) {
		ValueArray arr = m["images"];
		for(int i = 0; i < arr.GetCount(); i++) {
			if(!arr[i].Is<ValueMap>())
				continue;
			ValueMap im = arr[i];
			AnnImage& out = dataset.images.Add();
			out.id = IsNull(im["id"]) ? (i + 1) : (int)im["id"];
			out.filename = im["file_name"];
			out.width = IsNull(im["width"]) ? 0 : (int)im["width"];
			out.height = IsNull(im["height"]) ? 0 : (int)im["height"];
		}
	}

	if(m["annotations"].Is<ValueArray>()) {
		ValueArray arr = m["annotations"];
		for(int i = 0; i < arr.GetCount(); i++) {
			if(!arr[i].Is<ValueMap>())
				continue;
			ValueMap a = arr[i];
			AnnObject& out = dataset.annotations.Add();
			out.id = IsNull(a["id"]) ? (i + 1) : (int)a["id"];
			out.image_id = IsNull(a["image_id"]) ? 0 : (int)a["image_id"];
			out.category_id = IsNull(a["category_id"]) ? 0 : (int)a["category_id"];
			out.label = a["label"];
			out.notes = a["notes"];
			Vector<int> bbox;
			if(ReadIntArray(a["bbox"], 4, bbox)) {
				out.bbox.x = bbox[0];
				out.bbox.y = bbox[1];
				out.bbox.w = bbox[2];
				out.bbox.h = bbox[3];
			}
		}
	}

	EnsureInitialTaxonomy();
	return true;
}

bool AnnLib::Load(const String& annlib_path)
{
	Clear();
	String resolved = NormalizeProjectPath(annlib_path);
	if(resolved.IsEmpty() || !FileExists(resolved))
		return false;
	Value root = ParseJSON(LoadFile(resolved));
	return LoadProjectJson(resolved, root);
}

String AnnLib::BuildProjectJson() const
{
	JsonArray labels;
	for(const String& label : label_taxonomy)
		labels << label;

	JsonArray sources;
	for(const AnnVideoSource& src : video_sources) {
		sources << Json("label", src.label)
		              ("host", src.host)
		              ("port", src.port)
		              ("window_rect", JsonArray() << src.window_rect.x << src.window_rect.y
		                                           << src.window_rect.w << src.window_rect.h);
	}

	String resolved_data = GetResolvedDataDir();
	String relative_data = RelativizeToProject(project_path, resolved_data);
	if(relative_data.IsEmpty())
		relative_data = "./" + GetFileTitle(project_path) + "_data";

	return Json("version", 1)
	           ("data_dir", relative_data)
	           ("created_at", created_at.IsEmpty() ? Format(GetSysTime()) : created_at)
	           ("label_taxonomy", labels)
	           ("video_sources", sources);
}

String AnnLib::BuildAnnotationsJson() const
{
	JsonArray images;
	for(const AnnImage& img : dataset.images)
		images << Json("id", img.id)
		            ("file_name", img.filename)
		            ("width", img.width)
		            ("height", img.height);

	JsonArray categories;
	for(const AnnCategory& c : dataset.categories)
		categories << Json("id", c.id)
		                ("name", c.name)
		                ("supercategory", c.supercategory);

	JsonArray annotations;
	for(const AnnObject& obj : dataset.annotations)
		annotations << Json("id", obj.id)
		                 ("image_id", obj.image_id)
		                 ("category_id", obj.category_id)
		                 ("bbox", JsonArray() << obj.bbox.x << obj.bbox.y << obj.bbox.w << obj.bbox.h)
		                 ("label", obj.label)
		                 ("notes", obj.notes);

	return Json("images", images)
	           ("categories", categories)
	           ("annotations", annotations);
}

bool AnnLib::Save(const String& annlib_path) const
{
	String resolved = NormalizeProjectPath(annlib_path);
	if(resolved.IsEmpty())
		return false;

	String project_dir = GetFileDirectory(resolved);
	String data_root = data_dir.IsEmpty()
		? AppendFileName(project_dir, GetFileTitle(resolved) + "_data")
		: ResolveRelativePath(resolved, data_dir);
	String images_dir = AppendFileName(data_root, "images");
	if(!RealizeDirectory(images_dir))
		return false;

	AnnLib tmp = DeepCopyAnnLib(*this);
	tmp.project_path = resolved;
	tmp.data_dir = RelativizeToProject(resolved, data_root);

	if(!SaveFile(AppendFileName(data_root, "annotations.json"), tmp.BuildAnnotationsJson()))
		return false;
	return SaveFile(resolved, tmp.BuildProjectJson());
}

bool AnnLib::AddImage(const Image& img)
{
	if(img.IsEmpty())
		return false;
	String project = project_path;
	if(project.IsEmpty())
		project = AppendFileName(GetCurrentDirectory(), "annotation_project.annlib");

	if(data_dir.IsEmpty())
		data_dir = "./" + GetFileTitle(project) + "_data";
	project_path = project;

	String images_dir = GetImagesDir();
	if(images_dir.IsEmpty() || !RealizeDirectory(images_dir))
		return false;

	String png = PNGEncoder().SaveString(img);
	String sha = SHA256StringS(png);
	String rel_name = "images/" + sha + ".png";
	String abs_path = AppendFileName(GetResolvedDataDir(), rel_name);
	RealizeDirectory(GetFileDirectory(abs_path));
	if(!FileExists(abs_path) && !PNGEncoder().SaveFile(abs_path, img))
		return false;

	for(const AnnImage& it : dataset.images)
		if(it.filename == rel_name)
			return true;

	AnnImage& out = dataset.images.Add();
	out.id = NextImageId();
	out.filename = rel_name;
	out.width = img.GetWidth();
	out.height = img.GetHeight();
	return true;
}

bool AnnLib::AddImageFile(const String& file_path)
{
	Image img = StreamRaster::LoadFileAny(file_path);
	return !img.IsEmpty() && AddImage(img);
}

bool AnnLib::ExportZip(const String& zip_path) const
{
	String resolved = NormalizeProjectPath(project_path);
	if(resolved.IsEmpty())
		resolved = AppendFileName(GetCurrentDirectory(), "annotation_project.annlib");

	AnnLib tmp = DeepCopyAnnLib(*this);
	tmp.project_path = resolved;

	FileZip zip(zip_path);
	if(zip.IsError())
		return false;

	zip.WriteFile(tmp.BuildProjectJson(), GetFileName(resolved), Null, GetSysTime(), true);
	zip.WriteFile(tmp.BuildAnnotationsJson(), "annotations.json", Null, GetSysTime(), true);
	zip.WriteFile(tmp.BuildProjectJson(), "manifest.json", Null, GetSysTime(), true);

	String images_dir = GetImagesDir();
	for(const AnnImage& img : dataset.images) {
		String abs = AppendFileName(GetResolvedDataDir(), img.filename);
		if(FileExists(abs))
			zip.WriteFile(LoadFile(abs), img.filename, Null, GetSysTime(), true);
	}

	zip.Finish();
	return !zip.IsError();
}

bool AnnLib::ExportCOCO(const String& path) const
{
	return SaveFile(path, BuildAnnotationsJson());
}

END_UPP_NAMESPACE
