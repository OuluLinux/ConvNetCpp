#ifndef _AnnotationTool_AnnLib_h_
#define _AnnotationTool_AnnLib_h_

#include <Draw/Draw.h>

#include "AnnotationData.h"

NAMESPACE_UPP

class AnnLib {
public:
	bool Load(const String& annlib_path);
	bool Save(const String& annlib_path) const;
	bool AddImage(const Image& img);
	bool AddImageFile(const String& file_path);
	bool ExportZip(const String& zip_path) const;
	bool ExportCOCO(const String& path) const;

	void Clear();
	void SetDataDir(const String& dir)                    { data_dir = dir; }
	void SetProjectPath(const String& path)               { project_path = path; }
	void SetCreatedAt(const String& ts)                   { created_at = ts; }
	void SetLabelTaxonomy(const Vector<String>& values)   { label_taxonomy = clone(values); }
	void SetVideoSources(const Vector<AnnVideoSource>& v) { video_sources = clone(v); }

	const String&              GetProjectPath() const     { return project_path; }
	const String&              GetDataDir() const         { return data_dir; }
	const String&              GetCreatedAt() const       { return created_at; }
	const Vector<String>&      GetLabelTaxonomy() const   { return label_taxonomy; }
	const Vector<AnnVideoSource>& GetVideoSources() const { return video_sources; }
	AnnDataset&                GetDataset()               { return dataset; }
	const AnnDataset&          GetDataset() const         { return dataset; }
	String                     GetResolvedDataDir() const;
	String                     GetImagesDir() const;

private:
	bool   LoadProjectJson(const String& annlib_path, const Value& root);
	bool   LoadAnnotations(const String& annotations_path);
	String BuildProjectJson() const;
	String BuildAnnotationsJson() const;
	void   EnsureInitialTaxonomy();
	int    NextImageId() const;
	int    NextAnnotationId() const;

	String                 project_path;
	String                 data_dir;
	String                 created_at;
	Vector<String>         label_taxonomy;
	Vector<AnnVideoSource> video_sources;
	AnnDataset             dataset;
};

END_UPP_NAMESPACE

#endif
