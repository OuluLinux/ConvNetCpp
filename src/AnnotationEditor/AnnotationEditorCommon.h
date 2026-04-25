#ifndef _AnnotationEditor_AnnotationEditorCommon_h_
#define _AnnotationEditor_AnnotationEditorCommon_h_

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <Ctrl/Mlui/MluiCtrls.h>
#include <plugin/jpg/jpg.h>
#include <plugin/png/png.h>
#include "Dataset.h"
#include "CreateDatasetDialog.h"
#include "ObjectSettingsDialog.h"
#include "CategorySettingsDialog.h"
#include "CocoFormat.h"
#include "Command.h"
#include "CopyAnnotationsDialog.h"
#include "PolygonOps.h"
#include "KeypointSettingsDialog.h"
#include "RDP.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include <AnnLayCore/AnchoredSlotRecognizer.h>
#include "AnnLayTrainPanel.h"
#include "SlotTrainerWindow.h"
#include "ProjectManager.h"

NAMESPACE_UPP

template<> void Jsonize(JsonIO& jio, Pointf& p);

template <class T>
bool SaveAsJSON(const T& obj, const String& path, bool pretty = true) {
	try {
		String json = StoreAsJson(obj, pretty);
		return SaveFile(path, json);
	} catch(...) {
		return false;
	}
}

template <class T>
bool LoadFromJSON(T& obj, const String& path) {
	try {
		String json = LoadFile(path);
		if(json.IsEmpty()) return false;
		Value v = ParseJSON(json);
		if(IsNull(v)) return false;
		LoadFromJsonValue(obj, v);
		return true;
	} catch(...) {
		return false;
	}
}

struct ProjectSessionState {
	String project_path;
	int last_dataset_index = -1;
	int last_image_index = -1;
	String last_mlui_script_path;

	void Jsonize(JsonIO& jio) {
		jio("project_path", project_path)
		   ("last_dataset_index", last_dataset_index)
		   ("last_image_index", last_image_index)
		   ("last_mlui_script_path", last_mlui_script_path);
	}
};

bool IsPointInPolygon(Pointf pt, const Vector<Pointf>& poly);

inline double DistSq(Pointf p1, Pointf p2) {
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

bool IsRectangle(const Vector<Pointf>& poly, double tolerance = 0.1);

String MakeIsoTimestamp();

Image LoadImageByExtension(const String& path);

END_UPP_NAMESPACE

#endif
