#ifndef _AnnotationEditor_Command_h_
#define _AnnotationEditor_Command_h_

#include <Core/Core.h>
#include "Dataset.h"

using namespace Upp;

class Command {
public:
	virtual ~Command() {}
	virtual void Do() = 0;
	virtual void Undo() = 0;
	virtual String GetName() const = 0;
};

class CreateObjectCommand : public Command {
	ImageEntry& entry;
	AnnotationObject obj;
public:
	CreateObjectCommand(ImageEntry& e, const AnnotationObject& o);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Create Object " + obj.name; }
};

class DeleteObjectCommand : public Command {
	ImageEntry& entry;
	AnnotationObject obj;
	int index;
public:
	DeleteObjectCommand(ImageEntry& e, int idx);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Delete Object " + obj.name; }
};

class AddPolygonCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	Vector<Pointf> poly;
public:
	AddPolygonCommand(ImageEntry& e, int id, const Vector<Pointf>& p);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Add Polygon"; }
};

class MoveVertexCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	int poly_idx;
	int pt_idx;
	Pointf old_pos;
	Pointf new_pos;
	bool is_rect;
	Vector<Pointf> old_poly;
	Vector<Pointf> new_poly;

public:
	MoveVertexCommand(ImageEntry& e, int id, int pix, int kix, Pointf oldp, Pointf newp, bool rect = false);
	MoveVertexCommand(ImageEntry& e, int id, int pix, const Vector<Pointf>& old_p, const Vector<Pointf>& new_p);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Move Vertex"; }
};

class MoveObjectsCommand : public Command {
	ImageEntry& entry;
	Vector<int> obj_indices;
	Vector<Vector<Vector<Pointf>>> old_polys;
	Vector<Vector<KeypointInstance>> old_kps;
	Vector<Vector<Vector<Pointf>>> new_polys;
	Vector<Vector<KeypointInstance>> new_kps;
public:
	MoveObjectsCommand(ImageEntry& e, const Vector<int>& idxs,
	                   const Vector<Vector<Vector<Pointf>>>& op, const Vector<Vector<KeypointInstance>>& ok,
	                   const Vector<Vector<Vector<Pointf>>>& np, const Vector<Vector<KeypointInstance>>& nk);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Move Objects"; }
};

class DeleteVertexCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	int poly_idx;
	int pt_idx;
	Pointf pos;
public:
	DeleteVertexCommand(ImageEntry& e, int id, int pix, int kix, Pointf p);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Delete Vertex"; }
};

class CopyAnnotationsCommand : public Command {
	ImageEntry& target_entry;
	Vector<AnnotationObject> copied_objects;
	Vector<int> new_ids;
public:
	CopyAnnotationsCommand(ImageEntry& target, const Vector<AnnotationObject>& objs);
	void Do() override;
	void Undo() override;
	String GetName() const override;
};

class BrushEditCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	Vector<Vector<Pointf>> old_polys;
	Vector<Vector<Pointf>> new_polys;
	bool is_eraser;
public:
	BrushEditCommand(ImageEntry& e, int id, const Vector<Vector<Pointf>>& op, const Vector<Vector<Pointf>>& np, bool erase);
	void Do() override;
	void Undo() override;
	String GetName() const override { return is_eraser ? "Eraser Applied" : "Brush Applied"; }
};

class ClearAnnotationsCommand : public Command {
	ImageEntry& entry;
	Vector<AnnotationObject> old_annotations;
	bool old_has_annotations;
public:
	ClearAnnotationsCommand(ImageEntry& e);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Clear All Annotations"; }
};

class AddKeypointCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	KeypointInstance kp;
public:
	AddKeypointCommand(ImageEntry& e, int oid, const KeypointInstance& k);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Add Keypoint " + kp.label; }
};

class DeleteKeypointCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	KeypointInstance kp;
public:
	DeleteKeypointCommand(ImageEntry& e, int oid, const KeypointInstance& k);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Delete Keypoint " + kp.label; }
};

class MoveKeypointCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	int kp_id;
	Pointf old_pos;
	Pointf new_pos;
public:
	MoveKeypointCommand(ImageEntry& e, int oid, int kid, Pointf op, Pointf np);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Move Keypoint"; }
};

class AcceptSuggestionCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	AnnotationObject obj;
public:
	AcceptSuggestionCommand(ImageEntry& e, int oid);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Accept AI Suggestion"; }
};

class RejectSuggestionCommand : public Command {
	ImageEntry& entry;
	int obj_id;
	AnnotationObject obj;
	int original_index = -1;
public:
	RejectSuggestionCommand(ImageEntry& e, int oid);
	void Do() override;
	void Undo() override;
	String GetName() const override { return "Reject AI Suggestion"; }
};

class CommandManager {
	Array<Command> undo_stack;
	Array<Command> redo_stack;
	int max_undo = 500;
public:
	void SetMaxUndo(int n) { max_undo = n; }
	void Execute(Command* cmd);
	bool CanUndo() const { return !undo_stack.IsEmpty(); }
	bool CanRedo() const { return !redo_stack.IsEmpty(); }
	void Undo();
	void Redo();
	void Clear();
	const Array<Command>& GetUndoStack() const { return undo_stack; }
	const Array<Command>& GetRedoStack() const { return redo_stack; }
};

#endif
