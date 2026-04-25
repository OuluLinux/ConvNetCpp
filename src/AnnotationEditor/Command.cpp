#include "Command.h"

CreateObjectCommand::CreateObjectCommand(ImageEntry& e, const AnnotationObject& o) : entry(e) {
	obj = o;
}

void CreateObjectCommand::Do() {
	entry.annotations.Add(obj);
	entry.has_annotations = true;
}

void CreateObjectCommand::Undo() {
	for(int i = 0; i < entry.annotations.GetCount(); i++) {
		if(entry.annotations[i].id == obj.id) {
			entry.annotations.Remove(i);
			break;
		}
	}
	if(entry.annotations.IsEmpty())
		entry.has_annotations = false;
}

DeleteObjectCommand::DeleteObjectCommand(ImageEntry& e, int idx) : entry(e), index(idx) {
	obj = e.annotations[idx];
}

void DeleteObjectCommand::Do() {
	entry.annotations.Remove(index);
	if(entry.annotations.IsEmpty())
		entry.has_annotations = false;
}

void DeleteObjectCommand::Undo() {
	entry.annotations.Insert(index, obj);
	entry.has_annotations = true;
}

AddPolygonCommand::AddPolygonCommand(ImageEntry& e, int id, const Vector<Pointf>& p) : entry(e), obj_id(id) {
	poly <<= p;
}

void AddPolygonCommand::Do() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			o.polygons.Add() <<= poly;
			o.UpdateBBox();
			break;
		}
	}
}

void AddPolygonCommand::Undo() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			if(!o.polygons.IsEmpty())
				o.polygons.Drop();
			o.UpdateBBox();
			break;
		}
	}
}

MoveVertexCommand::MoveVertexCommand(ImageEntry& e, int id, int pix, int kix, Pointf oldp, Pointf newp, bool rect)
	: entry(e), obj_id(id), poly_idx(pix), pt_idx(kix), old_pos(oldp), new_pos(newp), is_rect(rect) {}

MoveVertexCommand::MoveVertexCommand(ImageEntry& e, int id, int pix, const Vector<Pointf>& old_p, const Vector<Pointf>& new_p)
	: entry(e), obj_id(id), poly_idx(pix), pt_idx(-1), is_rect(true) {
	old_poly <<= old_p;
	new_poly <<= new_p;
}

void MoveVertexCommand::Do() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			if(is_rect)
				o.polygons[poly_idx] <<= new_poly;
			else
				o.polygons[poly_idx][pt_idx] = new_pos;
			o.UpdateBBox();
			break;
		}
	}
}

void MoveVertexCommand::Undo() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			if(is_rect)
				o.polygons[poly_idx] <<= old_poly;
			else
				o.polygons[poly_idx][pt_idx] = old_pos;
			o.UpdateBBox();
			break;
		}
	}
}

MoveObjectsCommand::MoveObjectsCommand(ImageEntry& e, const Vector<int>& idxs,
                                       const Vector<Vector<Vector<Pointf>>>& op, const Vector<Vector<KeypointInstance>>& ok,
                                       const Vector<Vector<Vector<Pointf>>>& np, const Vector<Vector<KeypointInstance>>& nk)
	: entry(e) {
	obj_indices <<= idxs;
	old_polys <<= op;
	old_kps <<= ok;
	new_polys <<= np;
	new_kps <<= nk;
}

void MoveObjectsCommand::Do() {
	for(int i = 0; i < obj_indices.GetCount(); i++) {
		int oidx = obj_indices[i];
		entry.annotations[oidx].polygons <<= new_polys[i];
		entry.annotations[oidx].keypoints <<= new_kps[i];
		entry.annotations[oidx].UpdateBBox();
	}
}

void MoveObjectsCommand::Undo() {
	for(int i = 0; i < obj_indices.GetCount(); i++) {
		int oidx = obj_indices[i];
		entry.annotations[oidx].polygons <<= old_polys[i];
		entry.annotations[oidx].keypoints <<= old_kps[i];
		entry.annotations[oidx].UpdateBBox();
	}
}

DeleteVertexCommand::DeleteVertexCommand(ImageEntry& e, int id, int pix, int kix, Pointf p)
	: entry(e), obj_id(id), poly_idx(pix), pt_idx(kix), pos(p) {}

void DeleteVertexCommand::Do() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			o.polygons[poly_idx].Remove(pt_idx);
			o.UpdateBBox();
			break;
		}
	}
}

void DeleteVertexCommand::Undo() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			o.polygons[poly_idx].Insert(pt_idx, pos);
			o.UpdateBBox();
			break;
		}
	}
}

CopyAnnotationsCommand::CopyAnnotationsCommand(ImageEntry& target, const Vector<AnnotationObject>& objs)
	: target_entry(target) {
	for(const auto& o : objs)
		copied_objects.Add(AnnotationObject(o));
}

void CopyAnnotationsCommand::Do() {
	new_ids.Clear();
	for(auto& o : copied_objects) {
		AnnotationObject& added = target_entry.annotations.Add(o);
		added.UpdateBBox();
		new_ids.Add(added.id);
	}
	target_entry.has_annotations = true;
}

void CopyAnnotationsCommand::Undo() {
	for(int id : new_ids) {
		for(int i = 0; i < target_entry.annotations.GetCount(); i++) {
			if(target_entry.annotations[i].id == id) {
				target_entry.annotations.Remove(i);
				break;
			}
		}
	}
	if(target_entry.annotations.IsEmpty())
		target_entry.has_annotations = false;
}

String CopyAnnotationsCommand::GetName() const {
	return Format("Copy %d Annotations", copied_objects.GetCount());
}

BrushEditCommand::BrushEditCommand(ImageEntry& e, int id, const Vector<Vector<Pointf>>& op, const Vector<Vector<Pointf>>& np, bool erase)
	: entry(e), obj_id(id), is_eraser(erase) {
	old_polys <<= op;
	new_polys <<= np;
}

void BrushEditCommand::Do() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			o.polygons <<= new_polys;
			o.UpdateBBox();
			break;
		}
	}
}

void BrushEditCommand::Undo() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			o.polygons <<= old_polys;
			o.UpdateBBox();
			break;
		}
	}
}

ClearAnnotationsCommand::ClearAnnotationsCommand(ImageEntry& e) : entry(e) {
	old_annotations <<= e.annotations;
	old_has_annotations = e.has_annotations;
}

void ClearAnnotationsCommand::Do() {
	entry.annotations.Clear();
	entry.has_annotations = false;
}

void ClearAnnotationsCommand::Undo() {
	entry.annotations <<= old_annotations;
	entry.has_annotations = old_has_annotations;
}

AddKeypointCommand::AddKeypointCommand(ImageEntry& e, int oid, const KeypointInstance& k)
	: entry(e), obj_id(oid), kp(k) {}

void AddKeypointCommand::Do() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			o.keypoints.Add(kp);
			o.UpdateBBox();
			break;
		}
	}
}

void AddKeypointCommand::Undo() {
	for(auto& o : entry.annotations) {
		if(o.id != obj_id)
			continue;
		for(int i = 0; i < o.keypoints.GetCount(); i++) {
			if(o.keypoints[i].id == kp.id) {
				o.keypoints.Remove(i);
				break;
			}
		}
		o.UpdateBBox();
		break;
	}
}

DeleteKeypointCommand::DeleteKeypointCommand(ImageEntry& e, int oid, const KeypointInstance& k)
	: entry(e), obj_id(oid), kp(k) {}

void DeleteKeypointCommand::Do() {
	for(auto& o : entry.annotations) {
		if(o.id != obj_id)
			continue;
		for(int i = 0; i < o.keypoints.GetCount(); i++) {
			if(o.keypoints[i].id == kp.id) {
				o.keypoints.Remove(i);
				break;
			}
		}
		o.UpdateBBox();
		break;
	}
}

void DeleteKeypointCommand::Undo() {
	for(auto& o : entry.annotations) {
		if(o.id == obj_id) {
			o.keypoints.Add(kp);
			o.UpdateBBox();
			break;
		}
	}
}

MoveKeypointCommand::MoveKeypointCommand(ImageEntry& e, int oid, int kid, Pointf op, Pointf np)
	: entry(e), obj_id(oid), kp_id(kid), old_pos(op), new_pos(np) {}

void MoveKeypointCommand::Do() {
	for(auto& o : entry.annotations) {
		if(o.id != obj_id)
			continue;
		for(auto& k : o.keypoints) {
			if(k.id == kp_id) {
				k.x = new_pos.x;
				k.y = new_pos.y;
				break;
			}
		}
		o.UpdateBBox();
		break;
	}
}

void MoveKeypointCommand::Undo() {
	for(auto& o : entry.annotations) {
		if(o.id != obj_id)
			continue;
		for(auto& k : o.keypoints) {
			if(k.id == kp_id) {
				k.x = old_pos.x;
				k.y = old_pos.y;
				break;
			}
		}
		o.UpdateBBox();
		break;
	}
}

AcceptSuggestionCommand::AcceptSuggestionCommand(ImageEntry& e, int oid) : entry(e), obj_id(oid) {}

void AcceptSuggestionCommand::Do() {
	for(int i = 0; i < entry.suggestions.GetCount(); i++) {
		if(entry.suggestions[i].id != obj_id)
			continue;
		obj = entry.suggestions[i];
		obj.accepted = true;
		obj.rejected = false;
		entry.suggestions.Remove(i);
		entry.annotations.Add(obj);
		entry.has_annotations = true;
		break;
	}
}

void AcceptSuggestionCommand::Undo() {
	for(int i = 0; i < entry.annotations.GetCount(); i++) {
		if(entry.annotations[i].id != obj_id)
			continue;
		entry.annotations.Remove(i);
		obj.accepted = false;
		entry.suggestions.Add(obj);
		if(entry.annotations.IsEmpty())
			entry.has_annotations = false;
		break;
	}
}

RejectSuggestionCommand::RejectSuggestionCommand(ImageEntry& e, int oid) : entry(e), obj_id(oid) {}

void RejectSuggestionCommand::Do() {
	for(int i = 0; i < entry.suggestions.GetCount(); i++) {
		if(entry.suggestions[i].id != obj_id)
			continue;
		obj = entry.suggestions[i];
		obj.rejected = true;
		obj.accepted = false;
		original_index = i;
		entry.suggestions.Remove(i);
		entry.rejected_suggestions.Add(obj);
		break;
	}
}

void RejectSuggestionCommand::Undo() {
	for(int i = 0; i < entry.rejected_suggestions.GetCount(); i++) {
		if(entry.rejected_suggestions[i].id != obj_id)
			continue;
		entry.rejected_suggestions.Remove(i);
		obj.rejected = false;
		if(original_index != -1)
			entry.suggestions.Insert(original_index, obj);
		else
			entry.suggestions.Add(obj);
		break;
	}
}

void CommandManager::Execute(Command* cmd) {
	cmd->Do();
	undo_stack.Add(cmd);
	redo_stack.Clear();
	if(undo_stack.GetCount() > max_undo)
		undo_stack.Remove(0);
}

void CommandManager::Undo() {
	if(undo_stack.IsEmpty())
		return;
	Command* cmd = undo_stack.Detach(undo_stack.GetCount() - 1);
	cmd->Undo();
	redo_stack.Add(cmd);
}

void CommandManager::Redo() {
	if(redo_stack.IsEmpty())
		return;
	Command* cmd = redo_stack.Detach(redo_stack.GetCount() - 1);
	cmd->Do();
	undo_stack.Add(cmd);
}

void CommandManager::Clear() {
	undo_stack.Clear();
	redo_stack.Clear();
}
