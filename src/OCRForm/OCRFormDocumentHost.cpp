#include "OCRForm.h"

NAMESPACE_UPP

namespace {

Rect NormalizeAnnotationRect(const Rect& rect)
{
	int left = min(rect.left, rect.right);
	int top = min(rect.top, rect.bottom);
	int right = max(rect.left, rect.right);
	int bottom = max(rect.top, rect.bottom);

	if(right <= left)
		right = left + 1;
	if(bottom <= top)
		bottom = top + 1;
	return Rect(left, top, right, bottom);
}

Size ResolveCanvasSize(const OcrFormDocument& document, const Image& background)
{
	if(document.layouts.GetCount()) {
		const Size& canvas = document.layouts[0].canvas_size;
		if(canvas.cx > 0 && canvas.cy > 0)
			return canvas;
	}
	if(!IsNull(background))
		return background.GetSize();
	return Size(1280, 720);
}

String ResolveMediaPath(const String& document_path, const String& media_path)
{
	if(media_path.IsEmpty())
		return String();
	if(IsFullPath(media_path))
		return media_path;
	if(document_path.IsEmpty())
		return media_path;
	return AppendFileName(GetFileDirectory(document_path), media_path);
}

String BuildAnnotationPath(const String& prefix, const OcrAnnotation& annotation, int index)
{
	String label = annotation.name;
	if(label.IsEmpty())
		label = annotation.widget_type;
	if(label.IsEmpty())
		label = Format("annotation_%d", index);
	return prefix.IsEmpty() ? label : prefix + "/" + label;
}

String FormatRectSummary(const Rect& rect)
{
	Rect normalized = NormalizeAnnotationRect(rect);
	return Format("%d,%d %dx%d", normalized.left, normalized.top,
	              normalized.GetWidth(), normalized.GetHeight());
}

String BuildIndentLabel(const String& label, int depth)
{
	String out;
	for(int i = 0; i < depth; i++)
		out << "  ";
	out << label;
	return out;
}

bool IsMeaningfulProposal(const OcrAnnotation& current, const OcrAnnotation& proposal)
{
	if(!proposal.children.IsEmpty())
		return true;
	if(!proposal.widget_type.IsEmpty() && proposal.widget_type != current.widget_type)
		return true;
	if(!proposal.name.IsEmpty() && proposal.name != current.name)
		return true;
	if(!proposal.hints.IsEmpty())
		return true;
	if(!proposal.rect.IsEmpty()) {
		Rect cur = NormalizeAnnotationRect(current.rect);
		Rect pr = NormalizeAnnotationRect(proposal.rect);
		if(cur != pr)
			return true;
	}
	return false;
}

double ResolveProposalConfidence(const AnnotationProposal& proposal)
{
	if(proposal.confidence > 0.0)
		return proposal.confidence;
	if(proposal.proposed.confidence > 0.0)
		return proposal.proposed.confidence;
	return 0.0;
}

const char* kAnalysisModeManual = "manual";
const char* kAnalysisModeOnChange = "on_change";
const char* kAnalysisModeContinuous = "continuous";
const char* kAnalysisModeOnSave = "on_save";

String AnalysisModeLabel(const String& mode)
{
	if(mode == kAnalysisModeManual)
		return "manual";
	if(mode == kAnalysisModeContinuous)
		return "continuous";
	if(mode == kAnalysisModeOnSave)
		return "on save";
	return "on frame change";
}

void DrawChangedOverlay(Draw& w, const Rect& r)
{
	Rect rect = r;
	if(rect.IsEmpty())
		return;

	Color tint(255, 234, 128);
	Color border(235, 186, 42);

	for(int y = rect.top; y < rect.bottom; y += 4)
		w.DrawRect(rect.left, y, rect.GetWidth(), 1, tint);
	for(int x = rect.left; x < rect.right; x += 4)
		w.DrawRect(x, rect.top, 1, rect.GetHeight(), tint);

	w.DrawRect(rect.left, rect.top, rect.GetWidth(), 1, border);
	w.DrawRect(rect.left, rect.bottom - 1, rect.GetWidth(), 1, border);
	w.DrawRect(rect.left, rect.top, 1, rect.GetHeight(), border);
	w.DrawRect(rect.right - 1, rect.top, 1, rect.GetHeight(), border);
}

}

OcrFormPreview::OcrFormPreview()
{
	NoWantFocus();
	IgnoreMouse();
}

void OcrFormPreview::SetChangedRegions(Vector<Rect> regions)
{
	changed_regions = pick(regions);
	Refresh();
}

void OcrFormPreview::ClearChangedRegions()
{
	if(changed_regions.IsEmpty())
		return;
	changed_regions.Clear();
	Refresh();
}

void OcrFormPreview::Paint(Draw& w)
{
	FormView::Paint(w);

	if(!IsLayout())
		return;

	for(int i = 0; i < changed_regions.GetCount(); i++) {
		Rect zr = Zoom(Offseted(changed_regions[i]));
		if(zr.GetWidth() < 2 || zr.GetHeight() < 2)
			continue;
		DrawChangedOverlay(w, zr);
	}

	Font font = StdFont();
	for(int i = 0; i < GetObjectCount(); i++) {
		FormObject* object = GetObject(i);
		if(!object)
			continue;

		String label = object->Get("OCR.Label");
		if(label.IsEmpty())
			continue;

		Rect rect = Zoom(Offseted(object->GetRect()));
		if(rect.GetWidth() < 10 || rect.GetHeight() < 10)
			continue;

		Size text_size = GetTextSize(label, font);
		int max_width = max(20, rect.GetWidth() - 8);
		Rect chip(rect.left + 3, rect.top + 3,
		          rect.left + min(max_width, text_size.cx + 8),
		          rect.top + text_size.cy + 7);
		if(chip.right <= chip.left || chip.bottom <= chip.top)
			continue;

		w.DrawRect(chip, Color(255, 252, 214));
		DrawTextEllipsis(w, chip.left + 3, chip.top + 2, chip.GetWidth() - 6, label, "...",
		                 font, Black());
	}
}

OCRFormDocumentHost::OCRFormDocumentHost()
{
	Add(scroll.SizePos());
	view.SetContainer(scroll);
	view.New();
	view.SetBool("Grid.Visible", false);
	view.SetBool("Grid.Binding", false);
	view.HideInfo();
	scroll.Set(view, view.GetFormSize());

	annotation_list.AddColumn("Annotation");
	annotation_list.AddColumn("Rect");
	annotation_list.AddColumn("Analyser");
	annotation_list.ColumnWidths("6 3 2");
	annotation_list.WhenSel = [this] { SyncSelectionFromPane(); };

	annotation_details.SetReadOnly();
	annotation_details.SetFont(Courier(12));

	Ptr<OCRFormDocumentHost> self = this;
	analysis_worker.WhenProposalsReady = [self](const AnalysisResult& result) {
		if(!self)
			return;
		self->OnAnalysisResult(result);
	};
	analysis_worker.Start();
}

OCRFormDocumentHost::~OCRFormDocumentHost()
{
	live_refresh.Kill();
	analysis_tick.Kill();
	analysis_worker.Stop();
	ResetVideoSource();
}

bool OCRFormDocumentHost::Load(const String& path_)
{
	Stop();
	path = path_;
	modified = false;
	document = OcrFormDocument();
	source_status.Clear();
	pending_proposals.Clear();
	analysis_worker.ClearQueue();
	analysis_inflight = false;
	analysis_generation++;
	analysis_last_applied = 0;
	analysis_job_seq = 0;
	analysis_last_frame_hash = 0;
	analysis_last_frame = Null;
	last_live_frame = Null;
	changed_area_blocks = 0;
	view.ClearChangedRegions();
	ResetVideoSource();

	if(FileExists(path) && !document.Load(path))
		return false;

	RefreshViewFromDocument();
	return true;
}

bool OCRFormDocumentHost::Save()
{
	if(path.IsEmpty())
		return false;

	if(document.analysis_trigger_mode == kAnalysisModeOnSave)
		RunAnalysersNow(false);

	if(!document.Save(path))
		return false;

	modified = false;
	RefreshViewFromDocument();
	return true;
}

bool OCRFormDocumentHost::SaveAs(const String& path_)
{
	path = path_;
	return Save();
}

void OCRFormDocumentHost::SetFocus()
{
	scroll.SetFocus();
}

void OCRFormDocumentHost::ActivateUI()
{
	if(PythonIDE* ide = dynamic_cast<PythonIDE*>(Ctrl::GetTopWindow())) {
		Ptr<PythonIDE> p = ide;
		Ptr<OCRFormDocumentHost> self = this;
		PostCallback([p, self] {
			if(!p || !self)
				return;
			if(p->active_file < 0 || p->active_file >= p->open_files.GetCount())
				return;
			if(p->open_files[p->active_file].editor != self)
				return;

			auto add_right_tab = [&](const String& id, const String& title, Ctrl& ctrl) {
				int q = p->plugin_panes.Find(id);
				if(q >= 0) {
					DockableCtrl& pane = p->plugin_panes[q];
					if(ctrl.GetParent() != &pane) {
						ctrl.Remove();
						pane.Add(ctrl.SizePos());
					}
					pane.Show();
					return;
				}

				DockableCtrl& pane = p->plugin_panes.Add(id);
				pane.Title(title);
				pane.Add(ctrl.SizePos());
				p->Register(pane);
				p->Tabify(*p->var_explorer, pane);
				pane.Show();
			};

			add_right_tab("OCRFormAnnotations", "OCRForm Annotations", self->annotation_list);
			add_right_tab("OCRFormDetails", "OCRForm Details", self->annotation_details);
		});
	}
}

void OCRFormDocumentHost::DeactivateUI()
{
	if(PythonIDE* ide = dynamic_cast<PythonIDE*>(Ctrl::GetTopWindow())) {
		Ptr<PythonIDE> p = ide;
		PostCallback([p] {
			if(!p)
				return;
			if(p->active_file >= 0 && p->active_file < p->open_files.GetCount()) {
				if(dynamic_cast<OCRFormDocumentHost*>(p->open_files[p->active_file].editor))
					return;
			}

			auto hide_pane = [&](const String& id) {
				int q = p->plugin_panes.Find(id);
				if(q < 0)
					return;
				p->plugin_panes[q].Close();
				p->plugin_panes[q].Hide();
			};

			hide_pane("OCRFormAnnotations");
			hide_pane("OCRFormDetails");
		});
	}
}

void OCRFormDocumentHost::MainMenu(Bar& bar)
{
	auto apply_analysis_mode = [this](const String& mode) {
		if(document.analysis_trigger_mode == mode)
			return;
		document.analysis_trigger_mode = mode;
		modified = true;
		analysis_worker.ClearQueue();
		analysis_inflight = false;
		analysis_last_frame_hash = 0;
		analysis_last_frame = Null;
		analysis_tick.Kill();
		if(running && (document.analysis_trigger_mode == kAnalysisModeOnChange
		            || document.analysis_trigger_mode == kAnalysisModeContinuous)) {
			if(document.analysis_interval_ms > 0)
				analysis_tick.Set(-document.analysis_interval_ms, [this] { OnAnalysisTick(); });
			OnAnalysisTick();
		}
		RefreshHostUI();
	};

	bar.Sub("OCRForm", [this, apply_analysis_mode](Bar& menu) {
		menu.Add(running ? "Stop preview" : "Run preview",
		         running ? Icons::Stop() : Icons::Run(),
		         [this] { if(running) Stop(); else Run(); });
		menu.Add("Reload preview", Icons::OpenFile(), [this] { RefreshViewFromDocument(); });
		menu.Add("Export to .form...", Icons::Save(), [this] { ExportToForm(); });
		menu.Separator();
		menu.Sub("Source Type", [this](Bar& src_menu) {
			VideoSourceRegistry& registry = VideoSourceRegistry::Get();
			Vector<String> ids = registry.GetTypeIDs();
			if(ids.IsEmpty()) {
				src_menu.Add("No registered sources", [] {}).Enable(false);
				return;
			}

			for(int i = 0; i < ids.GetCount(); i++) {
				String type_id = ids[i];
				String name = registry.GetName(type_id);
				if(name.IsEmpty())
					name = type_id;

				src_menu.Add(name, [this, type_id] {
					if(document.video_source_type == type_id)
						return;
					document.video_source_type = type_id;
					modified = true;
					ResetVideoSource();
					RefreshViewFromDocument();
					RefreshHostUI();
				}).Check(document.video_source_type == type_id);
			}
		});
		menu.Add("Set source parameter...", [this] {
			String param = document.video_source_param;
			if(!EditText(param, "OCRForm Source Parameter", "Parameter:"))
				return;
			if(document.video_source_param == param)
				return;

			document.video_source_param = param;
			modified = true;
			ResetVideoSource();
			RefreshViewFromDocument();
			RefreshHostUI();
		});
		if(document.video_source_type == "screen") {
			menu.Add("Use full screen source", [this] {
				if(document.video_source_param == "full")
					return;
				document.video_source_param = "full";
				modified = true;
				ResetVideoSource();
				RefreshViewFromDocument();
				RefreshHostUI();
			});
		}
		menu.Separator();
		menu.Add("Select all annotations", [this] { SelectAll(); })
		    .Enable(annotation_list.GetCount() > 0);
		menu.Separator();
		menu.Sub("Analyser Proposals", [this, apply_analysis_mode](Bar& proposal_menu) {
			String selected_path = GetSelectedAnnotationPath();
			bool has_selected_pending = !selected_path.IsEmpty()
			                         && pending_proposals.Find(selected_path) >= 0;

			proposal_menu.Add("Run analysers now", Icons::Run(), [this] { RunAnalysersNow(); });
				proposal_menu.Sub("Trigger mode", [this, apply_analysis_mode](Bar& mode_menu) {
				mode_menu.Add("Manual only", [apply_analysis_mode] {
					apply_analysis_mode(kAnalysisModeManual);
				}).Check(document.analysis_trigger_mode == kAnalysisModeManual);
				mode_menu.Add("On frame change", [apply_analysis_mode] {
					apply_analysis_mode(kAnalysisModeOnChange);
				}).Check(document.analysis_trigger_mode == kAnalysisModeOnChange);
				mode_menu.Add("Continuous", [apply_analysis_mode] {
					apply_analysis_mode(kAnalysisModeContinuous);
				}).Check(document.analysis_trigger_mode == kAnalysisModeContinuous);
					mode_menu.Add("On save", [apply_analysis_mode] {
						apply_analysis_mode(kAnalysisModeOnSave);
					}).Check(document.analysis_trigger_mode == kAnalysisModeOnSave);
				});
				proposal_menu.Add("Set analysis interval...", [this] {
					String value = AsString(document.analysis_interval_ms);
					if(!EditText(value, "Analysis Interval", "Interval in milliseconds (0 = every UI tick):"))
						return;
					int parsed = StrInt(value);
					if(parsed < 0) {
						Exclamation("Interval must be >= 0.");
						return;
					}
					if(document.analysis_interval_ms == parsed)
						return;
					document.analysis_interval_ms = parsed;
					modified = true;
					analysis_tick.Kill();
					if(running && (document.analysis_trigger_mode == kAnalysisModeOnChange
					            || document.analysis_trigger_mode == kAnalysisModeContinuous)) {
						if(document.analysis_interval_ms > 0)
							analysis_tick.Set(-document.analysis_interval_ms, [this] { OnAnalysisTick(); });
						OnAnalysisTick();
					}
					RefreshHostUI();
				});
				proposal_menu.Add("Set changed-area threshold...", [this] {
					String value = AsString(document.changed_area_threshold);
					if(!EditText(value, "Changed Area Threshold", "Per-channel delta threshold (0-255):"))
						return;
					int parsed = StrInt(value);
					if(parsed < 0 || parsed > 255) {
						Exclamation("Threshold must be between 0 and 255.");
						return;
					}
					if(document.changed_area_threshold == parsed)
						return;
					document.changed_area_threshold = parsed;
					modified = true;
					if(running)
						OnLiveRefresh();
					RefreshHostUI();
				});
				proposal_menu.Add("Set changed-area block size...", [this] {
					String value = AsString(document.changed_area_block_size);
					if(!EditText(value, "Changed Area Block Size", "Block size in pixels (1-128):"))
						return;
					int parsed = StrInt(value);
					if(parsed < 1 || parsed > 128) {
						Exclamation("Block size must be between 1 and 128.");
						return;
					}
					if(document.changed_area_block_size == parsed)
						return;
					document.changed_area_block_size = parsed;
					modified = true;
					if(running)
						OnLiveRefresh();
					RefreshHostUI();
				});
				proposal_menu.Add("Set accept threshold...", [this] {
				String value = FormatDouble(document.proposal_accept_threshold);
				if(!EditText(value, "Proposal Acceptance Threshold", "Confidence (0.0 - 1.0):"))
					return;
				double parsed = StrDbl(value);
				if(parsed < 0.0 || parsed > 1.0) {
					Exclamation("Threshold must be between 0.0 and 1.0.");
					return;
				}
				document.proposal_accept_threshold = parsed;
				modified = true;
				RefreshHostUI();
			});
			proposal_menu.Add("Auto-accept above threshold", [this] {
				document.auto_accept_proposals = !document.auto_accept_proposals;
				modified = true;
				RefreshHostUI();
			}).Check(document.auto_accept_proposals);
			proposal_menu.Separator();
			proposal_menu.Add("Accept selected", Icons::Plus(), [this] { AcceptSelectedProposal(); })
			            .Enable(has_selected_pending);
			proposal_menu.Add("Reject selected", Icons::Minus(), [this] { RejectSelectedProposal(); })
			            .Enable(has_selected_pending);
			proposal_menu.Add("Accept all", Icons::Plus(), [this] { AcceptAllProposals(); })
			            .Enable(pending_proposals.GetCount() > 0);
			proposal_menu.Add("Clear all pending", Icons::Stop(), [this] { ClearPendingProposals(); })
			            .Enable(pending_proposals.GetCount() > 0);
		});
		menu.Separator();
		menu.Add(Format("Source: %s",
		                document.video_source_type.IsEmpty() ? "none" : ~document.video_source_type),
		         [] {}).Enable(false);
		if(!document.video_source_param.IsEmpty()) {
			menu.Add(Format("Param: %s", ~document.video_source_param), [] {})
			    .Enable(false);
		}
	});
}

void OCRFormDocumentHost::Toolbar(Bar& bar)
{
	bar.Add(running ? Icons::Stop() : Icons::Run(),
	        [this] { if(running) Stop(); else Run(); })
	    .Tip(running ? "Stop OCRForm preview" : "Run OCRForm preview");
	bar.Add(Icons::OpenFile(), [this] { RefreshViewFromDocument(); })
	    .Tip("Reload OCRForm preview");
	bar.Add(Icons::Run(), [this] { RunAnalysersNow(); })
	    .Tip("Run analysers");
	bar.Add(Icons::Plus(), [this] { AcceptAllProposals(); })
	    .Tip("Accept all pending proposals");
	bar.Add(Icons::Save(), [this] { ExportToForm(); })
	    .Tip("Export OCRForm to .form");
	bar.Add(Icons::Settings(), [this] {
		String param = document.video_source_param;
		if(!EditText(param, "OCRForm Source Parameter", "Parameter:"))
			return;
		if(document.video_source_param == param)
			return;
		document.video_source_param = param;
		modified = true;
		ResetVideoSource();
		RefreshViewFromDocument();
		RefreshHostUI();
	}).Tip("Set source parameter");
}

void OCRFormDocumentHost::Run()
{
	if(running)
		return;

	running = true;
	analysis_generation++;
	analysis_inflight = false;
	analysis_last_frame_hash = 0;
	analysis_last_frame = Null;
	last_live_frame = Null;
	changed_area_blocks = 0;
	view.ClearChangedRegions();
	OnLiveRefresh();
	int poll_ms = refresh_interval_ms;
	if(EnsureVideoSource() && video_source) {
		int preferred = video_source->GetPreferredPollMs();
		if(preferred > 0)
			poll_ms = preferred;
		else
			poll_ms = 0;
	}
	if(poll_ms > 0)
		live_refresh.Set(-poll_ms, [this] { OnLiveRefresh(); });
	if(document.analysis_trigger_mode == kAnalysisModeOnChange
	|| document.analysis_trigger_mode == kAnalysisModeContinuous) {
		if(document.analysis_interval_ms > 0)
			analysis_tick.Set(-document.analysis_interval_ms, [this] { OnAnalysisTick(); });
		OnAnalysisTick();
	}

	RefreshHostUI();
}

void OCRFormDocumentHost::Stop()
{
	live_refresh.Kill();
	analysis_tick.Kill();
	analysis_worker.ClearQueue();
	analysis_inflight = false;
	analysis_generation++;
	analysis_last_frame_hash = 0;
	analysis_last_frame = Null;
	last_live_frame = Null;
	changed_area_blocks = 0;
	view.ClearChangedRegions();
	if(!running) {
		RefreshHostUI();
		return;
	}

	running = false;
	RefreshHostUI();
}

void OCRFormDocumentHost::SelectAll()
{
	if(annotation_list.GetCount() <= 0)
		return;
	annotation_list.SetCursor(0);
	view.ClearSelection();
	for(int i = 0; i < view.GetObjectCount(); i++)
		view.AddToSelection(i);
	view.Refresh();
	RefreshAnnotationDetails();
}

String OCRFormDocumentHost::BuildAnnotationLabel(const OcrAnnotation& annotation) const
{
	String label = annotation.name;
	if(!annotation.widget_type.IsEmpty()) {
		if(label.IsEmpty())
			label = annotation.widget_type;
		else
			label << " [" << annotation.widget_type << "]";
	}

	if(label.IsEmpty())
		label = "annotation";

	if(annotation.confidence > 0.0)
		label << Format(" %.0f%%", annotation.confidence * 100.0);

	return label;
}

void OCRFormDocumentHost::RefreshViewFromDocument()
{
	int current_row = annotation_list.GetCursor();

	view.ClearBackgroundImage();
	view.New();
	view.SetBool("Grid.Visible", false);
	view.SetBool("Grid.Binding", false);
	view.HideInfo();

	Image background = CaptureSourceFrame();
	if(running && !IsNull(background)) {
		Vector<Rect> changed = ComputeChangedBlocks(last_live_frame,
		                                            background,
		                                            document.changed_area_threshold,
		                                            document.changed_area_block_size);
		changed_area_blocks = changed.GetCount();
		view.SetChangedRegions(pick(changed));
		last_live_frame = background;
	}
	else {
		changed_area_blocks = 0;
		view.ClearChangedRegions();
		if(!running)
			last_live_frame = Null;
	}

	Size canvas = ResolveCanvasSize(document, background);
	view.SetFormSize(canvas);
	view.ClearSelection();

	if(document.layouts.GetCount()) {
		const OcrFormLayout& layout = document.layouts[0];
		if(!layout.name.IsEmpty())
			view.UpdateLayoutName(0, layout.name);
		AppendAnnotations(layout.roots, String());
	}

	if(!IsNull(background))
		view.SetBackgroundImage(background);

	scroll.Set(view, canvas);
	RebuildAnnotationPane();
	if(current_row >= 0 && current_row < annotation_list.GetCount())
		annotation_list.SetCursor(current_row);
	else if(annotation_list.GetCount())
		annotation_list.SetCursor(0);
	else
		RefreshAnnotationDetails(-1);
	view.Refresh();
}

void OCRFormDocumentHost::AppendAnnotations(const Array<OcrAnnotation>& annotations, const String& path_prefix)
{
	Array<FormObject>* objects = view.GetObjects();
	if(!objects)
		return;

	for(int i = 0; i < annotations.GetCount(); i++) {
		const OcrAnnotation& annotation = annotations[i];
		FormObject& object = objects->Add(FormObject(NormalizeAnnotationRect(annotation.rect)));
		object.Set("Type", annotation.widget_type.IsEmpty() ? "annotation" : annotation.widget_type);
		object.Set("Variable", BuildAnnotationPath(path_prefix, annotation, i));
		object.Name = object.Get("Variable");
		object.Set("OCR.Name", annotation.name);
		String visual_label = BuildAnnotationLabel(annotation);
		if(pending_proposals.Find(object.Get("Variable")) >= 0)
			visual_label << " [pending]";
		object.Set("OCR.Label", visual_label);
		object.Set("OCR.WidgetType", annotation.widget_type);
		object.Set("OCR.AnalyserID", annotation.analyser_id);
		object.Set("OCR.Confidence", FormatDouble(annotation.confidence));
		object.SetBool("OutlineDraw", true);

		if(annotation.children.GetCount())
			AppendAnnotations(annotation.children, object.Get("Variable"));
	}
}

void OCRFormDocumentHost::AppendAnnotationRows(const Array<OcrAnnotation>& annotations,
                                               const String& path_prefix, int depth)
{
	for(int i = 0; i < annotations.GetCount(); i++) {
		const OcrAnnotation& annotation = annotations[i];
		String path = BuildAnnotationPath(path_prefix, annotation, i);
		String label = annotation.name;
		if(label.IsEmpty())
			label = annotation.widget_type.IsEmpty() ? "annotation" : annotation.widget_type;

		String analyser_display = annotation.analyser_id;
		int pending_i = pending_proposals.Find(path);
		if(pending_i >= 0) {
			const AnnotationProposal& proposal = pending_proposals[pending_i];
			String pending_analyser = proposal.analyser_id;
			if(pending_analyser.IsEmpty())
				pending_analyser = "<unknown>";
			if(analyser_display.IsEmpty())
				analyser_display = pending_analyser + " ?";
			else
				analyser_display << " -> " << pending_analyser << " ?";
		}

		annotation_list.Add(BuildIndentLabel(label, depth),
		                    FormatRectSummary(annotation.rect),
		                    analyser_display);
		annotation_paths.Add(path);

		String details;
		details << "Path: " << path << "\n";
		details << "Name: " << (annotation.name.IsEmpty() ? "<unnamed>" : annotation.name) << "\n";
		details << "Widget: " << (annotation.widget_type.IsEmpty() ? "<unset>" : annotation.widget_type) << "\n";
		details << "Rect: " << FormatRectSummary(annotation.rect) << "\n";
		details << "Analyser: " << (annotation.analyser_id.IsEmpty() ? "<none>" : annotation.analyser_id) << "\n";
		details << "Confidence: " << FormatDouble(annotation.confidence) << "\n";
		if(annotation.hints.GetCount()) {
			details << "Hints:\n";
			for(int j = 0; j < annotation.hints.GetCount(); j++)
				details << "  " << annotation.hints.GetKey(j) << " = " << annotation.hints[j] << "\n";
		}
		if(pending_i >= 0) {
			const AnnotationProposal& proposal = pending_proposals[pending_i];
			details << "Pending proposal:\n";
			details << "  analyser: " << (proposal.analyser_id.IsEmpty() ? "<unknown>" : proposal.analyser_id) << "\n";
			details << "  confidence: " << FormatDouble(ResolveProposalConfidence(proposal)) << "\n";
			details << "  widget: " << (proposal.proposed.widget_type.IsEmpty() ? "<unchanged>" : proposal.proposed.widget_type) << "\n";
			if(!proposal.proposed.rect.IsEmpty())
				details << "  rect: " << FormatRectSummary(proposal.proposed.rect) << "\n";
			details << "  children: " << proposal.proposed.children.GetCount() << "\n";
		}
		annotation_descriptions.Add(details);

		if(annotation.children.GetCount())
			AppendAnnotationRows(annotation.children, path, depth + 1);
	}
}

void OCRFormDocumentHost::RebuildAnnotationPane()
{
	annotation_list.Clear();
	annotation_paths.Clear();
	annotation_descriptions.Clear();

	if(document.layouts.GetCount())
		AppendAnnotationRows(document.layouts[0].roots, String(), 0);

	RefreshAnnotationDetails();
}

void OCRFormDocumentHost::RefreshAnnotationDetails(int row)
{
	if(row < 0)
		row = annotation_list.GetCursor();

	if(row >= 0 && row < annotation_descriptions.GetCount()) {
		annotation_details.Set(annotation_descriptions[row]);
		return;
	}

	String info;
	info << "Path: " << (path.IsEmpty() ? "<unsaved>" : path) << "\n";
	info << "Layouts: " << document.layouts.GetCount() << "\n";
	info << "Source type: "
	     << (document.video_source_type.IsEmpty() ? "<none>" : document.video_source_type) << "\n";
	info << "Source param: "
	     << (document.video_source_param.IsEmpty() ? "<none>" : document.video_source_param) << "\n";
	info << "Source state: "
	     << (source_status.IsEmpty() ? "<uninitialized>" : source_status) << "\n";
	info << "Mode: " << (running ? "live preview" : "edit preview") << "\n";
	info << "Analysis trigger: " << AnalysisModeLabel(document.analysis_trigger_mode) << "\n";
	info << "Analysis interval (ms): " << document.analysis_interval_ms << "\n";
	info << "Changed threshold: " << document.changed_area_threshold << "\n";
	info << "Changed block size: " << document.changed_area_block_size << "\n";
	info << "Accept threshold: " << FormatDouble(document.proposal_accept_threshold) << "\n";
	info << "Auto-accept: " << (document.auto_accept_proposals ? "on" : "off") << "\n";
	info << "Changed blocks: " << changed_area_blocks << "\n";
	info << "Pending proposals: " << pending_proposals.GetCount() << "\n";
	annotation_details.Set(info);
}

void OCRFormDocumentHost::SyncSelectionFromPane()
{
	int row = annotation_list.GetCursor();
	view.ClearSelection();
	if(row >= 0 && row < view.GetObjectCount())
		view.AddToSelection(row);
	view.Refresh();
	RefreshAnnotationDetails(row);
}

String OCRFormDocumentHost::ResolveSourceParam(const String& type_id, const String& param) const
{
	if(type_id == "static_image")
		return ResolveMediaPath(path, param);
	return param;
}

void OCRFormDocumentHost::ResetVideoSource()
{
	if(video_source)
		video_source->Close();
	video_source.Clear();
	active_source_type.Clear();
	active_source_param.Clear();
}

bool OCRFormDocumentHost::EnsureVideoSource()
{
	String type_id = document.video_source_type;
	if(type_id.IsEmpty() && !document.video_source_param.IsEmpty())
		type_id = "static_image";

	if(type_id.IsEmpty()) {
		ResetVideoSource();
		source_status = "No video source selected";
		return false;
	}

	String param = ResolveSourceParam(type_id, document.video_source_param);
	bool needs_recreate = !video_source || active_source_type != type_id;
	if(needs_recreate) {
		ResetVideoSource();
		video_source = VideoSourceRegistry::Get().Create(type_id);
		if(!video_source) {
			source_status = Format("Unknown source type: %s", type_id);
			return false;
		}
		active_source_type = type_id;
	}

	if(!video_source->IsOpen() || active_source_param != param) {
		if(!video_source->Open(param)) {
			source_status = Format("Failed to open source '%s' with param '%s'", type_id, param);
			return false;
		}
		active_source_param = param;
	}

	String source_name = VideoSourceRegistry::Get().GetName(type_id);
	if(source_name.IsEmpty())
		source_name = type_id;
	source_status = Format("Source: %s (%s)", source_name, active_source_param);
	return true;
}

Image OCRFormDocumentHost::CaptureSourceFrame()
{
	if(!EnsureVideoSource())
		return Null;

	Image frame = video_source->GetFrame();
	if(IsNull(frame)) {
		source_status = Format("Source '%s' returned no frame", active_source_type);
		return Null;
	}

	source_status << Format(", frame %dx%d", frame.GetWidth(), frame.GetHeight());
	return frame;
}

void OCRFormDocumentHost::RefreshHostUI()
{
	RefreshAnnotationDetails();
	if(PythonIDE* ide = dynamic_cast<PythonIDE*>(Ctrl::GetTopWindow()))
		ide->RefreshRunStateUI();
}

void OCRFormDocumentHost::OnLiveRefresh()
{
	RefreshViewFromDocument();
	if(running && video_source) {
		int preferred = video_source->GetPreferredPollMs();
		if(preferred > 0 && preferred != refresh_interval_ms)
			live_refresh.Set(-preferred, [this] { OnLiveRefresh(); });
	}
}

String OCRFormDocumentHost::GetSelectedAnnotationPath() const
{
	int row = annotation_list.GetCursor();
	if(row < 0 || row >= annotation_paths.GetCount())
		return String();
	return annotation_paths[row];
}

OcrAnnotation* OCRFormDocumentHost::FindAnnotationByPath(Array<OcrAnnotation>& annotations,
                                                         const String& path,
                                                         const String& path_prefix)
{
	for(int i = 0; i < annotations.GetCount(); i++) {
		OcrAnnotation& ann = annotations[i];
		String ann_path = BuildAnnotationPath(path_prefix, ann, i);
		if(ann_path == path)
			return &ann;
		if(ann.children.GetCount()) {
			if(OcrAnnotation* found = FindAnnotationByPath(ann.children, path, ann_path))
				return found;
		}
	}
	return NULL;
}

void OCRFormDocumentHost::ClearPendingByPrefix(const String& path_prefix)
{
	for(int i = pending_proposals.GetCount() - 1; i >= 0; i--) {
		String key = pending_proposals.GetKey(i);
		if(key == path_prefix || key.StartsWith(path_prefix + "/"))
			pending_proposals.Remove(i);
	}
}

bool OCRFormDocumentHost::RejectProposalByPath(const String& path)
{
	int q = pending_proposals.Find(path);
	if(q < 0)
		return false;
	ClearPendingByPrefix(path);
	return true;
}

bool OCRFormDocumentHost::AcceptProposalByPath(const String& path)
{
	if(document.layouts.IsEmpty())
		return false;

	int q = pending_proposals.Find(path);
	if(q < 0)
		return false;

	OcrAnnotation* ann = FindAnnotationByPath(document.layouts[0].roots, path);
	if(!ann) {
		pending_proposals.Remove(q);
		return false;
	}

	const AnnotationProposal& proposal = pending_proposals[q];
	const OcrAnnotation& proposed = proposal.proposed;

	if(!proposed.rect.IsEmpty())
		ann->rect = NormalizeAnnotationRect(proposed.rect);
	if(!proposed.widget_type.IsEmpty())
		ann->widget_type = proposed.widget_type;
	if(ann->name.IsEmpty() && !proposed.name.IsEmpty())
		ann->name = proposed.name;
	if(!proposed.children.IsEmpty())
		ann->children = clone(proposed.children);
	if(!proposed.hints.IsEmpty())
		ann->hints = clone(proposed.hints);

	ann->analyser_id = proposal.analyser_id.IsEmpty() ? proposed.analyser_id : proposal.analyser_id;
	ann->confidence = max(ann->confidence, ResolveProposalConfidence(proposal));

	ClearPendingByPrefix(path);
	modified = true;
	return true;
}

void OCRFormDocumentHost::ClearPendingProposals()
{
	if(pending_proposals.IsEmpty())
		return;
	pending_proposals.Clear();
	RefreshViewFromDocument();
	RefreshHostUI();
}

void OCRFormDocumentHost::AcceptSelectedProposal()
{
	String path = GetSelectedAnnotationPath();
	if(path.IsEmpty())
		return;
	if(!AcceptProposalByPath(path))
		return;
	RefreshViewFromDocument();
	RefreshHostUI();
}

void OCRFormDocumentHost::RejectSelectedProposal()
{
	String path = GetSelectedAnnotationPath();
	if(path.IsEmpty())
		return;
	if(!RejectProposalByPath(path))
		return;
	RefreshViewFromDocument();
	RefreshHostUI();
}

void OCRFormDocumentHost::AcceptAllProposals()
{
	if(pending_proposals.IsEmpty())
		return;

	Vector<String> paths;
	for(int i = 0; i < pending_proposals.GetCount(); i++)
		paths.Add(pending_proposals.GetKey(i));

	int accepted = 0;
	int skipped = 0;
	for(int i = 0; i < paths.GetCount(); i++) {
		int q = pending_proposals.Find(paths[i]);
		if(q < 0)
			continue;
		if(ResolveProposalConfidence(pending_proposals[q]) < document.proposal_accept_threshold) {
			skipped++;
			continue;
		}
		if(AcceptProposalByPath(paths[i]))
			accepted++;
	}

	RefreshViewFromDocument();
	RefreshHostUI();
	if(accepted > 0 || skipped > 0)
		PromptOK(Format("Accepted %d proposal(s), skipped %d below threshold %.2f.",
		               accepted, skipped, document.proposal_accept_threshold));
}

void OCRFormDocumentHost::RunAnalyserRecursive(const Image& frame,
                                               const OcrFormLayout& layout,
                                               const OcrAnnotation* parent,
                                               const Array<OcrAnnotation>& annotations,
                                               const String& path_prefix,
                                               Index<String>& visited_paths)
{
	static const VectorMap<String, String> empty_hints;

	for(int i = 0; i < annotations.GetCount(); i++) {
		const OcrAnnotation& ann = annotations[i];
		String path = BuildAnnotationPath(path_prefix, ann, i);
		visited_paths.FindAdd(path);

		Rect region = NormalizeAnnotationRect(ann.rect);
		if(!region.IsEmpty()) {
			OcrAnalyserContext ctx{frame, region, parent, layout, empty_hints};
			Vector<OcrAnnotation> proposals = OcrAnalyserRegistry::Get().AnalyseAll(ctx);

			bool has_pending = false;
			for(int j = 0; j < proposals.GetCount(); j++) {
				const OcrAnnotation& p = proposals[j];
				String analyser_id = p.analyser_id;
				if(analyser_id.IsEmpty())
					continue;
				if(analyser_id == "manual")
					continue;
				if(!IsMeaningfulProposal(ann, p))
					continue;

				AnnotationProposal& slot = pending_proposals.GetAdd(path);
				slot.analyser_id = analyser_id;
				slot.proposed = p;
				slot.confidence = p.confidence;
				has_pending = true;
				break;
			}

			if(!has_pending) {
				int q = pending_proposals.Find(path);
				if(q >= 0)
					pending_proposals.Remove(q);
			}
			else if(document.auto_accept_proposals) {
				int q = pending_proposals.Find(path);
				if(q >= 0 && ResolveProposalConfidence(pending_proposals[q]) >= document.proposal_accept_threshold)
					AcceptProposalByPath(path);
			}
		}

		if(ann.children.GetCount())
			RunAnalyserRecursive(frame, layout, &ann, ann.children, path, visited_paths);
	}
}

void OCRFormDocumentHost::RunAnalysersNow(bool interactive)
{
	if(document.layouts.IsEmpty())
		return;

	analysis_generation++;
	analysis_worker.ClearQueue();
	analysis_inflight = false;
	analysis_last_frame_hash = 0;
	analysis_last_frame = Null;

	Image frame = CaptureSourceFrame();
	if(IsNull(frame)) {
		if(interactive)
			Exclamation("Cannot run analysers without a valid frame.");
		return;
	}

	Index<String> visited_paths;
	RunAnalyserRecursive(frame, document.layouts[0], NULL, document.layouts[0].roots, String(), visited_paths);

	for(int i = pending_proposals.GetCount() - 1; i >= 0; i--) {
		if(visited_paths.Find(pending_proposals.GetKey(i)) < 0)
			pending_proposals.Remove(i);
	}

	RefreshViewFromDocument();
	RefreshHostUI();
}

uint64 OCRFormDocumentHost::ComputeFrameHash(const Image& frame) const
{
	if(frame.IsEmpty())
		return 0;

	int w = frame.GetWidth();
	int h = frame.GetHeight();
	uint64 hash = ((uint64)w << 32) ^ (uint64)h;
	hash ^= 1469598103934665603ull;

	const int sx_count = 6;
	const int sy_count = 4;
	for(int sy = 0; sy < sy_count; sy++) {
		int y = (h - 1) * sy / max(1, sy_count - 1);
		for(int sx = 0; sx < sx_count; sx++) {
			int x = (w - 1) * sx / max(1, sx_count - 1);
			const RGBA& px = frame[y][x];
			uint32 v = ((uint32)px.r << 24) | ((uint32)px.g << 16) | ((uint32)px.b << 8) | (uint32)px.a;
			hash ^= (uint64)v;
			hash *= 1099511628211ull;
		}
	}
	return hash;
}

void OCRFormDocumentHost::OnAnalysisTick()
{
	if(!running)
		return;

	if(document.analysis_trigger_mode != kAnalysisModeOnChange
	&& document.analysis_trigger_mode != kAnalysisModeContinuous)
		return;

	if(document.analysis_interval_ms > 0)
		analysis_tick.Set(-document.analysis_interval_ms, [this] { OnAnalysisTick(); });

	if(document.layouts.IsEmpty() || analysis_inflight)
		return;

	Image frame = last_live_frame;
	if(IsNull(frame))
		frame = CaptureSourceFrame();
	if(IsNull(frame))
		return;

	uint64 frame_hash = ComputeFrameHash(frame);
	if(document.analysis_trigger_mode == kAnalysisModeOnChange) {
		if(frame_hash == analysis_last_frame_hash)
			return;
	}

	Vector<Rect> changed_for_job = ComputeChangedBlocks(analysis_last_frame,
	                                                    frame,
	                                                    document.changed_area_threshold,
	                                                    document.changed_area_block_size);
	analysis_last_frame = frame;
	analysis_last_frame_hash = frame_hash;
	AnalysisJob job;
	job.id = ++analysis_job_seq;
	job.generation = analysis_generation;
	job.frame = frame;
	job.layout_snapshot = document.layouts[0];
	job.changed_regions = pick(changed_for_job);
	analysis_worker.SubmitJob(pick(job));
	analysis_inflight = true;
}

void OCRFormDocumentHost::OnAnalysisResult(const AnalysisResult& result)
{
	if(result.generation != analysis_generation)
		return;
	if(result.id < analysis_last_applied)
		return;

	analysis_last_applied = result.id;
	if(result.id == analysis_job_seq)
		analysis_inflight = false;

	Index<String> visited;
	for(int i = 0; i < result.visited_paths.GetCount(); i++)
		visited.FindAdd(result.visited_paths[i]);

	bool changed = false;
	Vector<String> auto_accept_paths;
	for(int i = pending_proposals.GetCount() - 1; i >= 0; i--) {
		if(visited.Find(pending_proposals.GetKey(i)) < 0) {
			pending_proposals.Remove(i);
			changed = true;
		}
	}

	for(int i = 0; i < result.proposals.GetCount(); i++) {
		String path = result.proposals.GetKey(i);
		const AnnotationProposal& proposal = result.proposals[i];
		if(document.auto_accept_proposals && ResolveProposalConfidence(proposal) >= document.proposal_accept_threshold)
			auto_accept_paths.Add(path);
		int q = pending_proposals.Find(path);
		if(q < 0) {
			pending_proposals.Add(path, proposal);
			changed = true;
		}
		else if(pending_proposals[q].analyser_id != proposal.analyser_id
		     || pending_proposals[q].confidence != proposal.confidence
		     || pending_proposals[q].proposed.widget_type != proposal.proposed.widget_type
		     || pending_proposals[q].proposed.children.GetCount() != proposal.proposed.children.GetCount()) {
			pending_proposals[q] = proposal;
			changed = true;
		}
	}

	for(int i = 0; i < auto_accept_paths.GetCount(); i++) {
		if(AcceptProposalByPath(auto_accept_paths[i]))
			changed = true;
	}

	if(changed) {
		RefreshViewFromDocument();
		RefreshHostUI();
	}
}

String OCRFormDocumentHost::SuggestFormExportPath() const
{
	String base = path.IsEmpty() ? "ocrform_export" : GetFileTitle(path);
	String dir = path.IsEmpty() ? GetCurrentDirectory() : GetFileDirectory(path);
	return AppendFileName(dir, base + ".form");
}

void OCRFormDocumentHost::ExportToForm()
{
	OcrFormExporter exporter;
	String xml = exporter.Export(document);
	if(xml.IsEmpty()) {
		Exclamation("Failed to export .form XML");
		return;
	}

	FileSel fs;
	fs.Type("Form files", "*.form");
	fs <<= SuggestFormExportPath();
	if(!fs.ExecuteSaveAs("Export OCRForm to .form"))
		return;

	String out_path = ~fs;
	if(out_path.IsEmpty())
		return;
	if(GetFileExt(out_path).IsEmpty())
		out_path << ".form";

	if(!SaveFile(out_path, xml)) {
		Exclamation(Format("Failed to write file:&%s", out_path));
		return;
	}

	String summary;
	summary << Format("Exported %d object(s) to:&%s", exporter.GetExportedCount(), out_path);
	const Vector<String>& warns = exporter.GetWarnings();
	if(warns.GetCount()) {
		summary << Format("&Warnings (%d):&", warns.GetCount());
		int limit = min(warns.GetCount(), 12);
		for(int i = 0; i < limit; i++)
			summary << " - " << warns[i] << "&";
		if(warns.GetCount() > limit)
			summary << Format(" - ... and %d more warning(s)&", warns.GetCount() - limit);
	}
	PromptOK(summary);

	if(PromptYesNo("Open exported .form in a new tab?")) {
		if(PythonIDE* ide = dynamic_cast<PythonIDE*>(Ctrl::GetTopWindow()))
			ide->LoadFile(out_path);
	}
}

END_UPP_NAMESPACE
