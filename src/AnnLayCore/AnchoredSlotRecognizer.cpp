#include "AnchoredSlotRecognizer.h"
#include "AnnMdl.h"
#include "AnchoredSlotClassifier.h"

#include <Draw/Draw.h>
#include <ComputerVision/ComputerVision.h>
#include <plugin/png/png.h>
#include <OCR/Tesseract.h>
#include <cstring>
#include <cmath>
#include <cfloat>

NAMESPACE_UPP

struct AnchoredSlotRecognizer::CvTemplateCache : Moveable<AnchoredSlotRecognizer::CvTemplateCache> {
	int group_index = -1;
	Size native_size_a = Size(0, 0);
	Size native_size_b = Size(0, 0);
	Vector< VectorMap<String, ByteMat> > label_a;  // [zoom_step][class_name]
	Vector< VectorMap<String, ByteMat> > label_b;  // [zoom_step][class_name]
};

static String SafeGetClass(::ConvNet::Session& ses, int i) {
	if(i >= 0 && i < ses.Data().GetClassCount())
		return ses.Data().GetClass(i);
	return AsString(i);
}

static String SafeFormatInt(int v) {
	if(IsNull(v)) return "N/A";
	return Format("%+d", v);
}

static int ParseSeatPanelId(const String& id) {
	if(!id.StartsWith("seat") || !id.EndsWith("_panel"))
		return -1;
	int pos = 4;
	String digits;
	while(pos < id.GetCount() && IsDigit((byte)id[pos]))
		digits.Cat(id[pos++]);
	if(digits.IsEmpty() || id.Mid(pos) != "_panel")
		return -1;
	int seat = StrInt(digits);
	return seat >= 1 && seat <= 8 ? seat : -1;
}

static Pointf AnchorCenterPx(const AnnLayAnchor& a, Size img_size) {
	return Pointf(a.cx * img_size.cx, a.cy * img_size.cy);
}

static bool IsZeroAnchor(const AnnLayAnchor& a) {
	return a.cx == 0.0 && a.cy == 0.0 && a.w == 0.0 && a.h == 0.0;
}

static AnnLayAnchor ExpandAnchor(const AnnLayAnchor& a, double bbox_expand) {
	AnnLayAnchor out = a;
	out.w *= (1.0 + 2.0 * bbox_expand);
	out.h *= (1.0 + 2.0 * bbox_expand);
	return out;
}

static Vector<int> MapCandidatesToSeats(const AnnLay& lay, const AnnLaySlot& slot, Size img_size) {
	Vector<Pointf> seat_center;
	seat_center.SetCount(9);
	Vector<bool> has_seat;
	has_seat.SetCount(9, false);

	for(const AnnLaySlot& s : lay.slots) {
		int seat = ParseSeatPanelId(s.id);
		if(seat > 0) {
			seat_center[seat] = AnchorCenterPx(s.anchor, img_size);
			has_seat[seat] = true;
		}
	}

	Vector<int> out;
	for(const AnnLayAnchor& cand : slot.anchor_candidates) {
		Pointf c = AnchorCenterPx(cand, img_size);
		int best_seat = -1;
		double best_d2 = DBL_MAX;
		for(int seat = 1; seat <= 8; seat++) {
			if(!has_seat[seat])
				continue;
			double dx = c.x - seat_center[seat].x;
			double dy = c.y - seat_center[seat].y;
			double d2 = dx * dx + dy * dy;
			if(d2 < best_d2) {
				best_d2 = d2;
				best_seat = seat;
			}
		}
		out.Add(best_seat);
	}
	return out;
}

static bool IsCandidateSeatMappingBijection(const Vector<int>& seats) {
	if(seats.GetCount() != 8)
		return false;
	Vector<bool> seen;
	seen.SetCount(9, false);
	for(int i = 0; i < seats.GetCount(); i++) {
		int seat = seats[i];
		if(seat < 1 || seat > 8 || seen[seat])
			return false;
		seen[seat] = true;
	}
	for(int seat = 1; seat <= 8; seat++) {
		if(!seen[seat])
			return false;
	}
	return true;
}

static String FormatCandidateSeatMapping(const Vector<int>& seats) {
	String out = "[GROUP_RECOGNIZE] seat ordering:";
	for(int ci = 0; ci < seats.GetCount(); ci++)
		out << (ci ? ", " : " ") << "ci" << ci << "->seat" << seats[ci];
	return out;
}

static void AppendProcessingWarning(ProcessingLogRecord* log, const String& warning) {
	if(!log)
		return;
	if(!log->warnings.IsEmpty())
		log->warnings << "\n";
	log->warnings << warning;
}

static String NormalizeRankCode(const String& v) {
	String s = TrimBoth(ToLower(v));
	if(s == "ace"   || s == "a")  return "A";
	if(s == "king"  || s == "k")  return "K";
	if(s == "queen" || s == "q")  return "Q";
	if(s == "jack"  || s == "j")  return "J";
	if(s == "10" || s == "t")      return "T";
	if(s.GetCount() == 1 && s[0] >= '2' && s[0] <= '9') return ToUpper(s);
	return v;
}

static String NormalizeSuitCode(const String& v) {
	String s = TrimBoth(ToLower(v));
	if(s.IsEmpty())
		return s;
	if(s == "spades"   || s == "type_c")   return "s";
	if(s == "hearts"   || s == "type_b")   return "h";
	if(s == "diamonds" || s == "type_d") return "d";
	if(s == "clubs"    || s == "type_a")    return "c";
	if(s.GetCount() == 1 && (s[0] == 's' || s[0] == 'h' || s[0] == 'd' || s[0] == 'c'))
		return s;
	bool all_digits = true;
	for(int i = 0; i < s.GetCount(); i++) {
		if(!IsDigit((byte)s[i])) {
			all_digits = false;
			break;
		}
	}
	if(all_digits) {
		int idx = StrInt(s);
		static const char* kSuitByIdx[] = {"s", "h", "d", "c"};
		if(idx >= 0 && idx < 4)
			return kSuitByIdx[idx];
	}
	return s;
}

static bool& FrameRecMemDumpFlag() {
	static bool enabled = false;
	return enabled;
}

static int64 ReadProcMemKb(const char* key) {
	String status = LoadFile("/proc/self/status");
	if(status.IsEmpty())
		return -1;
	Vector<String> lines = Split(status, '\n');
	for(const String& ln : lines) {
		if(ln.StartsWith(key)) {
			String tail = ln.Mid((int)strlen(key));
			String digits;
			for(int i = 0; i < tail.GetCount(); i++) {
				if(tail[i] >= '0' && tail[i] <= '9')
					digits.Cat(tail[i]);
			}
			if(digits.IsEmpty())
				return -1;
			int64 kb = ScanInt64(digits);
			return kb;
		}
	}
	return -1;
}

void SetFrameRecognizerMemoryDumpEnabled(bool enabled) {
	FrameRecMemDumpFlag() = enabled;
}

bool IsFrameRecognizerMemoryDumpEnabled() {
	static bool env_init = false;
	if(!env_init) {
		env_init = true;
		const char* env = getenv("FR_MEM_DUMP");
		if(env && *env && atoi(env) != 0)
			FrameRecMemDumpFlag() = true;
	}
	return FrameRecMemDumpFlag();
}

void DumpFrameRecognizerMemoryEvent(const String& event, const String& details) {
	if(!IsFrameRecognizerMemoryDumpEnabled())
		return;
	int64 rss_kb = ReadProcMemKb("VmRSS:");
	int64 hwm_kb = ReadProcMemKb("VmHWM:");
	int64 vms_kb = ReadProcMemKb("VmSize:");
	String msg;
	msg << "[FR_MEM] " << event;
	if(rss_kb >= 0) msg << Format(" rss=%.1f", (double)rss_kb / 1024.0) << "MB";
	if(hwm_kb >= 0) msg << Format(" hwm=%.1f", (double)hwm_kb / 1024.0) << "MB";
	if(vms_kb >= 0) msg << Format(" vms=%.1f", (double)vms_kb / 1024.0) << "MB";
	if(!details.IsEmpty()) msg << " (" << details << ")";
	RLOG(msg);
}

AnchoredSlotRecognizer::AnchoredSlotRecognizer()
{
}

AnchoredSlotRecognizer::~AnchoredSlotRecognizer()
{
}

Image RotateBilinear(const Image& src, double angle_deg) {
	if(src.IsEmpty()) return src;
	int sw = src.GetWidth();
	int sh = src.GetHeight();
	if(fabs(angle_deg) < 1e-6) return src;

	double rad = angle_deg * M_PI / 180.0;
	double cos_a = cos(rad);
	double sin_a = sin(rad);
	double cx = (sw - 1) * 0.5;
	double cy = (sh - 1) * 0.5;

	ImageBuffer dst(sw, sh);
	for(int y = 0; y < sh; y++) {
		RGBA* row = dst[y];
		for(int x = 0; x < sw; x++) {
			double dx = x - cx;
			double dy = y - cy;
			// Inverse mapping: rotate destination point by -angle to find source point
			double sx = cos_a * dx + sin_a * dy + cx;
			double sy = -sin_a * dx + cos_a * dy + cy;
			
			int ix = (int)floor(sx);
			int iy = (int)floor(sy);
			
			if(ix >= 0 && ix < sw - 1 && iy >= 0 && iy < sh - 1) {
				double fx = sx - ix;
				double fy = sy - iy;
				
				RGBA c00 = src[iy][ix];
				RGBA c10 = src[iy][ix+1];
				RGBA c01 = src[iy+1][ix];
				RGBA c11 = src[iy+1][ix+1];
				
				row[x].r = (byte)((1-fx)*(1-fy)*c00.r + fx*(1-fy)*c10.r + (1-fx)*fy*c01.r + fx*fy*c11.r);
				row[x].g = (byte)((1-fx)*(1-fy)*c00.g + fx*(1-fy)*c10.g + (1-fx)*fy*c01.g + fx*fy*c11.g);
				row[x].b = (byte)((1-fx)*(1-fy)*c00.b + fx*(1-fy)*c10.b + (1-fx)*fy*c01.b + fx*fy*c11.b);
				row[x].a = (byte)((1-fx)*(1-fy)*c00.a + fx*(1-fy)*c10.a + (1-fx)*fy*c01.a + fx*fy*c11.a);
			}
			else {
				row[x] = RGBAZero();
			}
		}
	}
	return dst;
}


// Legacy fallback until all projects provide explicit group preprocess config.
static bool UseHighLuminanceBinarizationForHeadLegacy(const String& head_id) {
	if(head_id.IsEmpty())
		return false;
	// Shared grouped heads
	if(head_id == "card_visibility_gate" || head_id == "element#level")
		return true;
	// Suffix-based heads
	if(head_id.EndsWith("#presence") || head_id.EndsWith("#level"))
		return true;
	if(head_id.EndsWith("_is_visible") || head_id.EndsWith("_rank"))
		return true;
	// Legacy bool slot ids
	if(head_id.StartsWith("is_board_card_") || head_id.StartsWith("is_hero_card_"))
		return true;
	return false;
}

bool AnchoredSlotRecognizer::ShouldUseHighLumaHead(const String& head_id) const {
	String mode = head_preprocess_mode_.Get(head_id, String());
	if(mode == "high_luma_bin")
		return true;
	if(mode == "color_raw")
		return false;
	return UseHighLuminanceBinarizationForHeadLegacy(head_id);
}

void AnchoredSlotRecognizer::SetOcrBackend(OCR::OCRBackend b) {
	ocr_backend = b;
}

OCR::OCRBackend AnchoredSlotRecognizer::GetOcrBackend() const {
	return ocr_backend;
}

bool AnchoredSlotRecognizer::Load(const String& annlay_path, const String& annmdl_path) {
	loaded = false;
	sessions.Clear();
	head_preprocess_mode_.Clear();
	DumpFrameRecognizerMemoryEvent("Recognizer.Load.begin", Format("annlay='%s' annmdl='%s'", annlay_path, annmdl_path));

	uint64 t_layout0 = GetTickCount();
	if(!lay.Load(annlay_path)) {
		DumpFrameRecognizerMemoryEvent("Recognizer.Load.fail", "layout load failed");
		return false;
	}
	DumpFrameRecognizerMemoryEvent("Recognizer.Load.layout",
		Format("slots=%d ms=%lld", lay.slots.GetCount(), (long long)(GetTickCount() - t_layout0)));

	AnnMdl mdl;
	String mdl_path = annmdl_path.IsEmpty() ? AnnMdl::PathFromAnnlay(annlay_path) : annmdl_path;
	uint64 t_mdl0 = GetTickCount();
	bool mdl_ok = annmdl_path.IsEmpty() ? mdl.Load(annlay_path) : mdl.LoadPath(annmdl_path);
	int64 total_net_bytes = 0;
	int64 total_blob_bytes = 0;
	for(int i = 0; i < mdl.entries.GetCount(); i++) {
		const AnnMdlEntry& me = mdl.entries[i];
		total_net_bytes += me.net_str.GetCount();
		// V3: account for external weights blob
		if(!me.net_ref.IsEmpty()) {
			String p = IsFullPath(me.net_ref)
			    ? NormalizePath(me.net_ref)
			    : NormalizePath(AppendFileName(GetFileDirectory(mdl_path), me.net_ref));
			if(FileExists(p))
				total_blob_bytes += max<int64>(0, (int64)GetFileLength(p));
		}
		// Legacy V2: external combined blob
		else if(!me.session_ref.IsEmpty()) {
			String p = IsFullPath(me.session_ref)
			    ? NormalizePath(me.session_ref)
			    : NormalizePath(AppendFileName(GetFileDirectory(mdl_path), me.session_ref));
			if(FileExists(p))
				total_blob_bytes += max<int64>(0, (int64)GetFileLength(p));
		}
		else {
			total_blob_bytes += me.session_data.GetCount();
		}
	}
	DumpFrameRecognizerMemoryEvent("Recognizer.Load.model_index",
		Format("ok=%s path='%s' entries=%d net=%.1f blobs=%.1f ms=%lld",
		       mdl_ok ? "yes" : "no", mdl_path, mdl.entries.GetCount(),
		       total_net_bytes / (1024.0 * 1024.0), total_blob_bytes / (1024.0 * 1024.0),
		       (long long)(GetTickCount() - t_mdl0)));

	String sln_dir = GetFileDirectory(annlay_path);
	OCR::OCRConfig ocr_cfg;
	ocr_cfg.backend = ocr_backend;
	ocr_cfg.classifier_model_path = AppendFileName(sln_dir, "classifier.ocrmodel");
	ocr_cfg.splitter_model_path = AppendFileName(sln_dir, "splitter.ocrmodel");

	if(ocr_cfg.backend == OCR::OCR_TESSERACT) {
		ocr_initialized = ocr_engine.Initialize(ocr_cfg);
	} else {
		if(FileExists(ocr_cfg.classifier_model_path))
			ocr_initialized = ocr_engine.Initialize(ocr_cfg);
		else
			ocr_initialized = false;
	}
	DumpFrameRecognizerMemoryEvent("Recognizer.Load.ocr",
		Format("backend=%d initialized=%s", (int)ocr_cfg.backend, ocr_initialized ? "yes" : "no"));

	// Load AnnSln if not already set — look for sidecar .annsln in the same dir.
	if(!sln_set) {
		String sln_path = ForceExt(annlay_path, ".annsln");
		if(!FileExists(sln_path)) {
			String dir = GetFileDirectory(annlay_path);
			String base = GetFileTitle(annlay_path);
			sln_path = AppendFileName(dir, base + ".annsln");
		}
		if(FileExists(sln_path))
			sln.Load(sln_path);
	}
	group_registry_.Build(lay, &sln);

	for(int i = 0; i < lay.slots.GetCount(); i++) {
		AnnLaySlot& slot = lay.slots[i];
		auto LoadHead = [&](const String& head_id, const String& preprocess_key = String()) {
			if(!head_filter.IsEmpty() && head_filter.Find(head_id) < 0)
				return;

			if(sessions.Find(head_id) >= 0)
				return;

			AnnMdlEntry* entry = nullptr;
			for(int ei = 0; ei < mdl.entries.GetCount(); ei++) {
				if(mdl.entries[ei].slot_id == head_id) {
					entry = &mdl.entries[ei];
					break;
				}
			}
			if(!entry || entry->net_str.IsEmpty())
				return;

			DumpFrameRecognizerMemoryEvent("Recognizer.Head.begin",
				Format("head='%s' net=%d net_ref='%s'", head_id, entry->net_str.GetCount(), entry->net_ref));

			One< ::ConvNet::Session >& ses_ptr = sessions.Add(head_id);
			ses_ptr.Create();
			if(!ses_ptr->MakeLayers(entry->net_str)) {
				sessions.Remove(sessions.GetCount() - 1);
				DumpFrameRecognizerMemoryEvent("Recognizer.Head.fail", "MakeLayers failed for " + head_id);
				return;
			}
			if(!entry->net_ref.IsEmpty()) {
				// V3: weights-only blob
				String weights_blob;
				if(!mdl.LoadWeightsData(*entry, weights_blob) || weights_blob.IsEmpty()) {
					sessions.Remove(sessions.GetCount() - 1);
					DumpFrameRecognizerMemoryEvent("Recognizer.Head.fail", "Missing weights data for " + head_id);
					return;
				}
				StringStream ss(weights_blob);
				ses_ptr->SerializeWeights(ss);
			}
			else {
				// V1/V2 legacy: combined session blob
				String session_blob;
				if(!mdl.LoadSessionData(*entry, session_blob) || session_blob.IsEmpty()) {
					sessions.Remove(sessions.GetCount() - 1);
					DumpFrameRecognizerMemoryEvent("Recognizer.Head.fail", "Missing session data for " + head_id);
					return;
				}
				StringStream ss(session_blob);
				ses_ptr->Serialize(ss);
			}

			auto* inp = ses_ptr->GetInput();
			String classes_dump;
			for(int ci = 0; ci < ses_ptr->Data().GetClassCount(); ci++) {
				if(ci) classes_dump << ", ";
				classes_dump << ses_ptr->Data().GetClass(ci);
			}
			DumpFrameRecognizerMemoryEvent("Recognizer.Head.ready",
				Format("head='%s' input=%d*%d*%d classes=[%s]", head_id,
				       inp ? inp->input_width : 0, inp ? inp->input_height : 0, inp ? inp->input_depth : 0,
				       classes_dump));

			entry->net_str.Clear();
			entry->net_data.Clear();
			entry->net_str.Shrink();
			entry->net_data.Shrink();

			String pp = preprocess_key.IsEmpty()
			          ? group_registry_.Preprocess(head_id)
			          : group_registry_.Preprocess(preprocess_key);
			if(pp.IsEmpty())
				pp = group_registry_.Preprocess(head_id);
			if(!pp.IsEmpty())
				head_preprocess_mode_.GetAdd(head_id) = pp;
		};

		if(slot.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
			auto LoadMapped = [&](String suffix) {
				String h_id = slot.id + "#" + suffix;
				String sh = slot.sub_heads.Get(suffix, "");
				if(!sh.IsEmpty()) h_id = sh;
				String configured_gkey = TrimBoth(slot.sub_groups.Get(suffix, ""));
				String gkey = AnchoredSlotClassifier::ResolveCanonicalGroupKey(lay,
				             configured_gkey.IsEmpty()
				             ? AnchoredSlotClassifier::BoolSlotGroupKey(slot.id + "#" + suffix, &lay)
				             : configured_gkey);

				if(!gkey.IsEmpty() && mdl.FindEntry(gkey))
					LoadHead(gkey, gkey);
				else
					LoadHead(h_id, gkey);
			};
			LoadMapped("presence");
			LoadMapped("level");
			LoadMapped("category");
			LoadMapped("zoom");
			LoadMapped("offset");
			LoadMapped("offset_x");
			LoadMapped("offset_y");
		}
		else if(slot.method == ANNLAY_CLASSIFIER_BOOL ||
		        slot.method == ANNLAY_CLASSIFIER_LABEL) {
			if(!mdl_ok)
				LoadHead(slot.id, slot.group);
			else {
				String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(slot.id, &lay);
				if(!gkey.IsEmpty() && mdl.FindEntry(gkey))
					LoadHead(gkey, gkey);
				else
					LoadHead(slot.id, slot.group);
			}
		}
	}

	LoadCvTemplateGroups(annlay_path);

	loaded = true;
	DumpFrameRecognizerMemoryEvent("Recognizer.Load.ready", Format("sessions=%d tmpl_groups=%d", sessions.GetCount(), tmpl_cache_.GetCount()));
	return true;
}

static void ImageToByteMatGray(const Image& img, ByteMat& out) {
	Size sz = img.GetSize();
	out.SetSize(sz.cx, sz.cy, 1);
	for(int y = 0; y < sz.cy; y++) {
		const RGBA* s = img[y];
		for(int x = 0; x < sz.cx; x++) {
			const RGBA& p = s[x];
			out.data[y * sz.cx + x] = (byte)(((int)p.r + (int)p.g + (int)p.b) / 3);
		}
	}
}

static ByteMat ScaleByteMatGray(const ByteMat& src, double scale) {
	if(scale <= 0 || scale == 1.0) return src;
	int nw = max(1, (int)round(src.cols * scale));
	int nh = max(1, (int)round(src.rows * scale));
	ByteMat out;
	out.SetSize(nw, nh, 1);
	for(int y = 0; y < nh; y++) {
		for(int x = 0; x < nw; x++) {
			int sx = min(src.cols - 1, (int)(x / scale));
			int sy = min(src.rows - 1, (int)(y / scale));
			out.data[y * nw + x] = src.data[sy * src.cols + sx];
		}
	}
	return out;
}

static VectorMap<String, ByteMat> LoadTemplateDir(const String& dir, const CvTemplateSubCrop& crop) {
	VectorMap<String, ByteMat> result;
	if(dir.IsEmpty() || !DirectoryExists(dir))
		return result;
	FindFile ff(AppendFileName(dir, "*.png"));
	while(ff) {
		String file_name = ToLower(ff.GetName());
		// Safeguard: allow disabling templates by naming like "*.disabled.png"
		// (or any filename containing ".disabled").
		if(file_name.Find(".disabled") >= 0) {
			ff.Next();
			continue;
		}
		String name = GetFileTitle(ff.GetName());
		Image img = StreamRaster::LoadFileAny(ff.GetPath());
		if(!img.IsEmpty()) {
			ByteMat mat;
			ImageToByteMatGray(img, mat);
			result.Add(name, pick(mat));
		}
		ff.Next();
	}
	return result;
}

void AnchoredSlotRecognizer::LoadCvTemplateGroups(const String& annlay_path) {
	tmpl_cache_.Clear();
	if(sln.cv_template_groups.IsEmpty())
		return;
	tmpl_cache_.SetCount(sln.cv_template_groups.GetCount());

	for(int gi = 0; gi < sln.cv_template_groups.GetCount(); gi++) {
		const CvTemplateGroup& g = sln.cv_template_groups[gi];

		// Load native-resolution templates
		VectorMap<String, ByteMat> raw_a = LoadTemplateDir(g.templates_label_a_dir, g.label_a_crop);
		VectorMap<String, ByteMat> raw_b = LoadTemplateDir(g.templates_label_b_dir, g.label_b_crop);
		// CV templates: grayscale + contrast normalization (linear min-max stretch).
		auto ContrastNormalizeTemplates = [](VectorMap<String, ByteMat>& templates) {
			for(int k = 0; k < templates.GetCount(); k++) {
				ByteMat& m = templates[k];
				int mn = 255, mx = 0;
				for(byte v : m.data) { if(v < mn) mn = v; if(v > mx) mx = v; }
				if(mx <= mn) continue;
				double scale = 255.0 / (mx - mn);
				for(byte& v : m.data)
					v = (byte)minmax((int)round((v - mn) * scale), 0, 255);
			}
		};
		ContrastNormalizeTemplates(raw_a);
		ContrastNormalizeTemplates(raw_b);

		if(raw_a.IsEmpty() && raw_b.IsEmpty())
			continue;

		CvTemplateCache& cache = tmpl_cache_[gi].Create();
		cache.group_index = gi;

		if(raw_a.GetCount() > 0)
			cache.native_size_a = Size(raw_a[0].cols, raw_a[0].rows);
		if(raw_b.GetCount() > 0)
			cache.native_size_b = Size(raw_b[0].cols, raw_b[0].rows);

		int steps = 1; // Testing only one calculated zoom value per user request
		cache.label_a.SetCount(steps);
		cache.label_b.SetCount(steps);

		for(int zi = 0; zi < steps; zi++) {
			for(int k = 0; k < raw_a.GetCount(); k++)
				cache.label_a[zi].Add(raw_a.GetKey(k), raw_a[k]);
			for(int k = 0; k < raw_b.GetCount(); k++)
				cache.label_b[zi].Add(raw_b.GetKey(k), raw_b[k]);
		}

		DumpFrameRecognizerMemoryEvent("CvTemplate.Loaded",
			Format("group='%s' a=%d b=%d zoom_steps=%d", g.name, raw_a.GetCount(), raw_b.GetCount(), steps));
	}
}

// Returns the TemplateMatchMethod enum value for a method name string
static TemplateMatchMethod ParseMatchMethod(const String& s) {
	if(s == "TM_CCORR")        return TM_CCORR;
	if(s == "TM_CCORR_NORMED") return TM_CCORR_NORMED;
	if(s == "TM_SQDIFF")       return TM_SQDIFF;
	if(s == "TM_SQDIFF_NORMED")return TM_SQDIFF_NORMED;
	if(s == "TM_CCOEFF")       return TM_CCOEFF;
	return TM_CCOEFF_NORMED;
}

void AnchoredSlotRecognizer::RecognizeCvTemplateGroup(
	int group_idx, const String& stem,
	const Image& img, double dx, double dy,
	Vector<SlotResult>& out, ProcessingLogRecord* log)
{
	if(group_idx < 0 || group_idx >= sln.cv_template_groups.GetCount()) return;
	if(group_idx >= tmpl_cache_.GetCount()) return;
	if(tmpl_cache_[group_idx].IsEmpty()) return;

	const CvTemplateGroup& g = sln.cv_template_groups[group_idx];
	const CvTemplateCache& cache = *tmpl_cache_[group_idx];

	// Find the slot for this stem to get its anchor
	const AnnLaySlot* slot = lay.FindSlot(stem + "#label_a");
	if(!slot) slot = lay.FindSlot(stem);
	if(!slot) return;

	// Crop the slot region
	Rect bbox = AnchorToRect(slot->anchor, img.GetSize(), dx, dy);
	if(bbox.IsEmpty()) return;
	Rect card_bbox = AnnLayResolveRegionRect(*slot, bbox, "card_region");
	Rect rank_bbox = AnnLayResolveRegionRect(*slot, bbox, "rank_region");
	Rect suit_bbox = AnnLayResolveRegionRect(*slot, bbox, "suit_region");
	if(card_bbox.IsEmpty()) card_bbox = bbox;
	if(rank_bbox.IsEmpty()) rank_bbox = card_bbox;
	if(suit_bbox.IsEmpty()) suit_bbox = card_bbox;
	Size isz = img.GetSize();
	auto ExpandRectByFraction = [&](const Rect& r, double pad_frac) -> Rect {
		if(r.IsEmpty())
			return r;
		int pad_x = max(1, (int)ceil(r.GetWidth() * pad_frac));
		int pad_y = max(1, (int)ceil(r.GetHeight() * pad_frac));
		int x1 = max(0, r.left - pad_x);
		int y1 = max(0, r.top - pad_y);
		int x2 = min(isz.cx, r.right + pad_x);
		int y2 = min(isz.cy, r.bottom + pad_y);
		if(x2 <= x1) x2 = min(isz.cx, x1 + 1);
		if(y2 <= y1) y2 = min(isz.cy, y1 + 1);
		return Rect(x1, y1, x2, y2);
	};
	// LABEL_A level matching needs more context: +100% padding on each side (3x3 area).
	Rect rank_search_bbox = ExpandRectByFraction(rank_bbox, 1.0);
	if(rank_search_bbox.IsEmpty())
		rank_search_bbox = rank_bbox;

	double slot_rot = 0;
	bool fixed_rotation = AnnLayTryGetSubLayoutRotationDeg(*slot, slot_rot);
	int fixed_rot_i = g.slot_rotation_deg.Find(stem);
	if(!fixed_rotation && fixed_rot_i >= 0) {
		fixed_rotation = true;
		slot_rot = g.slot_rotation_deg[fixed_rot_i];
	}
	double fixed_rot_deg = fixed_rotation ? slot_rot : 0.0;
	bool do_rotation = g.rotation_slots.Find(stem) >= 0;
	int rot_steps = fixed_rotation ? 1 : (do_rotation ? max(1, g.rotation_steps) : 1);

	// Calculate actual zoom size based on current level region width vs native template width
	double zoom = 1.0;
	if(cache.native_size_a.cx > 0)
		zoom = (double)rank_bbox.GetWidth() / (double)cache.native_size_a.cx;
	int zoom_steps = 1;

	TemplateMatchMethod method = ParseMatchMethod(g.match_method);
	bool sqdiff = (method == TM_SQDIFF || method == TM_SQDIFF_NORMED);

	double best_score = sqdiff ? 1e18 : -1e18;
	String best_a, best_b;
	double best_score_a = 0, best_score_b = 0;
	double best_zoom = zoom, best_rot = 0.0;
	int best_zi = 0, best_ri = 0;

	uint64 t0 = GetTickCount();
	auto BuildScaledRotatedCrop = [&](const Rect& src_rect, double zoom, double rot) -> Image {
		double inv_zoom = 1.0 / zoom;
		Size scaled_sz(max(1, (int)round(src_rect.GetWidth() * inv_zoom)),
		               max(1, (int)round(src_rect.GetHeight() * inv_zoom)));
		Image scaled = CropAndScale(img, src_rect, scaled_sz);
		if(scaled.IsEmpty())
			return Image();
		if(fabs(rot) > 0.01)
			scaled = RotateBilinear(scaled, -rot);
		return scaled;
	};

	for(int zi = 0; zi < zoom_steps; zi++) {
		for(int ri = 0; ri < rot_steps; ri++) {
			double rot = fixed_rotation ? fixed_rot_deg : 0;
			if(!fixed_rotation && do_rotation && rot_steps > 1)
				rot = g.rotation_min_deg + (double)ri / (rot_steps - 1) * (g.rotation_max_deg - g.rotation_min_deg);

			Image rank_crop = BuildScaledRotatedCrop(rank_search_bbox, zoom, rot);
			Image suit_crop = BuildScaledRotatedCrop(suit_bbox, zoom, rot);
			if(rank_crop.IsEmpty() || suit_crop.IsEmpty())
				continue;
			ByteMat rank_mat, suit_mat;
			ImageToByteMatGray(AnchoredSlotClassifier::LinearContrastStretching(rank_crop), rank_mat);
			ImageToByteMatGray(AnchoredSlotClassifier::LinearContrastStretching(suit_crop), suit_mat);

			// Match all label_a templates
			double score_a = sqdiff ? 1e18 : -1e18;
			String cls_a;
			const VectorMap<String, ByteMat>& tmpl_a = cache.label_a[zi];
			for(int k = 0; k < tmpl_a.GetCount(); k++) {
				const ByteMat& tmpl = tmpl_a[k];
				if(rank_mat.cols < tmpl.cols || rank_mat.rows < tmpl.rows) continue;
				FloatMat res;
				MatchTemplate(rank_mat, tmpl, res, method);
				double mn, mx;
				MinMaxLoc(res, &mn, &mx, nullptr, nullptr);
				double s = sqdiff ? mn : mx;
				if(sqdiff ? (s < score_a) : (s > score_a)) {
					score_a = s;
					cls_a = tmpl_a.GetKey(k);
				}
			}

			// Match all label_b templates
			double score_b = sqdiff ? 1e18 : -1e18;
			String cls_b;
			const VectorMap<String, ByteMat>& tmpl_b = cache.label_b[zi];
			for(int k = 0; k < tmpl_b.GetCount(); k++) {
				const ByteMat& tmpl = tmpl_b[k];
				if(suit_mat.cols < tmpl.cols || suit_mat.rows < tmpl.rows) continue;
				FloatMat res;
				MatchTemplate(suit_mat, tmpl, res, method);
				double mn, mx;
				MinMaxLoc(res, &mn, &mx, nullptr, nullptr);
				double s = sqdiff ? mn : mx;
				if(sqdiff ? (s < score_b) : (s > score_b)) {
					score_b = s;
					cls_b = tmpl_b.GetKey(k);
				}
			}

			// Level is primary: select best zoom/rot by level score alone.
			// Category CV score is secondary and only used when category templates matched.
			bool rank_better = sqdiff ? (score_a < best_score_a || best_a.IsEmpty())
			                          : (score_a > best_score_a || best_a.IsEmpty());
			if(rank_better && !cls_a.IsEmpty()) {
				best_score_a = score_a;
				best_a = cls_a;
				best_zoom = zoom;
				best_rot = rot;
				best_zi = zi;
				best_ri = ri;
				// Update combined and category only when level improves
				best_score_b = score_b;
				best_b = cls_b;
				best_score = sqdiff ? max(score_a, score_b) : min(score_a, score_b);
			}
		}
	}

	double t_ms = (double)(GetTickCount() - t0);

	// Collect CV debug data for inspector (only when log is present, debug path only)
	Image cv_response_map_img;
	Image cv_input_crop_img;
	Vector<CvCandidateRecord> cv_candidates;
	if(log) {
		Image rank_crop_debug = BuildScaledRotatedCrop(rank_search_bbox, best_zoom, best_rot);
		Image suit_crop_debug = BuildScaledRotatedCrop(suit_bbox, best_zoom, best_rot);
		if(!rank_crop_debug.IsEmpty()) {
			rank_crop_debug = AnchoredSlotClassifier::LinearContrastStretching(rank_crop_debug);
			suit_crop_debug = AnchoredSlotClassifier::LinearContrastStretching(suit_crop_debug);
			cv_input_crop_img = rank_crop_debug;  // save for method-override replay
			ByteMat rank_mat_debug;
			ImageToByteMatGray(rank_crop_debug, rank_mat_debug);
			const VectorMap<String, ByteMat>& tmpl_a = cache.label_a[best_zi];
			FloatMat best_res;
			for(int k = 0; k < tmpl_a.GetCount(); k++) {
				const ByteMat& tmpl = tmpl_a[k];
				if(rank_mat_debug.cols < tmpl.cols || rank_mat_debug.rows < tmpl.rows) continue;
				FloatMat res;
				MatchTemplate(rank_mat_debug, tmpl, res, method);
				double mn, mx;
				MinMaxLoc(res, &mn, &mx, nullptr, nullptr);
				double s = sqdiff ? mn : mx;
				CvCandidateRecord cr;
				cr.class_name = tmpl_a.GetKey(k);
				cr.score = s;
				cr.zoom = best_zoom;
				cr.rot = best_rot;
				{
					ImageBuffer ib(tmpl.cols, tmpl.rows);
					for(int y = 0; y < tmpl.rows; y++) {
						RGBA* d = ib[y];
						for(int x = 0; x < tmpl.cols; x++) {
							byte v = tmpl.data[y * tmpl.cols + x];
							d[x] = {v, v, v, 255};
						}
					}
					cr.crop_image = ib;
				}
				if(cr.class_name == best_a)
					best_res = pick(res);
				cv_candidates.Add(pick(cr));
			}
			Sort(cv_candidates, [&](const CvCandidateRecord& a, const CvCandidateRecord& b) {
				return sqdiff ? (a.score < b.score) : (a.score > b.score);
			});
			cv_response_map_img = FloatMatToGrayImage(best_res, sqdiff);
		}
	}

	bool present = false;
	if(!sqdiff) {
		present = best_score_a >= g.match_threshold && best_score_b >= g.match_threshold;
	} else {
		present = best_score_a <= (1.0 - g.match_threshold) && best_score_b <= (1.0 - g.match_threshold);
	}

	String head_p, head_s, head_r;
	auto GetMappedSession = [&](const String& suffix, String* resolved_head = nullptr) -> int {
		String h_id = stem + "#" + suffix;
		String sh = slot->sub_heads.Get(suffix, "");
		if(!sh.IsEmpty()) h_id = sh;

		String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(stem + "#" + suffix, &lay);
		int k = -1;
		if(!gkey.IsEmpty()) k = sessions.Find(gkey);
		if(k < 0) k = sessions.Find(h_id);
		if(resolved_head) {
			if(k >= 0) *resolved_head = sessions.GetKey(k);
			else *resolved_head = (!gkey.IsEmpty() ? gkey : h_id);
		}
		return k;
	};
	int k_p = GetMappedSession("presence", &head_p);
	int k_s = GetMappedSession("category", &head_s);
	int k_r = GetMappedSession("level", &head_r);

	// Hybrid mode: keep level from CV template match, but prefer NN for category/presence
	// when those heads are available.
	Image nn_card_crop = BuildScaledRotatedCrop(card_bbox, best_zoom, best_rot);
	Image nn_suit_crop = BuildScaledRotatedCrop(suit_bbox, best_zoom, best_rot);

	if(!nn_card_crop.IsEmpty() && k_p >= 0) {
		nn_card_crop = HighLuminanceThresholdBinarization(nn_card_crop);
		Size gate_sz(sessions[k_p]->GetInput()->input_width, sessions[k_p]->GetInput()->input_height);
		if(nn_card_crop.GetSize() != gate_sz)
			nn_card_crop = Rescale(nn_card_crop, gate_sz);
		bool gate_gray = (sessions[k_p]->GetInput()->input_depth == 1);
		Vector<double> inp_p = ImageToVec(nn_card_crop, gate_gray, gate_sz, false);
		Vector<double> scores_p = sessions[k_p]->Predict(inp_p);
		if(scores_p.GetCount() >= 2) {
			double p_true = scores_p[1];
			present = (p_true >= slot->presence_threshold);
		}
	}
	if(!nn_suit_crop.IsEmpty() && k_s >= 0 && present) {
		Size suit_sz(sessions[k_s]->GetInput()->input_width, sessions[k_s]->GetInput()->input_height);
		if(nn_suit_crop.GetSize() != suit_sz)
			nn_suit_crop = Rescale(nn_suit_crop, suit_sz);
		bool suit_gray = (sessions[k_s]->GetInput()->input_depth == 1);
		Vector<double> inp_s = ImageToVec(nn_suit_crop, suit_gray, suit_sz, false);
		Vector<double> scores_s = sessions[k_s]->Predict(inp_s);
		if(scores_s.GetCount() > 0) {
			int best = 0;
			for(int i = 1; i < scores_s.GetCount(); i++)
				if(scores_s[i] > scores_s[best]) best = i;
			best_b = SafeGetClass(*sessions[k_s], best);
			best_score_b = scores_s[best];
		}
	}
	best_a = NormalizeRankCode(best_a);
	best_b = NormalizeSuitCode(best_b);

	String note = Format("score=%.3f zoom=%.2f rot=%.1f", best_score, best_zoom, best_rot);

	if(log) {
		// RECOGNIZE false is a valid classification outcome (not a processing error).
		log->AddStep("VISIBLE", 0, "OK",
		             present ? "true" : "false",
		             stem + "#presence", head_p, k_p >= 0);
		{
			ProcessingStepRecord& ps = log->steps.Top();
			ps.candidate_bbox = card_bbox;
			ps.crop_size = slot->crop_size;
			ps.angle = best_rot;
			ps.is_grayscale = true;
			ps.is_equalized = false;
			if(!head_p.IsEmpty() && head_p != stem + "#presence")
				ps.requested_head_id = stem + "#presence";
		}

		if(present) {
			log->AddStep("LEVEL", t_ms, "OK",
			             best_a, stem + "#level", head_r, false, note + " score_a=" + Format("%.3f", best_score_a));
			{
				ProcessingStepRecord& ps = log->steps.Top();
				ps.candidate_bbox = rank_search_bbox;
				ps.crop_size = slot->crop_size;
				ps.angle = best_rot;
				ps.is_grayscale = true;
				ps.is_equalized = false;
				ps.cv_response_map = cv_response_map_img;
				ps.cv_input_crop = cv_input_crop_img;
				ps.cv_match_method = g.match_method;
				ps.cv_candidates <<= cv_candidates;
				if(!head_r.IsEmpty() && head_r != stem + "#level")
					ps.requested_head_id = stem + "#level";
			}
			log->AddStep("CATEGORY", 0, "OK",
			             best_b, stem + "#category", head_s, k_s >= 0, "score_b=" + Format("%.3f", best_score_b));
			{
				ProcessingStepRecord& ps = log->steps.Top();
				ps.candidate_bbox = suit_bbox;
				Size suit_sz = (k_s >= 0) ? Size(sessions[k_s]->GetInput()->input_width, sessions[k_s]->GetInput()->input_height) : Size(22, 22);
				ps.crop_size = suit_sz;
				ps.angle = best_rot;
				ps.is_grayscale = (k_s >= 0) ? (sessions[k_s]->GetInput()->input_depth == 1) : false;
				ps.is_equalized = false;
				if(!head_s.IsEmpty() && head_s != stem + "#category")
					ps.requested_head_id = stem + "#category";
			}
		} else {
			log->AddStep("LEVEL", 0, "SKIPPED", "(skipped: 'is' gate)", stem + "#level");
			log->AddStep("CATEGORY", 0, "SKIPPED", "(skipped: 'is' gate)", stem + "#category");
		}
	}

	// Emit results
	{
		SlotResult r;
		r.slot_id = stem + "#label_a";
		r.method = "cv_template_match";
		r.raw_text = present ? best_a : String();
		r.top_class = r.raw_text;
		r.confidence = present ? best_score_a : 0.0;
		r.t_ms = t_ms;
		r.details = note;
		out.Add(pick(r));
	}
	{
		SlotResult r;
		r.slot_id = stem + "#label_b";
		r.method = "cv_template_match";
		r.raw_text = present ? best_b : String();
		r.top_class = r.raw_text;
		r.confidence = present ? best_score_b : 0.0;
		r.details = note;
		out.Add(pick(r));
	}
	{
		SlotResult r;
		r.slot_id = stem + "#present";
		r.method = "cv_template_match";
		r.raw_text = present ? "true" : "false";
		r.top_class = r.raw_text;
		r.confidence = present ? best_score : 0.0;
		out.Add(pick(r));
	}
}

Rect AnchoredSlotRecognizer::AnchorToRect(const AnnLayAnchor& anchor, Size img_size) const {
	return AnchorToRect(anchor, img_size, 0, 0);
}

Rect AnchoredSlotRecognizer::AnchorToRect(const AnnLayAnchor& anchor, Size img_size, double dx, double dy) const {
	int w = (int)round(anchor.w * img_size.cx);
	int h = (int)round(anchor.h * img_size.cy);
	int cx = (int)round((anchor.cx + dx / img_size.cx) * img_size.cx);
	int cy = (int)round((anchor.cy + dy / img_size.cy) * img_size.cy);
	return RectC(cx - w / 2, cy - h / 2, w, h);
}

Image AnchoredSlotRecognizer::CropAndScale(const Image& img, Rect r, Size target_size) const {
	if(r.IsEmpty() || target_size.cx <= 0 || target_size.cy <= 0)
		return Image();

	Image crop = Crop(img, r);
	if(crop.IsEmpty())
		return Image();

	if(crop.GetSize() == target_size)
		return crop;

	return Rescale(crop, target_size);
}

Image AnchoredSlotRecognizer::CropRotateScale(const Image& img, const AnnLayAnchor& anchor, Size target_size, double dx, double dy, double angle) const {
	if(img.IsEmpty() || target_size.cx <= 0 || target_size.cy <= 0) return Image();
	if(fabs(angle) < 0.01) return CropAndScale(img, AnchorToRect(anchor, img.GetSize(), dx, dy), target_size);

	Size isz = img.GetSize();
	// Use 1.5 expansion for rotation
	double exp = 1.5;
	int w = (int)round(anchor.w * isz.cx * exp);
	int h = (int)round(anchor.h * isz.cy * exp);
	int cx = (int)round((anchor.cx + dx / isz.cx) * isz.cx);
	int cy = (int)round((anchor.cy + dy / isz.cy) * isz.cy);
	
	Image patch = Crop(img, RectC(cx - w / 2, cy - h / 2, w, h));
	if(patch.IsEmpty()) return Image();
	
	Image rotated = RotateBilinear(patch, -angle);
	
	Size rsz = rotated.GetSize();
	int rcx = rsz.cx / 2;
	int rcy = rsz.cy / 2;
	int fw = (int)round(anchor.w * isz.cx);
	int fh = (int)round(anchor.h * isz.cy);
	
	Image sub = Crop(rotated, RectC(rcx - fw / 2, rcy - fh / 2, fw, fh));
	if(sub.IsEmpty()) return Image();
	return Rescale(sub, target_size);
}

Vector<double> AnchoredSlotRecognizer::ImageToVec(const Image& crop, bool grayscale, Size sz, bool equalize) const {
	Image out = equalize ? AnchoredSlotClassifier::LinearContrastStretching(crop) : crop;
	Vector<double> v;
	v.SetCount(sz.cx * sz.cy * (grayscale ? 1 : 3));
	int n = 0;
	for(int y = 0; y < sz.cy; y++) {
		for(int x = 0; x < sz.cx; x++) {
			Color c = out[y][x];
			if(grayscale) {
				v[n++] = Grayscale(c) / 255.0;
			}
			else {
				v[n++] = c.GetR() / 255.0;
				v[n++] = c.GetG() / 255.0;
				v[n++] = c.GetB() / 255.0;
			}
		}
	}
	return v;
}

SlotResult AnchoredSlotRecognizer::RecognizeClassifierWithOffset(const AnnLaySlot& slot, const Image& img,
                                                                 double dx, double dy, ProcessingLogRecord* log) {
	SlotResult r;
	r.slot_id = slot.id;
	r.method = AnnLayMethodToString(slot.method);
	r.offset_dx = dx;
	r.offset_dy = dy;

	String head_id = slot.id;
	String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(slot.id, &lay);
	if(!gkey.IsEmpty()) head_id = gkey;

	int k = sessions.Find(head_id);
	if(k < 0) {
		r.details << "Session not found for " << head_id << "\n";
		return r;
	}

	r.details << "NN Head: " << head_id << " (CLASSIFIER)\n";

	double angle = AnnLayGetSlotRotationDeg(slot, slot.id);
	AnnLayAnchor crop_anchor = slot.method == ANNLAY_CLASSIFIER_BOOL ? ExpandAnchor(slot.anchor, lay.bbox_expand) : slot.anchor;
	Image crop = CropRotateScale(img, crop_anchor, slot.crop_size, dx, dy, angle);
	if(crop.IsEmpty()) {
		r.details << "Empty crop\n";
		return r;
	}

	r.pixel_bbox = AnchorToRect(crop_anchor, img.GetSize(), dx, dy);
	bool gray = (slot.color_mode != "color");
	Vector<double> input = ImageToVec(crop, gray, slot.crop_size);

	Vector<double> scores = sessions[k]->Predict(input);
	int best = 0;
	for(int i = 1; i < scores.GetCount(); i++) {
		if(scores[i] > scores[best])
			best = i;
	}

	r.class_index = best;
	r.confidence = scores[best];
	if(best < sessions[k]->Data().GetClassCount())
		r.top_class = sessions[k]->Data().GetClass(best);
	else if(best < slot.classes.GetCount())
		r.top_class = slot.classes[best];
	else
		r.top_class = AsString(best);

	if(log) {
		if(r.confidence < 0.6) {
			log->watchlist << "  [WEAK] " << slot.id << ": Low confidence (" << Format("%.2f", r.confidence) << ") class: " << r.top_class << "\n";
		}
	}

	// Phase 6: override best class if layout specifies a custom presence_threshold for bool
	if(slot.method == ANNLAY_CLASSIFIER_BOOL && scores.GetCount() == 2) {
		double thr = slot.presence_threshold;
		if(thr > 0 && thr != 0.5) {
			// scores[1] is "true"
			if(scores[1] >= thr) {
				r.class_index = 1;
				r.confidence = scores[1];
				r.top_class = "true";
			}
			else {
				r.class_index = 0;
				r.confidence = scores[0];
				r.top_class = "false";
			}
		}
	}

	return r;
}

SlotResult AnchoredSlotRecognizer::RecognizeVariableClassifier(const AnnLaySlot& slot, const Image& img,
                                                                 double dx, double dy, ProcessingLogRecord* log) {
	SlotResult r;
	r.slot_id = slot.id;
	r.method = AnnLayMethodToString(slot.method);
	
	String head_id = slot.id;
	String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(slot.id, &lay);
	if(!gkey.IsEmpty()) head_id = gkey;

	int k = sessions.Find(head_id);
	if(k < 0) {
		r.details << "Session not found for " << head_id << "\n";
		return r;
	}

	r.details << "NN Head: " << head_id << " (VARIABLE CLASSIFIER, " << slot.anchor_candidates.GetCount() << " cands)\n";

	int best_cand = -1;
	double best_score = -1.0;
	Rect best_bbox;

	bool gray = (slot.color_mode != "color");
	double angle = AnnLayGetSlotRotationDeg(slot, slot.id);

	for(int i = 0; i < slot.anchor_candidates.GetCount(); i++) {
		uint64 t_cand0 = GetTickCount();
		AnnLayAnchor crop_anchor = slot.method == ANNLAY_CLASSIFIER_BOOL ? ExpandAnchor(slot.anchor_candidates[i], lay.bbox_expand) : slot.anchor_candidates[i];
		Image crop = CropRotateScale(img, crop_anchor, slot.crop_size, dx, dy, angle);
		if(crop.IsEmpty())
			continue;

		Rect bbox = AnchorToRect(crop_anchor, img.GetSize(), dx, dy);

		Vector<double> input = ImageToVec(crop, gray, slot.crop_size);
		Vector<double> scores = sessions[k]->Predict(input);

		// For bool gates, we assume class 1 is "true"
		double score = scores.GetCount() > 1 ? scores[1] : scores[0];

		if(log) {
			ProcessingStepRecord& ps = log->steps.Add();
			ps.step_id = log->next_step_id++;
			ps.stage = "CANDIDATE";
			ps.slot_id = slot.id;
			ps.head_id = head_id;
			ps.is_nn_step = true;
			ps.duration_ms = (double)(GetTickCount() - t_cand0);
			ps.status = "OK";
			ps.note = Format("Cand #%d: %.1f%% (%s)", i + 1, score * 100.0, score > 0.5 ? "true" : "false");
			ps.detailed_note << "Candidate index: " << i << "\n"
			                << "BBox: " << bbox << "\n"
			                << "Score (true): " << score << "\n";
			ps.candidate_bbox = bbox;
			ps.crop_size = slot.crop_size;
			ps.is_grayscale = gray;
			ps.is_equalized = false;
			ps.angle = angle;
		}

		if(score > best_score) {
			best_score = score;
			best_cand = i;
			best_bbox = bbox;
		}
	}

	if(best_cand >= 0) {
		r.class_index = best_score > 0.5 ? 1 : 0;
		r.confidence = best_score;
		r.top_class = r.class_index == 1 ? "true" : "false";
		r.pixel_bbox = best_bbox;
		r.winner_cand_index = best_cand;

		if(log && r.confidence < 0.6) {
			log->watchlist << "  [WEAK] " << slot.id << " (VAR): Low confidence (" << Format("%.2f", r.confidence) << ") class: " << r.top_class << "\n";
		}
	}

	return r;
}

void AnchoredSlotRecognizer::PredictCrop(const String& head_id, const Image& img, Rect bbox, Size crop_size, bool grayscale, double angle, bool equalize) {
	Mutex::Lock __(recognize_lock);
	int k = sessions.Find(head_id);
	if(k < 0) return;
	
	Image crop;
	if(fabs(angle) > 0.01) {
		// Reconstruct original anchor relative coordinates from bbox and img size
		Size isz = img.GetSize();
		AnnLayAnchor anchor;
		anchor.cx = (double)(bbox.left + bbox.right) / 2.0 / isz.cx;
		anchor.cy = (double)(bbox.top + bbox.bottom) / 2.0 / isz.cy;
		anchor.w = (double)bbox.GetWidth() / isz.cx;
		anchor.h = (double)bbox.GetHeight() / isz.cy;
		
		crop = CropRotateScale(img, anchor, crop_size, 0, 0, angle);
	}
	else {
		crop = CropAndScale(img, bbox, crop_size);
	}
	
	if(crop.IsEmpty()) return;
	if(ShouldUseHighLumaHead(head_id)) {
		crop = HighLuminanceThresholdBinarization(crop);
		equalize = false; // explicitly disable linear contrast stretching for these heads
	}

	Vector<double> input = ImageToVec(crop, grayscale, crop_size, equalize);
	sessions[k]->Predict(input);
}

void AnchoredSlotRecognizer::ComputeOcrOffset(const Image& img, double& out_dx, double& out_dy, ProcessingLogRecord* log) {
	uint64 t0 = GetTickCount();
	out_dx = 0;
	out_dy = 0;
	
	bool cache_hit = (cached_image_serial == img.GetSerialId() && cached_image_serial != -1 && !ocr_cache_.IsEmpty());
	if(!cache_hit) {
		ocr_cache_.Clear();
		cached_image_serial = img.GetSerialId();
	}

	String source = cache_hit ? "cache_hit" : "computed";
	int ocr_count = 0;
	int hit_count = 0;
	String items_dump;

	Size isz = img.GetSize();
	if(ocr_eager_) {
		for(int i = 0; i < lay.slots.GetCount(); i++) {
			const AnnLaySlot& slot = lay.slots[i];
			if(slot.method != ANNLAY_OCR_TEXT)
				continue;

			ocr_count++;
			int q = ocr_cache_.Find(slot.id);
			if(q >= 0) {
				hit_count++;
				continue;
			}

			String error;
			double conf = 0;
			Rect out_bbox;
			OCR::OCRPreprocessMode mode = OCR::OCR_PREPROCESS_GRAYSCALE;
			if(!slot.ocr_preprocess_mode.IsEmpty())
				mode = OCR::StringToOCRPreprocessMode(slot.ocr_preprocess_mode);

			auto GetFallbackMode = [](OCR::OCRPreprocessMode m) {
				switch(m) {
					case OCR::OCR_PREPROCESS_GRAYSCALE:                              return OCR::OCR_PREPROCESS_GRAYSCALE_INVERSE;
					case OCR::OCR_PREPROCESS_GRAYSCALE_INVERSE:                    return OCR::OCR_PREPROCESS_GRAYSCALE;
					case OCR::OCR_PREPROCESS_GRAYSCALE_INVERSE_CONTRAST_STRETCH:   return OCR::OCR_PREPROCESS_GRAYSCALE;
					case OCR::OCR_PREPROCESS_ADAPTIVE_THRESHOLD:                   return OCR::OCR_PREPROCESS_ADAPTIVE_THRESHOLD_INVERSE;
					case OCR::OCR_PREPROCESS_ADAPTIVE_THRESHOLD_INVERSE:   return OCR::OCR_PREPROCESS_ADAPTIVE_THRESHOLD;
					case OCR::OCR_PREPROCESS_OTSU:                         return OCR::OCR_PREPROCESS_OTSU_INVERSE;
					case OCR::OCR_PREPROCESS_OTSU_INVERSE:                 return OCR::OCR_PREPROCESS_OTSU;
					default:                                               return OCR::OCR_PREPROCESS_GRAYSCALE_INVERSE;
				}
			};

			Rect crop_bbox = AnchorToRect(slot.anchor, isz) & Rect(img.GetSize());
			Image crop = crop_bbox.IsEmpty() ? Image() : Crop(img, crop_bbox);
			String text = RunTesseract(crop, slot.id, error, conf, out_bbox, mode, slot.ocr_psm, slot.ocr_whitelist, slot.ocr_blacklist);

			if(text.IsEmpty()) {
				OCR::OCRPreprocessMode fallback = GetFallbackMode(mode);
				double fb_conf = 0;
				Rect fb_bbox;
				String fb_error;
				String fb_text = RunTesseract(crop, slot.id, fb_error, fb_conf, fb_bbox, fallback, slot.ocr_psm, slot.ocr_whitelist, slot.ocr_blacklist);
				if(!fb_text.IsEmpty()) {
					text = fb_text;
					conf = fb_conf;
					out_bbox = fb_bbox;
				}
			}

			SlotResult& r = ocr_cache_.Add(slot.id);
			r.slot_id = slot.id;
			r.method = "ocr_text";
			r.raw_text = text;
			r.confidence = conf;
			r.pixel_bbox = out_bbox;
		}
	}
	else {
		for(int i = 0; i < lay.slots.GetCount(); i++) {
			const AnnLaySlot& slot = lay.slots[i];
			if(slot.method != ANNLAY_OCR_TEXT) continue;
			ocr_count++;
			if(ocr_cache_.Find(slot.id) >= 0) hit_count++;
		}
	}

	if(ocr_initialized && hit_count < ocr_count) {
		// Attempt alignment based on pot or table elements if needed
		// For now, we assume global offset 0 if not explicitly calculated
	}

	if(log) {
		ProcessingStepRecord& s = log->steps.Add();
		s.step_id = log->next_step_id++;
		s.stage = "ComputeOcrOffset";
		s.duration_ms = (GetTickCount() - t0);
		s.status = "OK";
		s.note = Format("hit=%d/%d source=%s", hit_count, ocr_count, source);
	}
}

String AnchoredSlotRecognizer::RunTesseract(const Image& crop, const String& slot_id, String& error_reason, double& confidence, Rect& out_bbox, OCR::OCRPreprocessMode mode, int psm, const String& whitelist, const String& blacklist) {
	if(!ocr_initialized) {
		error_reason = "OCR not initialized";
		return String();
	}

	OCR::OCRConfig cfg = ocr_engine.GetConfig();
	cfg.mode = mode;
	if(psm >= 0) cfg.tesseract_psm = psm;
	if(!whitelist.IsEmpty()) cfg.ocr_whitelist = whitelist;
	if(!blacklist.IsEmpty()) cfg.ocr_blacklist = blacklist;
	ocr_engine.Initialize(cfg);

	OCR::OCRResult res = ocr_engine.RecognizePage(crop);
	confidence = res.avg_confidence;
	if(res.full_text.IsEmpty()) {
		error_reason = "No text found";
		return String();
	}
	return res.full_text;
}

void AnchoredSlotRecognizer::ComputeLocalOffset(const AnnLaySlot& slot, const Image& img, double global_dx, double global_dy, double& out_dx, double& out_dy) {
	out_dx = global_dx;
	out_dy = global_dy;
}

void AnchoredSlotRecognizer::ApplyBoolGates(Vector<SlotResult>& results) {
	for(int i = 0; i < results.GetCount(); i++) {
		SlotResult& r = results[i];
		String gate_id;
		if(const AnnLaySlot* slot = lay.FindSlot(r.slot_id))
			gate_id = TrimBoth(slot->gate);
		
		if(gate_id.IsEmpty() && r.slot_id.EndsWith("_bet")) {
			// seatN_bet -> seatN_is_bet_chip
			// seat_N_bet -> seat_N_is_bet_chip
			gate_id = r.slot_id.Left(r.slot_id.GetCount() - 4) + "_is_bet_chip";
		}
		else if(gate_id.IsEmpty() && r.slot_id == "previous_streets_pot") {
			gate_id = "is_previous_streets_pot";
		}
		else if(gate_id.IsEmpty() && r.slot_id == "side_pot") {
			gate_id = "is_side_pot";
		}
		else if(gate_id.IsEmpty() && r.slot_id == "action_call") {
			gate_id = "is_action_call";
		}
		else if(gate_id.IsEmpty() && r.slot_id == "action_fold") {
			gate_id = "is_action_fold";
		}
		else if(gate_id.IsEmpty() && r.slot_id == "action_raise") {
			gate_id = "is_action_raise";
		}

		if(!gate_id.IsEmpty()) {
			r.gate_slot_id = gate_id;
			int gate_idx = -1;
			for(int j = 0; j < results.GetCount(); j++) {
				if(results[j].slot_id == gate_id) {
					gate_idx = j;
					break;
				}
			}

			bool clear = false;
			String reason;
			String status = "pass";

			if(gate_idx < 0) {
				if(bool_gate_policy == BOOL_GATE_STRICT) {
					clear = true;
					reason = "missing gate slot: " + gate_id;
					status = "blocked_missing";
				}
				else {
					status = "pass"; // permissive: fail-open
				}
			}
			else {
				const SlotResult& gate = results[gate_idx];
				if(gate.method == "classifier_bool") {
					bool is_true = (gate.top_class == "true" || gate.class_index == 1);
					if(!is_true) {
						clear = true;
						reason = gate_id + " is false";
						status = "blocked_false";
					}
				}
				else {
					if(bool_gate_policy == BOOL_GATE_STRICT) {
						clear = true;
						reason = "gate " + gate_id + " is not classifier_bool (method=" + gate.method + ")";
						status = "blocked_invalid";
					}
				}
			}

			r.gate_status = status;
			if(clear) {
				if(!r.top_class.IsEmpty() || !r.raw_text.IsEmpty()) {
					Cerr() << "[AnchoredSlotRecognizer] Bool gate (" << (bool_gate_policy == BOOL_GATE_STRICT ? "STRICT" : "permissive")
					       << "): CLEARING " << r.slot_id << " because " << reason << "\n";
					r.top_class = "";
					r.raw_text = "";
					r.confidence = 0.0;
				}
			}
		}
	}
}

SlotResult AnchoredSlotRecognizer::RecognizeCompositeElement(const AnnLaySlot& slot, const Image& img, double dx, double dy, ProcessingLogRecord* log) {
	SlotResult r;
	r.slot_id = slot.id;
	r.method = "composite_card";

	auto GetMappedSession = [&](String suffix, String* resolved_head = nullptr) -> int {
		String h_id = slot.id + "#" + suffix;
		String sh = slot.sub_heads.Get(suffix, "");
		if(!sh.IsEmpty()) h_id = sh;
		
		String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(slot.id + "#" + suffix, &lay);
		int k = -1;
		if(!gkey.IsEmpty()) k = sessions.Find(gkey);
		if(k < 0) k = sessions.Find(h_id);
		if(resolved_head) {
			if(k >= 0) *resolved_head = sessions.GetKey(k);
			else *resolved_head = (!gkey.IsEmpty() ? gkey : h_id);
		}
		return k;
	};

	String head_p, head_r, head_s, head_ox, head_oy, head_z;
	int k_p = GetMappedSession("presence", &head_p);
	int k_r = GetMappedSession("level", &head_r);
	int k_s = GetMappedSession("category", &head_s);
	int k_ox = GetMappedSession("offset_x", &head_ox);
	int k_oy = GetMappedSession("offset_y", &head_oy);
	int k_z = GetMappedSession("zoom", &head_z);

	double lx = 0, ly = 0;
	double cur_zoom = 1.0;
	uint64 t0 = GetTickCount();

	Rect full_bbox, card_bbox, rank_bbox, suit_bbox;

	double angle = AnnLayGetSlotRotationDeg(slot, slot.id);
	Size isz = img.GetSize();

	// 1. ZOOM (MANDATORY)
	if(k_z >= 0) {
		uint64 t_zoom = GetTickCount();
		Size suit_probe_sz(slot.crop_size.cx, slot.crop_size.cy / 2);
		Image full_probe = CropRotateScale(img, slot.anchor, slot.crop_size, dx, dy, angle);
		Image suit_crop;
		Rect suit_bbox_probe = AnnLayResolveRegionRect(slot, AnchorToRect(slot.anchor, isz, dx, dy), "suit_region");
		if(!full_probe.IsEmpty()) {
			Rect suit_local = AnnLayResolveRegionRect(slot, full_probe.GetSize(), "suit_region");
			suit_crop = Crop(full_probe, suit_local);
			if(!suit_crop.IsEmpty() && suit_crop.GetSize() != suit_probe_sz)
				suit_crop = Rescale(suit_crop, suit_probe_sz);
		}
		if(!suit_crop.IsEmpty()) {
			bool zoom_gray = (sessions[k_z]->GetInput()->input_depth == 1);
			Vector<double> inp = ImageToVec(suit_crop, zoom_gray, suit_probe_sz);
			Vector<double> res = sessions[k_z]->Predict(inp);
			if(res.GetCount() > 0) {
				int best = 0;
				for(int i = 1; i < res.GetCount(); i++) if(res[i] > res[best]) best = i;
				
				String cls = SafeGetClass(*sessions[k_z], best);
				if(cls.StartsWith("z"))
					cur_zoom = atof(cls.Mid(1));
				
				if(log) {
					ProcessingStepRecord& s = log->steps.Add();
					s.step_id = log->next_step_id++;
					s.stage = "ZOOM";
					s.slot_id = slot.id;
					s.head_id = head_z;
					s.is_nn_step = true;
					s.duration_ms = (GetTickCount() - t_zoom);
					s.status = "OK";
					s.top_class = cls;
					s.confidence = res[best];
					s.note = Format("zoom=%.2f", cur_zoom);
					s.is_equalized = true;
					// Trace the probe area
					s.candidate_bbox = suit_bbox_probe;
					s.crop_size = suit_probe_sz;
					s.is_grayscale = zoom_gray;
					s.angle = angle;
				}
			}
			else {
				if(log) log->AddStep("ZOOM", (double)(GetTickCount() - t_zoom), "ERROR", "Empty zoom prediction output", slot.id, head_z, true);
				return r;
			}
		}
		else {
			if(log) log->AddStep("ZOOM", (double)(GetTickCount() - t_zoom), "ERROR", "Empty zoom probe crop", slot.id, head_z, true);
			return r;
		}
	}
	else {
		if(log) log->AddStep("ZOOM", 0, "MISSING_HEAD", "Zoom head not found in model", slot.id, head_z, true);
		return r;
	}

	// 2. X_OFFSET & Y_OFFSET (Never gated)
	// We use the predicted zoom to adjust the local crop anchor.
	AnnLayAnchor zoomed_anchor = slot.anchor;
	zoomed_anchor.w /= cur_zoom;
	zoomed_anchor.h /= cur_zoom;

	full_bbox = AnchorToRect(zoomed_anchor, isz, dx, dy);
	card_bbox = AnnLayResolveRegionRect(slot, full_bbox, "card_region");
	rank_bbox = AnnLayResolveRegionRect(slot, full_bbox, "rank_region");
	suit_bbox = AnnLayResolveRegionRect(slot, full_bbox, "suit_region");

	if(k_ox >= 0) {
			uint64 t_off = GetTickCount();
			Image full_crop = CropRotateScale(img, zoomed_anchor, slot.crop_size, dx, dy, angle);
			if(!full_crop.IsEmpty()) {
				Rect suit_local = AnnLayResolveRegionRect(slot, full_crop.GetSize(), "suit_region");
				Image suit_anchor = Crop(full_crop, suit_local);
				Size offset_crop_sz = suit_anchor.GetSize();
				
				bool off_gray = (sessions[k_ox]->GetInput()->input_depth == 1);
				Vector<double> inp = ImageToVec(suit_anchor, off_gray, offset_crop_sz);
				Vector<double> res = sessions[k_ox]->Predict(inp);
				int best = 0;
				for(int i = 1; i < res.GetCount(); i++) if(res[i] > res[best]) best = i;
				lx = StrInt(SafeGetClass(*sessions[k_ox], best));
				if(log) {
					log->AddStep("X_OFFSET", (double)(GetTickCount() - t_off), "OK", SafeFormatInt((int)round(dx + lx)), slot.id, head_ox, true);
					ProcessingStepRecord& psx = log->steps.Top();
					psx.candidate_bbox = suit_bbox;
					psx.crop_size = offset_crop_sz;
					psx.is_grayscale = off_gray;
					psx.angle = angle;
				}
			}
		} else if(log) log->AddStep("X_OFFSET", 0, "OK", SafeFormatInt((int)round(dx)), slot.id, head_ox, false);

		if(k_oy >= 0) {
			uint64 t_off = GetTickCount();
			Image full_crop = CropRotateScale(img, zoomed_anchor, slot.crop_size, dx, dy, angle);
			if(!full_crop.IsEmpty()) {
				Rect suit_local = AnnLayResolveRegionRect(slot, full_crop.GetSize(), "suit_region");
				Image suit_anchor = Crop(full_crop, suit_local);
				Size offset_crop_sz = suit_anchor.GetSize();

				bool off_gray = (sessions[k_oy]->GetInput()->input_depth == 1);
				Vector<double> inp = ImageToVec(suit_anchor, off_gray, offset_crop_sz);
				Vector<double> res = sessions[k_oy]->Predict(inp);
				int best = 0;
				for(int i = 1; i < res.GetCount(); i++) if(res[i] > res[best]) best = i;
				ly = StrInt(SafeGetClass(*sessions[k_oy], best));
				if(log) {
					log->AddStep("Y_OFFSET", (double)(GetTickCount() - t_off), "OK", SafeFormatInt((int)round(dy + ly)), slot.id, head_oy, true);
					ProcessingStepRecord& psy = log->steps.Top();
					psy.candidate_bbox = suit_bbox;
					psy.crop_size = offset_crop_sz;
					psy.is_grayscale = off_gray;
					psy.angle = angle;
				}
			}
		} else if(log) log->AddStep("Y_OFFSET", 0, "OK", SafeFormatInt((int)round(dy)), slot.id, head_oy, false);
	
	r.offset_dx = dx + lx;
	r.offset_dy = dy + ly;

	// Update bboxes with final local offsets AND zoom
	full_bbox = AnchorToRect(zoomed_anchor, isz, r.offset_dx, r.offset_dy);
	card_bbox = AnnLayResolveRegionRect(slot, full_bbox, "card_region");
	rank_bbox = AnnLayResolveRegionRect(slot, full_bbox, "rank_region");
	suit_bbox = AnnLayResolveRegionRect(slot, full_bbox, "suit_region");

	// 2. VISIBLE (Visibility)
	if(k_p < 0) {
		if(log) log->AddStep("VISIBLE", 0, "ERROR", "Session not found for " + slot.id + "#presence", slot.id, head_p, false);
		return r;
	}

	// Use local-offset-corrected AND zoom-corrected crop for visibility/level/category
	Image full_crop = CropRotateScale(img, zoomed_anchor, slot.crop_size, r.offset_dx, r.offset_dy, angle);
	if(full_crop.IsEmpty()) {
		if(log) log->AddStep("VISIBLE", (double)(GetTickCount() - t0), "ERROR", "Empty crop", slot.id, head_p, false);
		return r;
	}
	r.pixel_bbox = card_bbox;
	Rect card_local = AnnLayResolveRegionRect(slot, full_crop.GetSize(), "card_region");
	Rect rank_local = AnnLayResolveRegionRect(slot, full_crop.GetSize(), "rank_region");
	Rect suit_local = AnnLayResolveRegionRect(slot, full_crop.GetSize(), "suit_region");
	Image card_crop = Crop(full_crop, card_local);
	Image rank_crop = Crop(full_crop, rank_local);
	Image suit_crop = Crop(full_crop, suit_local);
	if(card_crop.IsEmpty() || rank_crop.IsEmpty() || suit_crop.IsEmpty()) {
		if(log) log->AddStep("VISIBLE", (double)(GetTickCount() - t0), "ERROR", "Empty sub-layout crop", slot.id, head_p, false);
		return r;
	}
	if(card_crop.GetSize() != slot.crop_size)
		card_crop = Rescale(card_crop, slot.crop_size);
	if(rank_crop.GetWidth() != 22 || rank_crop.GetHeight() != 22)
		rank_crop = Rescale(rank_crop, 22, 22);
	if(suit_crop.GetWidth() != 22 || suit_crop.GetHeight() != 22)
		suit_crop = Rescale(suit_crop, 22, 22);
	card_crop = HighLuminanceThresholdBinarization(card_crop);
	rank_crop = HighLuminanceThresholdBinarization(rank_crop);

	// element visibility gate model is grayscale.
	bool gray = (sessions[k_p]->GetInput()->input_depth == 1);
	Vector<double> inp_p = ImageToVec(card_crop, gray, slot.crop_size, false);
	Vector<double> scores_p = sessions[k_p]->Predict(inp_p);
	if(scores_p.GetCount() < 2) {
		if(log) log->AddStep("VISIBLE", (double)(GetTickCount() - t0), "ERROR", "Invalid prediction output", slot.id, head_p, true);
		return r;
	}

	double p_true = scores_p[1];
	r.confidence = p_true;
	bool visible = (p_true >= slot.presence_threshold);

	if(log) {
		String requested_h = slot.id + "#presence";
		log->AddStep("VISIBLE", (double)(GetTickCount() - t0), "OK", visible ? "true" : "false", slot.id, head_p, k_p >= 0);
		ProcessingStepRecord& ps = log->steps.Top();
		ps.candidate_bbox = card_bbox;
		ps.crop_size = slot.crop_size;
		ps.is_grayscale = gray;
		ps.is_equalized = false;
		ps.angle = angle;
		if(!head_p.IsEmpty() && head_p != requested_h)
			ps.requested_head_id = requested_h;
	}

	if(!visible) {
		r.top_class = ""; // no element
		if(log) {
			log->AddStep("LEVEL", 0, "SKIPPED_NO_CARD", "", slot.id, head_r, false);
			log->AddStep("CATEGORY", 0, "SKIPPED_NO_CARD", "", slot.id, head_s, false);
		}
		return r;
	}

	// 3. LEVEL
	uint64 t_rank = GetTickCount();
	String level = "?";
	if(k_r >= 0) {
		bool rank_gray = (sessions[k_r]->GetInput()->input_depth == 1);
		Vector<double> inp_r = ImageToVec(rank_crop, rank_gray, Size(22, 22), false);
		Vector<double> scores_r = sessions[k_r]->Predict(inp_r);
		int best = 0;
		for(int i = 1; i < scores_r.GetCount(); i++)
			if(scores_r[i] > scores_r[best]) best = i;
		level = SafeGetClass(*sessions[k_r], best);
		if(log) {
			String requested_h = slot.id + "#level";
			log->AddStep("LEVEL", (double)(GetTickCount() - t_rank), "OK", level, slot.id, head_r, k_r >= 0);
			ProcessingStepRecord& ps = log->steps.Top();
			ps.candidate_bbox = rank_bbox;
			ps.crop_size = Size(22, 22);
			ps.is_grayscale = rank_gray;
			ps.is_equalized = false;
			ps.angle = angle;
			if(!head_r.IsEmpty() && head_r != requested_h)
				ps.requested_head_id = requested_h;
		}
	} else {
		if(log) log->AddStep("LEVEL", 0, "MISSING_HEAD", "", slot.id, head_r, false);
	}

	// 4. CATEGORY
	uint64 t_suit = GetTickCount();
	String category = "?";
	if(k_s >= 0) {
		bool suit_gray = (sessions[k_s]->GetInput()->input_depth == 1);
		Vector<double> inp_s = ImageToVec(suit_crop, suit_gray, Size(22, 22), false);
		Vector<double> scores_s = sessions[k_s]->Predict(inp_s);
		int best = 0;
		for(int i = 1; i < scores_s.GetCount(); i++)
			if(scores_s[i] > scores_s[best]) best = i;
		category = SafeGetClass(*sessions[k_s], best);
		if(log) {
			String requested_h = slot.id + "#category";
			log->AddStep("CATEGORY", (double)(GetTickCount() - t_suit), "OK", category, slot.id, head_s, k_s >= 0);
			ProcessingStepRecord& ps = log->steps.Top();
			ps.candidate_bbox = suit_bbox;
			ps.crop_size = Size(22, 22);
			ps.is_grayscale = suit_gray;
			ps.is_equalized = false;
			ps.angle = angle;
			if(!head_s.IsEmpty() && head_s != requested_h)
				ps.requested_head_id = requested_h;
		}
	} else {
		if(log) log->AddStep("CATEGORY", 0, "MISSING_HEAD", "", slot.id, head_s, false);
	}

	r.top_class = level + category;
	return r;
}

SlotResult AnchoredSlotRecognizer::RecognizeCompositeElementIterative(const AnnLaySlot& slot, const Image& img, double dx, double dy, ProcessingLogRecord* log) {
	// Stub for PROCSTEP-06: 1-5 iterations with majority voting and nudging.
	// Currently just does one pass.
	int iterations = 1;
	SlotResult best_r;
	
	for(int i = 0; i < iterations; i++) {
		SlotResult r = RecognizeCompositeElement(slot, img, dx, dy, log);
		if(i == 0) best_r = r;
		
		// TODO: majority voting for level/category
		// TODO: nudge dx/dy if stuck at [0,0]
	}
	
	return best_r;
}

SlotResult AnchoredSlotRecognizer::RecognizeOrb(const AnnLaySlot& slot, const Image& img) {
	SlotResult r;
	r.slot_id = slot.id;
	r.method = AnnLayMethodToString(slot.method);
	(void)img;
	// TODO: Implement ORB matching with OpenCV-based descriptor matching.
	r.class_index = -1;
	r.confidence = 0.0;
	return r;
}

Vector<SlotResult> AnchoredSlotRecognizer::Recognize(const Image& img, ProcessingLogRecord* log) {
	Mutex::Lock __(recognize_lock);
	Vector<SlotResult> results;
	if(!loaded || img.IsEmpty())
		return results;

	if(!offset_strategy_logged) {
		String strategy = "none";
		if(offset_mode == OFFSET_NONE)
			strategy = "none (disabled)";
		else {
			bool can_comb = false;
			bool can_split = false;
			for(const AnnLaySlot& s : lay.slots) {
				if(sessions.Find(s.id + "#offset") >= 0) can_comb = true;
				if(sessions.Find(s.id + "#offset_x") >= 0 || sessions.Find(s.id + "#offset_y") >= 0) can_split = true;
			}
			if(offset_mode == OFFSET_COMBINED) strategy = can_comb ? "combined" : "none (heads missing)";
			else if(offset_mode == OFFSET_SPLIT) strategy = can_split ? "split" : "none (heads missing)";
			else if(offset_mode == OFFSET_AUTO) {
				if(can_comb) strategy = "combined (auto)";
				else if(can_split) strategy = "split (fallback)";
				else strategy = "none";
			}
		}
		Cerr() << "[AnchoredSlotRecognizer] Active offset strategy: " << strategy << "\n";
		offset_strategy_logged = true;
	}

	double global_dx = 0.0;
	double global_dy = 0.0;
	
	bool has_ocr_in_filter = false;
	if(!slot_filter.IsEmpty()) {
		for(int i = 0; i < lay.slots.GetCount(); i++) {
			if(lay.slots[i].method == ANNLAY_OCR_TEXT && slot_filter.Find(lay.slots[i].id) >= 0) {
				has_ocr_in_filter = true;
				break;
			}
		}
	}

	if(slot_filter.IsEmpty() || has_ocr_in_filter) {
		ComputeOcrOffset(img, global_dx, global_dy, log);
		if(global_dx != 0.0 || global_dy != 0.0) {
			Cerr() << "[AnchoredSlotRecognizer] Global OCR offset applied: (" << global_dx << ", " << global_dy << ")\n";
		}
	}

	for(int i = 0; i < lay.slots.GetCount(); i++) {
		const AnnLaySlot& slot = lay.slots[i];
		if(slot.method != ANNLAY_OCR_TEXT)
			continue;

		if(!slot_filter.IsEmpty() && slot_filter.Find(slot.id) < 0)
			continue;

		uint64 t0 = GetTickCount();
		int q = ocr_cache_.Find(slot.id);
		SlotResult r;
		OCR::OCRPreprocessMode mode = OCR::OCR_PREPROCESS_GRAYSCALE;
		if(!slot.ocr_preprocess_mode.IsEmpty())
			mode = OCR::StringToOCRPreprocessMode(slot.ocr_preprocess_mode);

		if(q >= 0)
			r = ocr_cache_[q];
		else {
			r.slot_id = slot.id;
			r.method = AnnLayMethodToString(slot.method);
			if(ocr_initialized) {
				String error;
				double conf = 0;
				Rect out_bbox;
				Rect crop_bbox = AnchorToRect(slot.anchor, img.GetSize()) & Rect(img.GetSize());
				Image crop = crop_bbox.IsEmpty() ? Image() : Crop(img, crop_bbox);
				r.raw_text = RunTesseract(crop, slot.id, error, conf, out_bbox, mode, slot.ocr_psm, slot.ocr_whitelist, slot.ocr_blacklist);
				r.confidence = conf;
				r.pixel_bbox = out_bbox;
				SlotResult& cached = ocr_cache_.Add(slot.id);
				cached = r;
				q = ocr_cache_.GetCount() - 1;
			}
		}
		r.t_ms = (GetTickCount() - t0);
		results.Add(r);

		if(log) {
			ProcessingStepRecord& s = log->steps.Add();
			s.step_id = log->next_step_id++;
			s.stage = "OCR";
			s.slot_id = slot.id;
			s.duration_ms = r.t_ms;
			s.status = "OK";
			s.note = r.raw_text;
			s.detailed_note << "Preprocess mode: " << OCR::OCRPreprocessModeToString(mode) << "\n";
			s.ocr_mode = (int)mode;
			
			// Set the candidate_bbox so it can be picked up as a named input (I1, I2, etc.)
			s.candidate_bbox = AnchorToRect(slot.anchor, img.GetSize());
		}
	}

	// cv_template_match: run once per stem per group
	Index<String> cv_stems_done;
	for(int gi = 0; gi < (int)sln.cv_template_groups.GetCount(); gi++) {
		const CvTemplateGroup& g = sln.cv_template_groups[gi];
		for(const String& stem : g.slot_stems) {
			if(!slot_filter.IsEmpty() && slot_filter.Find(stem) < 0 &&
			   slot_filter.Find(stem + "#label_a") < 0 && slot_filter.Find(stem + "#present") < 0)
				continue;
			if(cv_stems_done.Find(stem) >= 0) continue;
			cv_stems_done.Add(stem);
			RecognizeCvTemplateGroup(gi, stem, img, global_dx, global_dy, results, log);
		}
	}

	// Phase 6: Element runtime ordering contract.
	// 1. zoom, 2. is_visible, 3. x_offset, 4. y_offset, 5. level, 6. category.
	struct OrderedSlot : Moveable<OrderedSlot> {
		int index;
		int priority;
		String stem;
		String stage_name;
		
		bool operator<(const OrderedSlot& b) const {
			if(stem != b.stem) return stem < b.stem;
			return priority < b.priority;
		}
	};
	Vector<OrderedSlot> ordered;
	for(int i = 0; i < lay.slots.GetCount(); i++) {
		const AnnLaySlot& slot = lay.slots[i];
		if(slot.method == ANNLAY_OCR_TEXT) continue;
		
		OrderedSlot& os = ordered.Add();
		os.index = i;
		os.priority = 100;
		os.stem = slot.id;
		os.stage_name = "RECOGNIZE";
		
		static const struct { const char* suffix; int prio; const char* stage; } kPrios[] = {
			{"_is_visible", 1, "VISIBLE"},
			{"_x_offset", 2, "X_OFFSET"},
			{"_y_offset", 3, "Y_OFFSET"},
			{"_rank", 4, "LEVEL"},
			{"_suit", 5, "CATEGORY"},
			{nullptr, 0, nullptr}
		};
		for(int p = 0; kPrios[p].suffix; p++) {
			if(slot.id.EndsWith(kPrios[p].suffix)) {
				os.priority = kPrios[p].prio;
				os.stem = slot.id.Left(slot.id.GetCount() - strlen(kPrios[p].suffix));
				os.stage_name = kPrios[p].stage;
				break;
			}
		}
	}
	Sort(ordered);

	for(int oi = 0; oi < ordered.GetCount(); oi++) {
		const OrderedSlot& os = ordered[oi];
		const AnnLaySlot& slot = lay.slots[os.index];
		
		if(!slot_filter.IsEmpty() && slot_filter.Find(slot.id) < 0)
			continue;

		// Skip redundant visibility slots for cards
		if(slot.method == ANNLAY_CLASSIFIER_BOOL && 
		   (slot.id.StartsWith("is_board_card_") || slot.id.StartsWith("is_hero_card_")))
			continue;
		// If this element stem is managed by cv_template_groups, suppress the
		// legacy composite-element NN pipeline entirely even when .annlay still
		// has old method/composite settings.
		if(slot.composite_type == ANNLAY_COMPOSITE_ELEMENT &&
		   cv_stems_done.Find(slot.id) >= 0)
			continue;
		// cv_template_match results are emitted per stem group above.
		// Never run legacy per-slot/element pipeline for these slots.
		if(slot.method == ANNLAY_CV_TEMPLATE_MATCH)
			continue;

		uint64 t0 = GetTickCount();
		SlotResult r;
		if(slot.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
			double ldx = global_dx, ldy = global_dy;
			ComputeLocalOffset(slot, img, global_dx, global_dy, ldx, ldy);
			r = RecognizeCompositeElementIterative(slot, img, ldx, ldy, log);
		}
		else {
			double ldx = global_dx, ldy = global_dy;
			ComputeLocalOffset(slot, img, global_dx, global_dy, ldx, ldy);

			switch(slot.method) {
			case ANNLAY_CLASSIFIER_BOOL:
			case ANNLAY_CLASSIFIER_LABEL:
				if(IsZeroAnchor(slot.anchor) && slot.anchor_candidates.IsEmpty() && slot.group != "dealer_chip" && slot.group != "active_timer")
					break;
				if(!slot.anchor_candidates.IsEmpty())
					r = RecognizeVariableClassifier(slot, img, ldx, ldy, log);
				else
					r = RecognizeClassifierWithOffset(slot, img, ldx, ldy, log);
				break;

			case ANNLAY_ORB_MATCH:
				r = RecognizeOrb(slot, img);
				break;

			case ANNLAY_CV_TEMPLATE_MATCH:
				// Handled below per stem group — skip individual slot processing
				break;

			case ANNLAY_IGNORED:
			default:
				break;
			}
			if(log && !r.slot_id.IsEmpty()) {
				bool is_nn = (slot.method == ANNLAY_CLASSIFIER_BOOL || slot.method == ANNLAY_CLASSIFIER_LABEL);
				
				String head_id = slot.id;
				if(slot.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
					if(os.stage_name == "VISIBLE") head_id = slot.id + "#presence";
					else if(os.stage_name == "LEVEL") head_id = slot.id + "#level";
					else if(os.stage_name == "CATEGORY") head_id = slot.id + "#category";
				}
				
				String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(head_id, &lay);
				if(!gkey.IsEmpty()) head_id = gkey;
				
				if(slot.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
					if(os.stage_name == "VISIBLE" || os.stage_name == "LEVEL" || os.stage_name == "CATEGORY")
						is_nn = true;
				}

				ProcessingStepRecord& ps = log->steps.Add();
				ps.step_id = log->next_step_id++;
				ps.stage = os.stage_name;
				ps.slot_id = slot.id;
				ps.head_id = is_nn ? head_id : String();
				ps.is_nn_step = is_nn;
				ps.candidate_bbox = r.pixel_bbox;
				ps.crop_size = slot.crop_size;
				ps.is_equalized = true;
				if(slot.method == ANNLAY_CLASSIFIER_BOOL || slot.method == ANNLAY_CLASSIFIER_LABEL)
					ps.is_equalized = false;
				
				bool effective_gray = (slot.color_mode != "color");
				if(is_nn) {
					int sk = sessions.Find(head_id);
					if(sk >= 0) effective_gray = (sessions[sk]->GetInput()->input_depth == 1);
				}
				ps.is_grayscale = effective_gray;
				if(is_nn && ShouldUseHighLumaHead(head_id))
					ps.is_equalized = false;

				ps.duration_ms = (double)(GetTickCount() - t0);
				ps.status = "OK";
				ps.note = r.raw_text.IsEmpty() ? r.top_class : r.raw_text;
				if(ps.note.IsEmpty() && r.class_index >= 0) ps.note = AsString(r.class_index);
				if(!r.details.IsEmpty()) {
					if(!ps.note.IsEmpty()) ps.note << " | ";
					ps.note << r.details;
					ps.note.Replace("\n", " ");
				}

				// Emit per-seat GROUP_RECOGNIZE steps for winner-takes-all group slots
				// (e.g. dealer_chip, active_timer: one candidate per seat, highest score wins)
				if(!slot.anchor_candidates.IsEmpty() && !slot.group.IsEmpty() &&
				   r.winner_cand_index >= 0) {
					String group = slot.group;
					String seat_suffix;
					if(group == "dealer_chip")      seat_suffix = "#is_dealer";
					else if(group == "active_timer") seat_suffix = "#active_timer";

					if(!seat_suffix.IsEmpty()) {
						String head_grp = head_id;
						Vector<int> candidate_seats = MapCandidatesToSeats(lay, slot, img.GetSize());
						bool valid_seat_mapping = IsCandidateSeatMappingBijection(candidate_seats);
						String seat_mapping_note = FormatCandidateSeatMapping(candidate_seats);
						if(!valid_seat_mapping) {
							AppendProcessingWarning(log, "[GROUP_RECOGNIZE] invalid seat ordering for " + slot.id + "; falling back to candidate index order");
							seat_mapping_note << " (invalid; fallback ci+1)";
						}
						for(int ci = 0; ci < slot.anchor_candidates.GetCount(); ci++) {
							ProcessingStepRecord& gs = log->steps.Add();
							gs.step_id = log->next_step_id++;
							gs.stage = "GROUP_RECOGNIZE";
							int seat_no = valid_seat_mapping ? candidate_seats[ci] : ci + 1;
							gs.slot_id = "seat" + AsString(seat_no) + seat_suffix;
							gs.head_id = head_grp;
							gs.is_nn_step = true;
							gs.candidate_bbox = AnchorToRect(slot.anchor_candidates[ci], img.GetSize());
							gs.crop_size = slot.crop_size;
							gs.is_grayscale = (slot.color_mode != "color");
							gs.is_equalized = false;
							gs.duration_ms = 0;
							gs.status = "OK";
							bool is_winner = (ci == r.winner_cand_index);
							gs.note = is_winner ? "true" : "false";
							if(ci == 0)
								gs.detailed_note = seat_mapping_note;
						}
					}
				}
			}
		}
		if(!r.slot_id.IsEmpty()) {
			r.t_ms = (GetTickCount() - t0);
			results.Add(r);
		}
	}

	// Synthesize seat1/in_game as a LOGICAL step (card1.is && card2.is) when
	// no seat1_in_game classifier slot exists (hero seat has no in_game model).
	{
		bool has_seat1_in_game = false;
		for(int i = 0; i < lay.slots.GetCount(); i++) {
			if(lay.slots[i].id == "seat1_in_game") { has_seat1_in_game = true; break; }
		}
		if(!has_seat1_in_game) {
			// Find hero element presence results
			bool card1_is = false, card2_is = false;
			for(int i = 0; i < results.GetCount(); i++) {
				const SlotResult& r = results[i];
				if(r.slot_id == "hero_card_1" || r.slot_id == "hero_card_1#present" || r.slot_id == "hero_card_1#presence")
					card1_is = (r.top_class == "true" || r.top_class == "1");
				if(r.slot_id == "hero_card_2" || r.slot_id == "hero_card_2#present" || r.slot_id == "hero_card_2#presence")
					card2_is = (r.top_class == "true" || r.top_class == "1");
			}
			bool in_game_val = card1_is && card2_is;

			SlotResult gr;
			gr.slot_id = "seat1_in_game";
			gr.method = "logical";
			gr.top_class = in_game_val ? "true" : "false";
			gr.class_index = in_game_val ? 1 : 0;
			results.Add(gr);

			if(log) {
				ProcessingStepRecord& ps = log->steps.Add();
				ps.step_id = log->next_step_id++;
				ps.stage = "LOGICAL";
				ps.slot_id = "seat1_in_game";
				ps.is_nn_step = false;
				ps.duration_ms = 0;
				ps.status = "OK";
				ps.note = in_game_val ? "true" : "false";
			}
		}
	}

	uint64 t_gate0 = GetTickCount();
	ApplyBoolGates(results);
	
	if(log) {
		// Slice 19: Back-propagate gate status to step records for UI transparency
		for(const auto& r : results) {
			if(!r.gate_slot_id.IsEmpty()) {
				bool blocked = r.gate_status.StartsWith("blocked");
				for(int i = 0; i < log->steps.GetCount(); i++) {
					ProcessingStepRecord& ps = log->steps[i];
					if(ps.slot_id == r.slot_id) {
						ps.gate_slot_id = r.gate_slot_id;
						ps.gate_status = r.gate_status;
						if(blocked && ps.stage == "OCR") {
							ps.note = "(skipped: 'is' gate)";
							ps.status = "SKIPPED";
						}
					}
				}
			}
		}

		for(const auto& r : results) {
			if(r.method == "ocr_text") {
				if(r.raw_text.IsEmpty()) {
					log->watchlist << "  [FAIL] " << r.slot_id << ": No text recognized\n";
				}
				else if(r.confidence < 0.6) {
					log->watchlist << "  [WEAK] " << r.slot_id << ": Low confidence (" << Format("%.2f", r.confidence) << ")\n";
				}
			}
		}

		ProcessingStepRecord& s = log->steps.Add();
		s.step_id = log->next_step_id++;
		s.stage = "ApplyBoolGates";
		s.duration_ms = (GetTickCount() - t_gate0);
		s.status = "OK";
	}

	return results;
}

void AnchoredSlotRecognizer::Recognize(const Image& img, VectorMap<String, SlotResult>& out, ProcessingLogRecord* log) {
	Vector<SlotResult> v = Recognize(img, log);
	for(int i = 0; i < v.GetCount(); i++)
		out.GetAdd(v[i].slot_id) = v[i];
}


String ProcessingLogRecord::FormatVerbose() const {
	String s;
	s << "=== Frame Processing Trace ===\n";
	s << "Sequence ID: " << frame_seq << "\n";
	s << "Image Name:  " << image_name << "\n";
	s << "Image Path:  " << image_path << "\n";
	s << "Status:      " << status << "\n";
	s << "Warnings:    " << warnings << "\n";
	
	s << "\n--- Timings (ms) ---\n";
	s << Format("Queue Wait:  %8.2f\n", t_queue_ms);
	s << Format("Load/Decode: %8.2f\n", t_load_ms);
	s << Format("Recognize:   %8.2f\n", t_recognize_ms);
	s << Format("Script/PyVM: %8.2f\n", t_script_ms);
	s << Format("Overlay:     %8.2f\n", t_overlay_ms);
	s << "------------------------\n";
	s << Format("Total Pipeline: %8.2f\n", t_total_ms);
	
	s << "\n--- Parsed Content Summary ---\n";
	s << "Detections Good:    " << detections_good << "\n";
	s << "Detections Missing: " << detections_missing << "\n";
	
	if(results.GetCount() > 0) {
		s << "\n--- Slot Details ---\n";
		s << Format("%-25s %-15s %-6s %-8s %-10s %-15s %-10s\n", "Slot ID", "Value", "Conf%", "T(ms)", "Offset", "Gate Slot", "Gate Stat");
		for(const auto& r : results) {
			String val = r.raw_text.IsEmpty() ? r.top_class : r.raw_text;
			String conf = r.confidence > 0 ? Format("%.0f", r.confidence * 100) : "N/A";
			String t_ms = Format("%.1f", r.t_ms);
			String off = (fabs(r.offset_dx) >= 0.1 || fabs(r.offset_dy) >= 0.1) 
			             ? SafeFormatInt((int)round(r.offset_dx)) + "," + SafeFormatInt((int)round(r.offset_dy)) 
			             : "0,0";
			s << Format("%-25s %-15s %-6s %-8s %-10s %-15s %-10s\n", 
			            ~r.slot_id, ~val, ~conf, ~t_ms, ~off, ~r.gate_slot_id, ~r.gate_status);
			if(!r.details.IsEmpty()) {
				String d = r.details;
				d.Replace("\n", "\n    ");
				s << "    " << d;
				if(!d.EndsWith("\n")) s << "\n";
			}
		}
	}

	if(steps.GetCount() > 0) {
		s << "\n--- Element Stage Trace (Structured) ---\n";
		s << Format("%-20s %-25s %-25s %-8s %-10s %s\n", "Stage", "Slot ID", "Head ID", "T(ms)", "Status", "Note");
		for(const auto& step : steps) {
			s << Format("%-20s %-25s %-25s %-8.1f %-10s %s\n", 
			            ~step.stage, ~step.slot_id, ~step.head_id, step.duration_ms, ~step.status, ~step.note);
		}

		s << "\n--- Element Stage Contract Proof ---\n";
		for(const auto& step : steps) {
			if(step.slot_id.StartsWith("board_card_") || step.slot_id.StartsWith("hero_card_")) {
				s << Format("PROOF: %s stage %s -> %s\n", ~step.slot_id, ~step.stage, ~step.status);
			}
		}
	}
	
	if(!script_output.IsEmpty()) {
		s << "\n--- Script Output ---\n";
		s << script_output;
		if(!script_output.EndsWith("\n")) s << "\n";
	}
	
	if(!script_error.IsEmpty()) {
		s << "\n--- Script Error ---\n";
		s << script_error;
		if(!script_error.EndsWith("\n")) s << "\n";
	}
	
	if(!watchlist.IsEmpty()) {
		s << "\n--- Quality Watchlist (Potential Issues) ---\n";
		s << watchlist << "\n";
	}
	
	if(meta.GetCount() > 0) {
		s << "\n--- Final Metadata (RecognitionScript) ---\n";
		for(int i = 0; i < meta.GetCount(); i++) {
			s << Format("%-25s : %s\n", ~meta.GetKey(i), ~meta[i]);
		}
	}
	
	return s;
}

::ConvNet::Session* AnchoredSlotRecognizer::GetSession(const String& head_id) {
	Mutex::Lock __(recognize_lock);
	int k = sessions.Find(head_id);
	if(k >= 0)
		return ~sessions[k];
	return nullptr;
}

Vector<String> AnchoredSlotRecognizer::GetLoadedSessionHeads() {
	Mutex::Lock __(recognize_lock);
	Vector<String> out;
	out.Reserve(sessions.GetCount());
	for(int i = 0; i < sessions.GetCount(); i++)
		out.Add(sessions.GetKey(i));
	return out;
}
void ProcessingLogRecord::AddStep(const String& stage, double duration_ms, const String& status, const String& note, const String& slot_id, const String& head_id, bool is_nn, const String& detailed_note, const String& gate_slot_id, const String& gate_status) {
	ProcessingStepRecord& s = steps.Add();
	s.step_id = next_step_id++;
	s.stage = stage;
	s.slot_id = slot_id;
	s.duration_ms = duration_ms;
	s.status = status;
	s.note = note;
	s.detailed_note = detailed_note;
	s.head_id = head_id;
	s.gate_slot_id = gate_slot_id;
	s.gate_status = gate_status;
	s.is_nn_step = is_nn;

	// Heuristic: if stage is LEVEL/CATEGORY and note is small, it's likely the result class.
	if((stage == "LEVEL" || stage == "CATEGORY") && note.GetCount() > 0 && note.GetCount() <= 5) {
		s.top_class = note;
	}
}

namespace {

static bool IsAllDigits(const String& s) {
	if(s.IsEmpty())
		return false;
	for(int i = 0; i < s.GetCount(); i++) {
		char c = s[i];
		if(c < '0' || c > '9')
			return false;
	}
	return true;
}

static int ParsePrefixedNumber(const String& s, const char* prefix) {
	String pfx(prefix);
	if(!s.StartsWith(pfx))
		return -1;
	String tail = s.Mid(pfx.GetCount());
	if(!IsAllDigits(tail))
		return -1;
	return atoi(~tail);
}

static bool IsSeatRootPath(const String& p) {
	return ParsePrefixedNumber(p, "seat") > 0 && p.Find('/') < 0;
}

static bool IsCardPath(const String& p) {
	int slash = p.ReverseFind('/');
	String tail = slash >= 0 ? p.Mid(slash + 1) : p;
	return ParsePrefixedNumber(tail, "element") > 0;
}

static String GetCvDisplayName(const AnnSln& sln, const String& full_sid, const String& fallback) {
	int hash = full_sid.Find('#');
	if(hash < 0)
		return fallback;

	String stem = full_sid.Left(hash);
	String suffix = full_sid.Mid(hash);
	for(const CvTemplateGroup& g : sln.cv_template_groups) {
		bool has_stem = false;
		for(const String& s : g.slot_stems) {
			if(s == stem) {
				has_stem = true;
				break;
			}
		}
		if(!has_stem)
			continue;
		if(suffix == "#label_a") {
			String v = TrimBoth(g.label_a_display);
			return v.IsEmpty() ? fallback : v;
		}
		if(suffix == "#label_b") {
			String v = TrimBoth(g.label_b_display);
			return v.IsEmpty() ? fallback : v;
		}
		if(suffix == "#present") {
			String v = TrimBoth(g.present_display);
			return v.IsEmpty() ? fallback : v;
		}
	}
	return fallback;
}

static bool MapStepToTreeLeaf(const ProcessingStepRecord& s, int source_index, const AnnSln& sln, ProcessingTreeNode& out) {
	if(s.stage == "TOTAL" || s.stage == "ComputeOcrOffset")
		return false;

	String sid = s.slot_id;
	String sid_stem = sid;
	int sid_hash = sid.Find('#');
	if(sid_hash >= 0)
		sid_stem = sid.Left(sid_hash);

	String sid_suffix = sid_hash >= 0 ? sid.Mid(sid_hash + 1) : String();

	// Handle GROUP_RECOGNIZE steps: slot_id is like "seat1#is_dealer"
	if(s.stage == "GROUP_RECOGNIZE") {
		if(sid_suffix == "is_dealer" && sid_stem.StartsWith("seat")) {
			out.label = "is_dealer";
			out.path = sid_stem;
			out.stage = "RECOGNIZE";
			out.duration_ms = s.duration_ms;
			out.status = s.status;
			out.note = s.note;
			out.is_leaf = true;
			out.step_id = s.step_id;
			out.source_index = source_index;
			return true;
		}
		if(sid_suffix == "active_timer" && sid_stem.StartsWith("seat")) {
			out.label = "is";
			out.path = sid_stem + "/turn/active_timer";
			out.stage = "RECOGNIZE";
			out.duration_ms = s.duration_ms;
			out.status = s.status;
			out.note = s.note;
			out.is_leaf = true;
			out.step_id = s.step_id;
			out.source_index = source_index;
			return true;
		}
		return false;
	}

	// Hide redundant legacy visibility wrappers for cards.
	if(sid_stem.StartsWith("is_board_card_") || sid_stem.StartsWith("is_hero_card_") || sid_stem.EndsWith("_is_visible"))
		return false;
	if(sid_stem == "active_timer")
		return false;

	String stage = s.stage;
	if(stage == "PRESENT" || stage == "VISIBLE" || stage == "CATEGORY")
		stage = "RECOGNIZE";
	else if(stage == "LEVEL")
		stage = "MATCH";

	String display_name = s.stage;
	if(s.stage == "RECOGNIZE" || s.stage == "VISIBLE" || s.stage == "PRESENT")
		display_name = "is";
	else if(s.stage == "LEVEL")
		display_name = "level";
	else if(s.stage == "CATEGORY")
		display_name = "category";
	else if(s.stage == "X_OFFSET")
		display_name = "x_offset";
	else if(s.stage == "Y_OFFSET")
		display_name = "y_offset";
	else if(s.stage == "ZOOM")
		display_name = "zoom";
	else if(s.stage == "LOAD")
		display_name = "FrameImageFileLoad";
	else if(s.stage == "OVERLAY")
		display_name = "FrameOverlayRender";
	else if(s.stage == "SCRIPT")
		display_name = "RecognitionScriptRun";
	else if(s.stage == "OCR_PROCESS")
		display_name = "OcrProcess";
	else if(s.stage == "PRESENT")
		display_name = GetCvDisplayName(sln, sid, "is");

	String path;
	if(!sid.IsEmpty()) {
		if(s.stage == "SCRIPT" && sid_stem == "GameState") {
			path.Clear();
			display_name = "GameState";
		}
		else if(sid_stem.StartsWith("board_card_"))
			path = "board/element" + sid_stem.Mid(11);
		else if(sid_stem.StartsWith("is_board_card_"))
			path = "board/element" + sid_stem.Mid(14);
		else if(sid_stem.StartsWith("hero_card_"))
			path = "seat1/element" + sid_stem.Mid(10);
		else if(sid_stem.StartsWith("is_hero_card_"))
			path = "seat1/element" + sid_stem.Mid(13);
		else if(sid_stem.StartsWith("action_") || sid_stem.StartsWith("is_action_")) {
			String action = sid_stem.StartsWith("is_action_") ? sid_stem.Mid(10) : sid_stem.Mid(7);
			path = "seat1/turn/action/" + action;
			display_name = sid_stem.StartsWith("is_action_") ? "is" : "text";
		}
		else if(sid_stem == "game_name") {
			path = "game";
			display_name = "name";
		}
		else if(sid_stem == "side_pot" || sid_stem == "total_pot" || sid_stem == "pot_previous" ||
		        sid_stem == "is_side_pot" || sid_stem == "is_previous_pot" || sid_stem == "is_total_pot") {
			String base = sid_stem;
			if(sid_stem == "pot_previous")
				base = "previous_pot";
			if(sid_stem == "is_side_pot") {
				base = "side_pot";
				display_name = "is";
			}
			else if(sid_stem == "is_previous_pot") {
				base = "previous_pot";
				display_name = "is";
			}
			else if(sid_stem == "is_total_pot") {
				base = "total_pot";
				display_name = "is";
			}
			else if(stage == "OCR") {
				display_name = "text";
			}
			path = base.IsEmpty() ? "board" : "board/" + base;
		}
		else if(sid_stem.StartsWith("tourney_")) {
			path = "game";
			display_name = sid_stem.Mid(8);
		}
		else if(sid_stem == "dealer_chip") {
			// Dealer chip is seat-scoped now; hide old root node.
			return false;
		}
		else if(sid_stem.StartsWith("seat")) {
			int us = sid_stem.Find('_');
			if(us > 0) {
				String seat = sid_stem.Left(us);
				String tail = sid_stem.Mid(us + 1);
				if(tail == "panel") {
					path = seat;
					display_name = "is";
				}
				else if(tail == "is_bet_chip") {
					path = seat + "/bet";
					display_name = "is";
				}
				else if(tail == "bet") {
					path = seat + "/bet";
					display_name = "text";
				}
				else {
					path = seat;
					display_name = tail;
				}
			}
			else {
				path = sid_stem;
				display_name = "status";
			}
		}
		else if(sid_stem == "OcrProcess") {
			path.Clear();
		}
		else {
			path = sid_stem;
		}
	}

	if(s.stage == "CANDIDATE" && sid_stem != "dealer_chip" && !sid_stem.EndsWith("_is_bet_chip"))
		path = sid_stem + "/multi_eval";

	out.label = display_name;
	out.path = path;
	out.stage = stage;
	out.duration_ms = s.duration_ms;
	out.status = s.status;
	out.note = s.note;
	out.is_leaf = true;
	out.step_id = s.step_id;
	out.source_index = source_index;
	return true;
}

static int NodeOrder(const String& parent_path, const ProcessingTreeNode& n) {
	const String& label = n.label;
	if(parent_path.IsEmpty()) {
		if(label == "FrameImageFileLoad") return 0;
		if(label == "game") return 10;
		if(label == "board") return 20;
		int seat_no = ParsePrefixedNumber(label, "seat");
		if(seat_no > 0) return 30 + seat_no;
		if(label == "ApplyBoolGates") return 90;
		return 80;
	}
	if(parent_path == "game") {
		if(label == "name") return 0;
		if(label == "tourney") return 10;
		return 20;
	}
	if(parent_path == "board") {
		int card_no = ParsePrefixedNumber(label, "element");
		if(card_no > 0) return card_no;
		if(label == "previous_pot") return 20;
		if(label == "side_pot") return 21;
		if(label == "total_pot") return 22;
		return 30;
	}
	if(IsSeatRootPath(parent_path)) {
		if(label == "is") return 0;
		if(label == "name") return 1;
		if(label == "balance") return 2;
		if(label == "in_game") return 3;
		if(label == "is_dealer") return 4;
		if(label == "bet") return 5;
		if(label == "turn") return 6;
		int card_no = ParsePrefixedNumber(label, "element");
		if(card_no > 0) return 10 + card_no;
		return 40;
	}
	if(parent_path.EndsWith("/turn")) {
		if(label == "active_timer") return 0;
		if(label == "action") return 1;
		if(label == "is") return 2;
		return 10;
	}
	if(parent_path.EndsWith("/turn/action")) {
		if(label == "call") return 0;
		if(label == "fold") return 1;
		if(label == "raise") return 2;
		return 10;
	}
	if(parent_path.EndsWith("/bet") ||
	   parent_path == "board/previous_pot" ||
	   parent_path == "board/side_pot" ||
	   parent_path == "board/total_pot") {
		if(label == "is") return 0;
		if(label == "text") return 1;
		if(label == "OCR") return 2;
		return 10;
	}
	if(IsCardPath(parent_path)) {
		if(label == "is") return 0;
		if(label == "level") return 1;
		if(label == "category") return 2;
		return 10;
	}
	if(label == "is") return 0;
	if(label == "text") return 1;
	return 50;
}

static bool ComesBefore(const String& parent_path, const ProcessingTreeNode& a, const ProcessingTreeNode& b) {
	int oa = NodeOrder(parent_path, a);
	int ob = NodeOrder(parent_path, b);
	if(oa != ob)
		return oa < ob;
	if(a.source_index != b.source_index)
		return a.source_index < b.source_index;
	return ToLower(a.label) < ToLower(b.label);
}

static String RectKey(const Rect& r) {
	return Format("%d,%d,%d,%d", r.left, r.top, r.right, r.bottom);
}

static bool StepUsesNamedInput(const ProcessingStepRecord& s) {
	String stage = s.stage;
	if(stage == "PRESENT" || stage == "VISIBLE" || stage == "CATEGORY")
		stage = "RECOGNIZE";
	else if(stage == "LEVEL")
		stage = "MATCH";
	return stage == "RECOGNIZE" || s.stage == "LEVEL" || s.stage == "CATEGORY" || s.stage == "LABEL_A" || s.stage == "LABEL_B" || s.stage == "OCR" || s.stage == "GROUP_RECOGNIZE";
}

static void SortChildren(Vector<ProcessingTreeNode>& nodes, int parent) {
	Vector<int>& ch = nodes[parent].children;
	for(int i = 1; i < ch.GetCount(); i++) {
		int key = ch[i];
		int j = i - 1;
		while(j >= 0 && ComesBefore(nodes[parent].path, nodes[key], nodes[ch[j]])) {
			ch[j + 1] = ch[j];
			j--;
		}
		ch[j + 1] = key;
	}
	for(int i = 0; i < ch.GetCount(); i++) {
		if(!nodes[ch[i]].is_leaf)
			SortChildren(nodes, ch[i]);
	}
	}

}

void BuildProcessingStepsTree(const ProcessingLogRecord& log, const AnnSln& sln, Vector<ProcessingTreeNode>& out_nodes,
                              Vector<ProcessingInputRef>* out_inputs) {
	out_nodes.Clear();
	if(out_inputs)
		out_inputs->Clear();

	ProcessingTreeNode& root = out_nodes.Add();
	root.label = "Total";
	root.path = "";
	root.stage = "TOTAL";
	root.duration_ms = log.t_total_ms;
	root.status = log.status;
	root.note = "Complete Frame";
	root.is_leaf = false;
	root.step_id = -1;
	root.source_index = -1;

	VectorMap<String, int> path_nodes;
	path_nodes.Add("", 0);

	VectorMap<String, String> bbox_to_id;
	VectorMap<int, String> step_to_input;
	Vector<ProcessingInputRef> inputs;
	for(int i = 0; i < log.steps.GetCount(); i++) {
		const ProcessingStepRecord& s = log.steps[i];
		if(s.step_id < 0 || s.candidate_bbox.IsEmpty() || !StepUsesNamedInput(s))
			continue;

		String key = RectKey(s.candidate_bbox);
		int q = bbox_to_id.Find(key);
		String input_id;
		if(q < 0) {
			input_id = Format("I%d", bbox_to_id.GetCount() + 1);
			bbox_to_id.Add(key, input_id);
			ProcessingInputRef& in = inputs.Add();
			in.id = input_id;
			in.bbox = s.candidate_bbox;
		}
		else {
			input_id = bbox_to_id[q];
		}
		step_to_input.GetAdd(s.step_id) = input_id;
	}

	auto GetPathNode = [&](const String& path, int source_index) -> int {
		if(path.IsEmpty())
			return 0;
		Vector<String> parts = Split(path, '/');
		int parent = 0;
		String cur;
		for(const String& part : parts) {
			if(!cur.IsEmpty()) cur << "/";
			cur << part;
			int q = path_nodes.Find(cur);
			if(q < 0) {
				ProcessingTreeNode& n = out_nodes.Add();
				n.label = part;
				n.path = cur;
				n.stage = "GROUP";
				n.status = "OK";
				n.is_leaf = false;
				n.step_id = -1;
				n.source_index = source_index;
				int idx = out_nodes.GetCount() - 1;
				out_nodes[parent].children.Add(idx);
				path_nodes.Add(cur, idx);
				parent = idx;
			}
			else {
				parent = path_nodes[q];
			}
		}
		return parent;
	};

	for(int i = 0; i < log.steps.GetCount(); i++) {
		const ProcessingStepRecord& s = log.steps[i];
		ProcessingTreeNode leaf;
		if(!MapStepToTreeLeaf(s, i, sln, leaf))
			continue;
		if(leaf.step_id >= 0 && leaf.status != "SKIPPED" && (leaf.stage == "RECOGNIZE" || leaf.stage == "LABEL_A" || leaf.stage == "LABEL_B" || leaf.stage == "OCR")) {
			String input_id = step_to_input.Get(leaf.step_id, "");
			if(input_id.IsEmpty())
				input_id = "NA";
			if(!leaf.note.IsEmpty())
				leaf.note << " ";
			leaf.note << "input=" << input_id;
		}

		int parent = GetPathNode(leaf.path, i);
		int idx = out_nodes.GetCount();
		out_nodes.Add() <<= leaf;
		out_nodes[parent].children.Add(idx);
	}

	SortChildren(out_nodes, 0);
	if(out_inputs) {
		for(int i = 0; i < inputs.GetCount(); i++)
			out_inputs->Add() <<= inputs[i];
	}
}

END_UPP_NAMESPACE
