#include "AppMainGuiFlow.h"
#include "ProjectManager.h"
#include "AnnotationEditorWindow.h"
#include "SlotTrainerWindow.h"

namespace Upp {

void RunGuiFlow(const AppArguments& args) {
	WindowKind next_window = WIN_PROJECT_MANAGER;
	String next_sln_path;
	String next_annprj_path;
	String next_crops_dir;
	String next_annlay_path;
	String next_slot_key;
	String next_train_annprj_path;
	String next_train_images_dir;
	bool apply_editor_args = false;
	bool return_to_project_manager = false;

	switch(args.mode) {
	case AppArguments::MODE_PROJECT_MANAGER:
		next_window = WIN_PROJECT_MANAGER;
		break;
	case AppArguments::MODE_PROJECT_MANAGER_WITH_SLN:
		next_window = WIN_PROJECT_MANAGER;
		next_sln_path = args.annsln_path;
		break;
	case AppArguments::MODE_ANNOTATION_EDITOR_SINGLE_PROJECT:
		next_window = WIN_ANNOTATION_EDITOR;
		next_annprj_path = args.annprj_path;
		break;
	case AppArguments::MODE_ANNOTATION_EDITOR_ARGS:
		next_window = WIN_ANNOTATION_EDITOR;
		next_annprj_path = args.open_project;
		apply_editor_args = true;
		break;
	case AppArguments::MODE_SLOT_TRAINER_WINDOW:
		next_window = WIN_SLOT_TRAINER;
		next_crops_dir = args.crops_dir;
		next_annlay_path = args.annlay_path;
		next_slot_key = args.open_train_slot_key;
		next_train_annprj_path = args.open_train_annprj_path;
		next_train_images_dir = args.open_train_images_dir;
		break;
	case AppArguments::MODE_FRAME_RECOGNIZER_WINDOW:
		{
			// Need to load solution to get paths
			AnnSln sln;
			if(sln.Load(args.annsln_path)) {
				String sln_dir = GetFileDirectory(args.annsln_path);
				String annprj = NormalizePath(AppendFileName(sln_dir, sln.annprj));
				String model_set = args.open_fr_pass == 2 ? "pass2" : "pass1";

				FrameRecognizerWindow fr;
				fr.OpenSlideshow(sln, sln_dir, annprj, model_set);
				fr.Run();
			}
			return;
		}
	case AppArguments::MODE_TEST_PREFLIGHT:
		{
			ProjectManager pm;
			pm.TestPreflight(args.annsln_path, args.check_baseline_in_path, args.check_offset_baseline_in_path, args.mine_out_dir, args.split_file, args.fail_on_warnings_arg);
			return;
		}
	default:
		return;
	}

	while(next_window != WIN_EXIT) {
		if(next_window == WIN_PROJECT_MANAGER) {
			ProjectManager pm;
			if(!next_sln_path.IsEmpty())
				pm.OpenSln(next_sln_path);
			pm.Run();
			
			ProjectManager::OpenRequestType req = pm.GetOpenRequestType();
			if(req != ProjectManager::OPEN_NONE) {
				if(req == ProjectManager::OPEN_ANNOTATION_EDITOR) {
					next_window = WIN_ANNOTATION_EDITOR;
					next_annprj_path = pm.GetRequestedAnnprj();
					return_to_project_manager = true;
					continue;
				}
				if(req == ProjectManager::OPEN_SLOT_TRAINER) {
					next_window = WIN_SLOT_TRAINER;
					next_crops_dir = pm.GetRequestedCropsDir();
					next_annlay_path = pm.GetRequestedAnnlay();
					next_slot_key = pm.GetRequestedSlotKey();
					next_train_annprj_path = pm.GetRequestedAnnprj();
					next_train_images_dir = pm.GetRequestedImagesDir();
					return_to_project_manager = true;
					continue;
				}
			}
			next_window = WIN_EXIT;
			continue;
		}

		if(next_window == WIN_ANNOTATION_EDITOR) {
			AnnotationEditorWindow win;
			win.RegisterFocusActions();
			if(apply_editor_args) {
				ApplyAnnotationEditorArguments(win, args);
				apply_editor_args = false;
			}
			else if(!next_annprj_path.IsEmpty())
				win.LoadProject(next_annprj_path);
			win.Run();

			if(return_to_project_manager) {
				next_window = WIN_PROJECT_MANAGER;
				return_to_project_manager = false;
				next_annprj_path.Clear();
				continue;
			}
			next_window = WIN_EXIT;
			continue;
		}

		if(next_window == WIN_SLOT_TRAINER) {
			SlotTrainerWindow win;
			win.Configure(next_annlay_path, next_crops_dir, next_slot_key, next_train_annprj_path, next_train_images_dir);
			win.Run();

			if(return_to_project_manager) {
				next_window = WIN_PROJECT_MANAGER;
				return_to_project_manager = false;
				next_annlay_path.Clear();
				next_crops_dir.Clear();
				next_slot_key.Clear();
				next_train_annprj_path.Clear();
				next_train_images_dir.Clear();
				continue;
			}
			next_window = WIN_EXIT;
			continue;
		}
	}
}

}
