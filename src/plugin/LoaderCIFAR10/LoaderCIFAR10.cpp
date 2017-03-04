#include "LoaderCIFAR10.h"

namespace ConvNet {
using namespace Upp;

LoaderCIFAR10::LoaderCIFAR10() {
	ImageDraw id(64, 64);
	id.DrawRect(0,0,64,64,Black());
	id.DrawText(0,0, "C", Arial(64), White());
	Icon(id);
	
	Title("Loading CIFAR-10 images");
	Add(lbl.HSizePos(4, 4).TopPos(3, 24));
	Add(prog.HSizePos(4, 4).TopPos(33, 12));
	Add(sub.HSizePos(4, 4).TopPos(33+15, 12));
	Add(cancel.HCenterPos(100).TopPos(63, 24));
	SetRect(0, 0, 400, 90);
	cancel.SetLabel("Cancel");
	cancel <<= THISBACK(Cancel);
	prog.Set(0, 1);
	sub.Set(0, 1);
	ret_value = 0;
	
	
	ImageBank& bank = GetImageBank();
	bank.classes.Clear();
	bank.classes.Add("airplane");
	bank.classes.Add("automobile");
	bank.classes.Add("bird");
	bank.classes.Add("cat");
	bank.classes.Add("deer");
	bank.classes.Add("dog");
	bank.classes.Add("frog");
	bank.classes.Add("horse");
	bank.classes.Add("ship");
	bank.classes.Add("truck");
	
	fast_cache = false;
	
	Thread::Start(THISBACK(Load));
}

void LoaderCIFAR10::Progress(int actual, int total, String label) {
	LOG("Progress " << actual << "/" << total);
	prog.Set(actual, total);
	sub.Set(0, 1);
	lbl.SetLabel(label);
}

bool LoaderCIFAR10::SubProgress(int actual, int total) {
	GuiLock __;
	sub.Set(actual, total);
	return false;
}

void LoaderCIFAR10::Load() {
	
	ImageBank& bank = GetImageBank();
	
	// Try to load cached file, which is a lot faster than extracting included files.
	if (fast_cache) {
		TimeStop ts;
		FileIn fast_file(ConfigFile("cifar_databank.bin"));
		if (fast_file.IsOpen()) {
			PostCallback(THISBACK3(Progress, 0, 1, "Loading existing cache: cifar_databank.bin"));
			fast_file % bank;
			if (bank.images.GetCount() == 50000 && bank.labels.GetCount() == 50000 &&
				bank.test_images.GetCount() == 10000 && bank.test_labels.GetCount() == 10000) {
				LOG("Loaded data from cifar_databank.bin successfully in " << ts.ToString());
				PostCallback(THISBACK(Close0));
				return;
			} else {
				PromptOK("Error: loading fast cache file cifar_databank.bin failed.");
				ret_value = 1;
				PostCallback(THISBACK(Close0));
				return;
			}
		}
	}
	
	bank.images.Clear();
	bank.labels.Clear();
	bank.test_images.Clear();
	bank.test_labels.Clear();
	
	#ifdef LITTLEENDIAN
		#define SWAP32(x) x = SwapEndian32(x)
	#else
		#define SWAP32(x)
	#endif
	
	// .brc file embedding fails with these files
	int cifar_data_count = 6;
	
	int actual = 0;
	int total = cifar_data_count * 2 + 1;
	for(int i = 0; i < cifar_data_count && !IsFail(); i++) {
		String file = i < 5 ? "data_batch_" + IntStr(i+1) + ".bin" : "test_batch.bin";
		if (!FileExists(GetExeDirFile(file))) {
			PromptOK("Error: CIFAR-10 dataset file " + file + " is not included with this executable.");
			ret_value = 1;
			PostCallback(THISBACK(Close0));
			return;
		}
		
		FileIn in(GetExeDirFile(file));
		
		int length = in.GetSize();
		LOG("Load " <<  file << " size: " << length);
		actual++;
		if (IsFail()) break;
		
		PostCallback(THISBACK3(Progress, actual, total, "Reading: " + file));
		in.Seek(0);
		
		#undef ASSERT
		#define ASSERT(x) if (!(x)) {PromptOK("Reading failed in file '" + file + "'. Reason: '" #x "' is false"); ret_value = 1; PostCallback(THISBACK(Close0)); return;}
		
		byte cls;
		int items, rows, cols;
		bool main = file.Left(10) == "data_batch";
		
		int row_size = 1 + 32 * 32 * 3;
		
		items = in.GetSize() / row_size;
		rows = 32;
		cols = 32;
		ASSERT(items == 10000);
		
		byte r, g, b;
		Vector<Image>& images = main ? bank.images : bank.test_images;
		Vector<int>& labels = main ? bank.labels : bank.test_labels;
		int base = images.GetCount();
		images.SetCount(base + items);
		labels.SetCount(base + items);
		for(int j = 0; j < items && !in.IsEof() && !IsFail(); j++) {
			in.Get(&cls, 1);
			//SWAP32(magic);
			ASSERT(cls >= 0 && cls < 10);
			
			labels[base + j] = cls;

			ImageBuffer ib(cols, rows);
			RGBA* it;
			
			it = ib.Begin();
			for (int y = 0; y < rows && !in.IsEof(); y++) {
				for (int x = 0; x < cols && !in.IsEof(); x++) {
					in.Get(&r, 1);
					it->a = 255;
					it->r = r;
					it++;
				}
			}
			
			it = ib.Begin();
			for (int y = 0; y < rows && !in.IsEof(); y++) {
				for (int x = 0; x < cols && !in.IsEof(); x++) {
					in.Get(&g, 1);
					it->g = g;
					it++;
				}
			}
			
			it = ib.Begin();
			for (int y = 0; y < rows && !in.IsEof(); y++) {
				for (int x = 0; x < cols && !in.IsEof(); x++) {
					in.Get(&b, 1);
					it->b = b;
					it++;
				}
			}
			
			if ((j % 100) == 0)
				SubProgress(j, items);
			images[base + j] = ib;
		}
		
		
		LOG("Read OK: " << file);
		
		actual++;
	}
	
	if (fast_cache && !IsFail()) {
		PostCallback(THISBACK3(Progress, actual, total, "Saving fast cache takes few minutes: cifar_databank.bin"));
		FileOut fast_file(ConfigFile("cifar_databank.bin"));
		fast_file % bank;
	}
	
	PostCallback(THISBACK(Close0));
}

}
