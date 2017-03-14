#include "LoaderCIFAR10.h"

namespace ConvNet {
using namespace Upp;

LoaderCIFAR10::LoaderCIFAR10(Session& ses) {
	this->ses = &ses;
	
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
	
	#ifdef CPU_LITTLE_ENDIAN
		#define SWAP32(x) x = SwapEndian32(x)
	#else
		#define SWAP32(x)
	#endif
	
	// .brc file embedding fails with these files
	int cifar_data_count = 6;
	
	SessionData& d = ses->Data();
	
	d.BeginDataClass<VolumeDataDivider<byte, 255> >(10, 50000, 32, 32, 3, 10000);
	
	d.SetClass(0, "airplane");
	d.SetClass(1, "automobile");
	d.SetClass(2, "bird");
	d.SetClass(3, "cat");
	d.SetClass(4, "deer");
	d.SetClass(5, "dog");
	d.SetClass(6, "frog");
	d.SetClass(7, "horse");
	d.SetClass(8, "ship");
	d.SetClass(9, "truck");
	
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
		int items, rows, cols, len;
		bool main = file.Left(10) == "data_batch";
		
		int row_size = 1 + 32 * 32 * 3;
		
		items = in.GetSize() / row_size;
		rows = 32;
		cols = 32;
		len = rows * cols * 3;
		ASSERT(items == 10000);
		
		int base = i < 5 ? i * 10000 : 0;
		for(int j = 0; j < items && !in.IsEof() && !IsFail(); j++) {
			in.Get(&cls, 1);
			//SWAP32(magic);
			ASSERT(cls >= 0 && cls < 10);
			
			if (main)
				d.SetLabel(base + j, cls);
			else
				d.SetTestLabel(base + j, cls);
			
			VolumeDataBase& out = main ? d.Get(base + j) : d.GetTest(base + j);
			
			for (int clr = 0; clr < 3; clr++) {
				for (int y = 0; y < rows; y++) {
					for (int x = 0; x < cols; x++) {
						byte pixel;
						in.Get(&pixel, 1);
						out.Set(x, y, clr, cols, 3, pixel / 255.0);
					}
				}
			}
			
			if ((j % 100) == 0)
				SubProgress(j, items);
		}
		
		LOG("Read OK: " << file);
		
		actual++;
	}
	
	d.EndData();
	
	PostCallback(THISBACK(Close0));
}

}
