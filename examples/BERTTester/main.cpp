#include "BERTTester.h"

#define IMAGECLASS BERTTesterImg
#define IMAGEFILE <BERTTester/BERTTester.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
	try {
		BERTTester bert;

		// Here you would typically load text data for BERT training
		// This would involve tokenizing text and preparing for MLM and NSP tasks

		bert.Init();

		bert.Run();
	}
	catch (Exc e) {
		PromptOK(e);
	}
}