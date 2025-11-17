#include "CLIPTester.h"

#define IMAGECLASS CLIPImg
#define IMAGEFILE <CLIPTester/CLIPTester.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
    try {
        CLIPApp clip_app;
        clip_app.Run();
    }
    catch (Exc e) {
        PromptOK(e);
    }
}