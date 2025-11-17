#include "T5Tester.h"

#define IMAGECLASS T5Img
#define IMAGEFILE <T5Tester/T5Tester.iml>
#include <Draw/iml_source.h>

GUI_APP_MAIN
{
    try {
        T5App t5_app;
        t5_app.Run();
    }
    catch (Exc e) {
        PromptOK(e);
    }
}