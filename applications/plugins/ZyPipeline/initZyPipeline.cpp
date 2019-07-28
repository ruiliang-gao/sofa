#include "initZyPipeline.h"

extern "C" {
    void initExternalModule()
    {
        static bool first = true;
        if (first)
        {
            first = false;
        }
    }

    const char* getModuleName()
    {
        return "ZyPipeline";
    }

    const char* getModuleVersion()
    {
        return "0.0.2";
    }

    const char* getModuleLicense()
    {
        return "LGPL";
    }

    const char* getModuleDescription()
    {
        return "Zykl.io collision detection pipeline replacement.";
    }

    const char* getModuleComponentList()
    {
        return "ZyPipeline, ZyDefaultPipeline";
    }
}

SOFA_LINK_CLASS(ZyPipeline)
SOFA_LINK_CLASS(ZyDefaultPipeline)



namespace Zyklio
{
	namespace Pipeline
	{
        void initZyPipeline()
		{
			static bool first = true;
			if (first)
			{
				first = false;
			}
		}

        SOFA_LINK_CLASS(ZyDefaultPipeline)
	} // namespace component
} // namespace sofa
