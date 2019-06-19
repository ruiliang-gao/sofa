#include <sofa/helper/system/config.h>
#include <initZyPipeline.h>


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
