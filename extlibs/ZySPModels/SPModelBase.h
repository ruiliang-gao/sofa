#ifndef SPMODELS_MODELBASE_H
#define SPMODELS_MODELBASE_H

#include <sofa/defaulttype/Vec.h>

namespace SPModels
{
    using namespace sofa::defaulttype;

    class SPModelBase
    {
        public:
            SPModelBase();
            virtual ~SPModelBase();
    };

}

#endif //SPMODELS_MODELBASE_H
