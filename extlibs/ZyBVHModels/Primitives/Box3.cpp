#ifndef _WIN32
#include "Box3.inl"
#else
#include "Box3.h"
#endif

using namespace BVHModels;

template class Box3<SReal>;
