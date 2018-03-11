#ifndef SOFA_HELPER_TYPES_TYPEINFOS_H
#define SOFA_HELPER_TYPES_TYPEINFOS_H

#include <sofa/helper/types/config.h>
#include <string>
#include <typeinfo>

namespace sofa
{
namespace helper
{
namespace types
{
/// Decode the type's name to a more readable form if possible
std::string SOFA_HELPER_TYPES_API gettypename(const std::type_info& t);

}
}
}

#endif // 
