#ifndef ZY_COLLISION_MANAGER_H
#define ZY_COLLISION_MANAGER_H

#include "config_collisionManager.h"

#include <sofa/core/CollisionModel.h>

using namespace sofa;

class ZY_COLLISIONMANAGER_API ZyCollisionManager
{
    public:
        ZyCollisionManager() {}
        ~ZyCollisionManager() {}

	public:
		bool modelsCanCollide_BWListcheck(core::CollisionModel *cm1, core::CollisionModel *cm2);
};

#endif //ZY_COLLISION_MANAGER_H
