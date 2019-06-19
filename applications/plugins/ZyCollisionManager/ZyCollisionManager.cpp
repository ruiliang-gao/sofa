#include "ZyCollisionManager.h"

#include <sofa/defaulttype/Vec.h>

bool ZyCollisionManager::modelsCanCollide_BWListcheck(core::CollisionModel *cm1, core::CollisionModel *cm2)
{
    std::cout << "Checking collision lists for " << cm1->getName() << " - " << cm2->getName() << std::endl;
    core::CollisionModel * cm1Root = cm1->getLast();
    core::CollisionModel * cm2Root = cm2->getLast();

    const std::string& m1Name = cm1Root->getName();
    const std::string& m2Name = cm2Root->getName();

    const helper::vector<std::string>& m1Blacklist = cm1Root->getCollisionModelBlacklist();
    const helper::vector<std::string>& m2Blacklist = cm2Root->getCollisionModelBlacklist();
    const helper::vector<std::string>& m1Whitelist = cm1Root->getCollisionModelWhitelist();
    const helper::vector<std::string>& m2Whitelist = cm2Root->getCollisionModelWhitelist();

    bool onCollisionWhitelist = false;
    bool onCollisionBlacklist = false;
    bool whitelistRelevant = ((m1Whitelist.size() > 0) || (m2Whitelist.size() > 0));
    bool blacklistRelevant = ((m1Blacklist.size() > 0) || (m2Blacklist.size() > 0));

    //std::cout << " cm1Root/cm2Root: " << m1Name << "," << m2Name << std::endl;

    if (whitelistRelevant && blacklistRelevant)
    {
        //helper::system::sout << "WARNING: There exists at least one non-empty whitelist and one non-empty blacklist for the contact " << cm1Root->getName() << "-" << cm2Root->getName() << ". The blacklists are ignored." << std::endl;
        std::cout << "WARNING: There exists at least one non-empty whitelist and one non-empty blacklist for the contact " << cm1Root->getName() << "-" << cm2Root->getName() << ". The blacklists are ignored." << std::endl;
        blacklistRelevant = false;
    }

    // checks if (at least) one collision model's name appears in the whitelist of the other
    // (but not if both whitelists are empty - in that case, check the blacklists)
    if (whitelistRelevant)
    {
        //std::cout << "1- Whitelist Relevant" << std::endl;
        for (helper::vector<std::string>::const_iterator m1_it = m1Whitelist.begin(); m1_it != m1Whitelist.end(); ++m1_it)
        {
            if (m2Name.compare(*m1_it) == 0)
            {
                //std::cout << m1Name << " has a whitelist entry for " << m2Name << std::endl;
                onCollisionWhitelist = true;
                break;
            }
        }
        for (helper::vector<std::string>::const_iterator m2_it = m2Whitelist.begin(); m2_it != m2Whitelist.end(); ++m2_it)
        {
            if (m1Name.compare(*m2_it) == 0)
            {
                //std::cout << m2Name << " has a whitelist entry for " << m1Name << std::endl;
                onCollisionWhitelist = true;
                break;
            }
        }

    }
    // The blacklists are ignored if the at least one whitelist is not empty
    else
    {
        //std::cout << "1- Whitelist NOT Relevant" << std::endl;
        // checks if (at least) one collision model's name appears in the blacklist of the other
        // (but not if both blacklists are empty)
        if (blacklistRelevant)
        {
            //std::cout << "1- Blacklist Relevant" << std::endl;
            for (helper::vector<std::string>::const_iterator m1_it = m1Blacklist.begin(); m1_it != m1Blacklist.end(); ++m1_it)
            {
                if (m2Name.compare(*m1_it) == 0)
                {
                    //std::cout << m1Name << " has a blacklist entry for " << m2Name << std::endl;
                    onCollisionBlacklist = true;
                    break;
                }
            }
            for (helper::vector<std::string>::const_iterator m2_it = m2Blacklist.begin(); m2_it != m2Blacklist.end(); ++m2_it)
            {
                if (m1Name.compare(*m2_it) == 0)
                {
                    //std::cout << m2Name << " has a blacklist entry for " << m1Name << std::endl;
                    onCollisionBlacklist = true;
                    break;
                }
            }
        }
    }

    if ((whitelistRelevant  && onCollisionWhitelist)
        || (blacklistRelevant  && !onCollisionBlacklist)
        || (!blacklistRelevant && !whitelistRelevant)
        )
    {
		// noop?
		return true;
    }
    else 
	{
        return false;
    }
}
