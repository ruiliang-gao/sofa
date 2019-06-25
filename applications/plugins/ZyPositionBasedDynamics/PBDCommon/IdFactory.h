#ifndef IDFACTORY_H
#define IDFACTORY_H

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            /** Factory for unique ids.
              */
            class IDFactory
            {
            private:
                    /** Current id */
                    static int id;

            public:
                    static int getId() { return id++; }
            };
        }
    }
}

#endif // IDFACTORY_H
