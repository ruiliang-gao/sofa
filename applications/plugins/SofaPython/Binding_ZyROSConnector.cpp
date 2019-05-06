#include "Binding_ZyROSConnector.h"

#include "PythonToSofa.inl"

using namespace Zyklio::ROSConnector;
using namespace Zyklio::ROSConnectionManager;

static inline ZyROSConnector* get_ZyROSConnector(PyObject* obj)
{
    return sofa::py::unwrap<ZyROSConnector>(obj);
}

static inline ZyROSConnectionManager* get_ZyROSConnectionManager(PyObject* obj)
{
    return sofa::py::unwrap<ZyROSConnectionManager>(obj);
}

static inline ZyROSPublisher* get_ZyROSConnectorTopicPublisher(PyObject* obj)
{
    return sofa::py::unwrap<ZyROSPublisher>(obj);
}

static inline ZyROSListener* get_ZyROSListener(PyObject* obj)
{
    return sofa::py::unwrap<ZyROSListener>(obj);
}

static inline ZyROSServiceClient* get_ZyROSConnectorServiceClient(PyObject* obj)
{
    return sofa::py::unwrap<ZyROSServiceClient>(obj);
}

static inline ZyROSConnectorServiceServer* get_ZyROSConnectorServiceServer(PyObject* obj)
{
    return sofa::py::unwrap<ZyROSConnectorServiceServer>(obj);
}

static PyObject* ZyROSConnector_startComponent(PyObject* self, PyObject* /*args*/)
{
    ZyROSConnector* obj = get_ZyROSConnector(self);
    obj->startComponent();
    Py_RETURN_NONE;
}

static PyObject* ZyROSConnector_stopComponent(PyObject* self, PyObject* /*args*/)
{
    ZyROSConnector* obj = get_ZyROSConnector(self);
    obj->stopComponent();
    Py_RETURN_NONE;
}

static PyObject* ZyROSConnector_pauseComponent(PyObject* self, PyObject* /*args*/)
{
    ZyROSConnector* obj = get_ZyROSConnector(self);
    obj->pauseComponent();
    Py_RETURN_NONE;
}

static PyObject* ZyROSConnector_resumeComponent(PyObject* self, PyObject* /*args*/)
{
    ZyROSConnector* obj = get_ZyROSConnector(self);
    obj->resumeComponent();
    Py_RETURN_NONE;
}

static PyObject* ZyROSConnector_isConnected(PyObject* self, PyObject* /*args*/)
{
    ZyROSConnector* obj = get_ZyROSConnector(self);
    bool ret = obj->isConnected();
    return PyBool_FromLong((long) ret);
}

static PyObject* ZyROSConnector_isThreadRunning(PyObject* self, PyObject* /*args*/)
{
    ZyROSConnector* obj = get_ZyROSConnector(self);
    bool ret = obj->isThreadRunning();
    return PyBool_FromLong((long) ret);
}

static PyObject* ZyROSConnector_setRosMasterURI(PyObject* self, PyObject* args)
{
    char* rosMasterURI;
    if (!PyArg_ParseTuple(args, "s", &rosMasterURI))
    {
        return nullptr;
    }

    ZyROSConnector* obj = get_ZyROSConnector(self);
    std::string masterURI(rosMasterURI);

    bool ret = obj->setRosMasterURI(masterURI);
    return PyBool_FromLong((long) ret);
}

/*bool addTopicListener(const boost::shared_ptr<ZyROSListener> &);
bool removeTopicListener(boost::shared_ptr<ZyROSListener>&);

bool addTopicPublisher(boost::shared_ptr<ZyROSPublisher>&);
bool removeTopicPublisher(boost::shared_ptr<ZyROSPublisher>&);

size_t getNumTopicListeners() const;
size_t getNumTopicPublishers() const;*/

SP_CLASS_METHODS_BEGIN(ZyROSConnector)
SP_CLASS_METHOD(ZyROSConnector, startComponent)
SP_CLASS_METHOD(ZyROSConnector, stopComponent)
SP_CLASS_METHOD(ZyROSConnector, pauseComponent)
SP_CLASS_METHOD(ZyROSConnector, resumeComponent)
SP_CLASS_METHOD(ZyROSConnector, isConnected)
SP_CLASS_METHOD(ZyROSConnector, isThreadRunning)
SP_CLASS_METHOD(ZyROSConnector, setRosMasterURI)
/*SP_CLASS_METHOD(ZyROSConnector, addTopicListener)
SP_CLASS_METHOD(ZyROSConnector, addTopicPublisher)
SP_CLASS_METHOD(ZyROSConnector, removeTopicListener)
SP_CLASS_METHOD(ZyROSConnector, removeTopicPublisher)*/
SP_CLASS_METHODS_END
