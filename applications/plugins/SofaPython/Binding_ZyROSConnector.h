/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef BINDING_ZYROSCONNECTOR_H
#define BINDING_ZYROSCONNECTOR_H

#include "PythonMacros.h"

#include <SofaROSConnector/ZyROSConnector/ZyROSConnector.h>
#include <SofaROSConnector/ZyROSConnectionManager/ZyROSConnectionManager.h>

#include <SofaROSConnector/ZyROSConnector/ZyROSConnectorTopicPublisher.h>
#include <SofaROSConnector/ZyROSConnector/ZyROSConnectorTopicSubscriber.h>
#include <SofaROSConnector/ZyROSConnector/ZyROSConnectorServiceClient.h>
#include <SofaROSConnector/ZyROSConnector/ZyROSConnectorServiceServer.h>

SP_DECLARE_CLASS_TYPE(ZyROSConnector)
SP_DECLARE_CLASS_TYPE(ZyROSConnectionManager)

SP_DECLARE_CLASS_TYPE(ZyROSConnectorTopicPublisher)
SP_DECLARE_CLASS_TYPE(ZyROSConnectorTopicSubscriber)
SP_DECLARE_CLASS_TYPE(ZyROSConnectorServiceClient)
SP_DECLARE_CLASS_TYPE(ZyROSConnectorServiceServer)

#endif // BINDING_ZYROSCONNECTOR_H
