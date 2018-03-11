#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import shutil
from Cheetah.Template import Template

template = """
#encoding utf-8
#compiler-settings
commentStartToken = //
cheetahVarStartToken = autopack::
#end compiler-settings

### CMakeLists.txt are generated from our lovely sofa-pck-manager tools 
### In case you need to customize this CMakeLists.txt please copy our template from
### sofa-pck-packages/templates/CMakeLists.tpl and drop it into you current directory
### then launch sofa-pck-manager regenerate CMakeLists.txt
cmake_minimum_required(VERSION 3.1)
project(autopack::package_name)

#for depend in autopack::dependencies
find_package(autopack::depend)
#end for

set(HEADER_FILES
#for filename in autopack::header_files   
    autopack::filename
#end for
)

set(SOURCE_FILES
#for filename in autopack::source_files 
    autopack::filename
#end for
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Messaging)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_HELPER_TYPES")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deprecated_layout>")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")

"""

def generateCMakeLists(packagename):
    finalpath = packagename.replace('.', '/')
    theFile = open(finalpath+"/CMakeLists.txt", "w+")
    
    t = Template(template, searchList=[loadPackage(packagename)])
    theFile.write(str(t))


def initCMakeLists(path, projectname):
    theFile = open(path+"/CMakeLists.txt", "w+")
    
    params = {
        "projectname" : projectname,
        "dependencies" : [],
        "source_files" : [],
        "header_files" : []
    } 
    t = Template(template, searchList=[params])
    theFile.write(str(t))

def loadSpm(spmfilename):
    return json.loads( open(spmfilename).read() ) 

def addPackageGroup(name):
    print("Adding a package group: "+name)

def savePackage(packagename, package):
    finalpath = packagename.replace('.', '/')
    spmdir = finalpath+"/"+".spm"
    spmfile = spmdir+"/config.json"      
    open(spmfile, "w").write( json.dumps( package ) )        

def loadPackage(packagename):
    finalpath = packagename.replace('.', '/')
    spmdir = finalpath+"/"+".spm"
    spmfile = spmdir+"/config.json"      
    if os.path.exists(spmfile):
        spm = loadSpm(spmfile)
        if spm["package_name"] == packagename:
            return spm
    return None

def addToProperty(packagename, name, value):
    print("- adding to property '"+name+"' to package '"+packagename+"'")
    package = loadPackage(packagename)
    if name not in package:
        print("  failure: property '"+name+"' alreay exists in package "+packagename)
    elif value not in package[name]:
        package[name].append(value)
    savePackage(packagename, package)   

def addProperty(packagename, name, value):
    print("- adding property '"+name+"' to package '"+packagename+"'")
    package = loadPackage(packagename)
    if name not in package:
        package[name] = value
    else:
        print("  failure: property '"+name+"' alreay exists in package "+packagename)
    savePackage(packagename, package)    
   
def addPackage(name, wipeall=True):
    finalpath = name.replace('.', '/')
    
    print("- adding package: '"+name+"'")
    print("            path: '"+finalpath+"'") 
    parentpath=""
    pathdec = name.split('.')
    for p in pathdec[:-1]:
        parentpath+=p+"/"
        
        if wipeall and os.path.exists(parentpath):    
            shutil.rmtree(finalpath.split('/')[0])

        if os.path.exists(parentpath):
            print("path exist:"+parentpath) 
        else:
            print("CREATE GROUP: "+parentpath)
            spmdir = parentpath+"/"+".spm"
            spmfile = spmdir+"/config.json"            
            os.mkdir(parentpath)
            os.mkdir(spmdir)
            open(spmfile, "w").write( json.dumps({"package-type":"group"}) )
            
    if pathdec[-1]:       
            if os.path.exists(finalpath):
                print("path exist:"+finalpath) 
                spmdir = finalpath+"/"+".spm"
                spmfile = spmdir+"/config.json"      
                if os.path.exists(spmfile):
                    print(" there is a spm file in:"+finalpath)                 
                    spm = loadSpm(spmfile)
                    if spm["package_name"] == name:
                        print("        this is package:"+name)                                     
            else:
                print("CREATE PACKAGE: "+finalpath)
            
                os.mkdir(finalpath)
                spmdir = finalpath+"/"+".spm"
                os.mkdir(spmdir)

                spmfile = spmdir+"/config.json"                      
                open(spmfile, "w").write( json.dumps({"package-type":"single"}) )
                initCMakeLists(finalpath, name)    
        
print("--== Sofa Package Manager v1.0 ==--")

addPackage("Sofa.Helper.Core", wipeall=False)
addProperty("Sofa.Helper.Core", "dependencies", [])
addToProperty("Sofa.Helper.Core", "dependencies", value="Sofa.Config")
addToProperty("Sofa.Helper.Core", "dependencies", value="Sofa.Messaging")
addToProperty("Sofa.Helper.Core", "header_files", value="src/sofa/helper/core/file1.h")
addToProperty("Sofa.Helper.Core", "source_files", value="src/sofa/helpr/core/file1.cpp")
addToProperty("Sofa.Helper.Core", "git", value="src/sofa/helpr/core/file1.cpp")

generateCMakeLists("Sofa.Helper.Core")

