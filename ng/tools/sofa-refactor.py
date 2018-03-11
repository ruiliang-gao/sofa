#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import re
import shutil
import spm
from Cheetah.Template import Template


# 
# sofa-pck-manager add Sofa.Helper.Types 

# Move the files (to git)
# git mv

# Add  the files to the package
# sofa-pck-manager package Sofa.Helper.Types add src/*

# Build the CMakeLists
# sofa-pck-manager package Sofa.Helper.Types generate-cmakelists

template = """
#ifndef $INCLUDE_GUARD
#define $INCLUDE_GUARD

// The content of this file has been refactored and moved to this new location 
// The best practice is now to update your code. By updating the including points
// as well as the namespace
$INCLUDE_PATH_FORWARD

$NS_FORWARD

#endif // $INCLUDE_GUARD
"""

def toCName(p):
    return p.upper().replace("/","_").replace(".","_")

def toCNS(p):
    ns = ""
    for k in p:
        v = p[k]      
        if isinstance(v, dict):
            ns += "namespace "+k+" \n"
            ns += "{\n"
            ns += toCNS(v) 
            ns += "}\n"
        else:
            ns += "    "+k + v + "\n"   
    return ns

def forwardCmd(cmd):
    print("forwarding: " + str(cmd))
    packagename = cmd[0]
    finalpath = packagename+"/deprecated_layout/"
    oldpath=cmd[1]
    newpath=cmd[2]
    nsmap=cmd[3]

    if not os.path.exists(finalpath):
        os.mkdir(finalpath)
    
    theFile = open(finalpath+"/"+oldpath, "w")
    templateMap = {
        "INCLUDE_GUARD" : toCName(oldpath),
        "INCLUDE_PATH_FORWARD" : "#include <"+newpath+">",
        "NS_FORWARD" : toCNS(nsmap) 
    }
    t = Template(template, searchList=[templateMap])
    theFile.write(str(t))

def branchCmd(cmd):
    print("branch: git" + str(cmd))

def mkdirCmd(cmd):
    print("mkdirs:" + str(cmd))
    if not os.path.exists(cmd[0]):
        os.makedirs(cmd[0])

def commitCmd(cmd):
    print("commit: git" + str(cmd))

def moveCmd(cmd):
    print("move: " + str(cmd))
    shutil.copy(cmd[0], cmd[1])

def fixheaderCmd(cmd):
    print("fixheader: " + str(cmd))
    path = cmd[0] 
    tmp = open(path+"_tmp","w")

    for line in open(path, "r"):
        iline = re.sub(cmd[1], cmd[2], line)
        if iline != line:
            print("- fix "+path+": "+iline),
        tmp.write(iline)        
    tmp.close()
    shutil.copyfile(path+"_tmp", path)
    os.remove(path+"_tmp")

def spmCmd(cmd):
    print("spm: " + str(cmd))
    pckcmd = cmd[0] 
    if pckcmd == "package":
        packagename = cmd[1]        
        target = cmd[2]
        if target == "property":
            targetname = cmd[3]
            targetcmd = cmd[4]
            value = cmd[5]
            if targetcmd == "add-to":
                spm.addToProperty(packagename, targetname, value) 
            else:
                print("Invalid property cmd:" +str(cmd))
        elif target == "init":
                spm.initPackage(packagename)
        elif target == "generate-cmakelists":
                spm.generateCMakeLists(packagename)
        else:
            print("Invalid property:" +str(cmd))
    else:
        print("Invalid smd cmd:" +str(cmd))
            
def renameCmd(cmd):
    print("rename: " + str(cmd))
    rootpath = cmd[0] 
    for (rootpath, dirname, filenames) in os.walk(rootpath):
        for filename in filenames:
            path = rootpath+"/"+filename
            tmp = open(path+"_tmp","w")
            for line in open(path, "r"):
                iline = re.sub(cmd[1], cmd[2], line)
                if iline != line:
                    print("- rename "+path+": "+iline),
                
                tmp.write(iline)
            
            tmp.close()
            shutil.copyfile(path+"_tmp", path)
            os.remove(path+"_tmp")

def replayCmd(cmd):
    if cmd[0] == "rename":
        renameCmd(cmd[1:])    
    elif cmd[0] == "move":
        moveCmd(cmd[1:])    
    elif cmd[0] == "mkdir":
        mkdirCmd(cmd[1:])    
    elif cmd[0] == "spm":
        spmCmd(cmd[1:])    
    elif cmd[0] == "commit":
        commitCmd(cmd[1:]) 
    elif cmd[0] == "branch":
        branchCmd(cmd[1:]) 
    elif cmd[0] == "fixheader":
        fixheaderCmd(cmd[1:])     
    elif cmd[0] == "mkforward":
        forwardCmd(cmd[1:])   
    else:
        print("Invalid commands")
        
def replayRefactoring(filename):
    cmds = json.loads( open(filename).read() )["commands"]
    for cmd in cmds:
        replayCmd(cmd)
        print("")

import sys
replayRefactoring(sys.argv[1])    

