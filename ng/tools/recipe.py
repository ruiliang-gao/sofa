import json
from Cheetah.Template import Template

def lprint(l):
    print(json.dumps(l, indent=4, separators=(',', ':')))
                
def actionFromTemplate(package_name, name, srcpath, srcincpath,  dstpath=None, oldheader=None, newheader=None, types=[".h", ".cpp"], forward=None):
    dstpath = package_name.replace(".", "/")+"/src/"+(package_name.lower().replace(".","/"))
    if oldheader == None:
        oldheader = package_name.upper()
    oldheader += "_"+name.upper()
    
    if newheader == None:
        newheader = (package_name.upper()+"_"+name.upper()).replace(".","_")

    tasks = []
    for t in types:    
        EXT = t.upper().replace(".","_")
        tasks.append(["move", srcpath+"/"+name+t, dstpath+"/"+name+t]) 
        tasks.append(["spm", "package", package_name, "property", "source_files", "add-to", dstpath+"/"+name+t])
        tasks.append(["rename", package_name.replace(".","/"), "#include <"+srcincpath+"/"+name+t+">", "#include <"+dstpath+"/"+name+t+">"])
        tasks.append(["fixheader", dstpath+"/"+name+t, oldheader+EXT, newheader+EXT])
        if forward != None and t == ".h":
            tasks.append(["mkforward", package_name.replace(".","/"), srcincpath+"/"+name+t, dstpath+"/"+name+t, forward])
   
    return tasks
    
package_name = "Sofa.Helper.Types"
srcpath = "../../SofaKernel/framework/sofa/helper"
srcincpath = "sofa/helper"
oldheader = "SOFA_HELPER"
tasks = [ ["spm", "package", package_name, "init"] ]
tasks += [["mkdir", "Sofa/Helper/Types/src/sofa/helper/types"]]
tasks += [["mkdir", "Sofa/Helper/Types/deprecated_layout/sofa/helper/"]]
tasks += actionFromTemplate(package_name, "OptionsGroup", srcpath, srcincpath, oldheader=oldheader)  
tasks += actionFromTemplate(package_name, "vector", srcpath, srcincpath,oldheader=oldheader, forward = {"sofa" : {"helper" : {"using" : " sofa::helper::types::vector ;"}}})  
tasks += actionFromTemplate(package_name, "deque", srcpath, srcincpath, oldheader=oldheader, types=[".h"])  

tasks += [ ["spm", "package", package_name, "generate-cmakelists"] ]


lprint({"commands" : tasks})
   


