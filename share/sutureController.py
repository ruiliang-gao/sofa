import Sofa
import sys
import numpy as np
from random import randint, uniform

class SofaVisitor(object):
    def __init__(self,name):
        print 'SofaVisitor constructor name='+name
        self.name = name

    def processNodeTopDown(self,node):
        # print 'SofaVisitor "'+self.name+'" processNodeTopDown node='+node.findData('name').value
        return -1

    def processNodeBottomUp(self,node):
        return -1 # dag
        # print 'SofaVisitor "'+self.name+'" processNodeBottomUp node='+node.findData('name').value

    def treeTraversal(self):
        # print 'SofaVisitor "'+self.name+'" treeTraversal'
        return -1 # dag
    def testFunction(self,node):
        print 'testFunction running...node=' + node.name              
        #you can animate simulation directly by uncommenting the following line:
        # node.animate=true      
        return 0

class SutureController(Sofa.PythonScriptController):
    # called once the script is loaded
    def onLoaded(self,node):
        self.rootNode = node.getRoot()
        print 'Controller script loaded from node '+node.findData('name').value
        return 0

    # optionnally, script can create a graph...
    def createGraph(self,node):
        print 'createGraph called (python side)'
        return 0


    # called once graph is created, to init some stuff...
    def initGraph(self,node):
        print 'initGraph called (python side), nodename='+node.name
        v = SofaVisitor('PythonVisitor')
        v.testFunction(self.rootNode)
        node.executeVisitor(v)
        self.solverNode= node.getChild("SolverNode")
        self.sutureNode = self.solverNode.getChild("SutureNode")
        self.findHaptics(self.rootNode)
        name1 = "fundus_old"
        name2 = ""
        if name1 != "":
            self.sutureObj1 = self.solverNode.getChild(name1)
            self.sutureObj1TriSurf = self.sutureObj1.getChild("CollisionOuter").getObject("TriangleModel")
            # self.sutureObj1FixConstraints = self.sutureObj1.getObject("partialFixedConstraint")
        self.sutureApplied = False

        return 0

    def applySuture(self):   
        if self.sutureObj1TriSurf:
            sutureTag1 = self.sutureObj1TriSurf.findData('tags').value
            if len(sutureTag1) > 2 and ['Sutured'] in sutureTag1:          
                res1 = [int(i[0]) for i in sutureTag1 if np.char.isdigit(i[0])]
                if len(res1) >= 2:
                    print('suture applied between nodes: ')
                    idx1 = res1[len(res1)-2]
                    idx2 = res1[len(res1)-1]
                    print(res1[len(res1)-2])
                    print(res1[len(res1)-1])
                    #check MO for updated position
                    verts = self.sutureObj1.getChild("CollisionOuter").getObject("MOC").findData("position").value 
                    print(type(verts[0]))
                    print('vert1 pos: ')
                    print(verts[idx1])
                    print('vert2 pos: ')
                    print(verts[idx2])
                    print('dist: ')
                    d = np.square(verts[idx1][0]-verts[idx2][0]) + np.square(verts[idx1][1]-verts[idx2][1]) + np.square(verts[idx1][2]-verts[idx2][2])
                    print(d)
                    self.sutureApplied = True
                    try:
                        self.sutureNode.getObject("sutureConstraint").findData("indices1").value = str(idx1)
                        self.sutureNode.getObject("sutureConstraint").findData("indices2").value = str(idx2)
                        # self.sutureNode.getObject("sutureConstraint2").findData("indices1").value = str(idx1)
                        # self.sutureObj1FixConstraints.findData("indices").value.append("idx1")
                    finally:
                        self.sutureNode.findData("sleeping").value = "false"
        return 0


    # find Fundus node from the root of the scene
    def findFundus(self,node):
        self.FundusNode = self.solverNode.getChild("fundus")
        self.FundusVisual = self.gallFundusNode.getChild("Visual")
        # self.FundusGrid = self.gallFundusNode.getObject("gallbladder-grid")

        print 'Gallbladder initial position:' 
        # print self.gallbladderGrid.findData('position').value
        
    def addFixedConstriant(self,node):
        print 'adding FixedConstriant...'
        return 0

    def addBilateralConstraint(self,node):
        print 'adding BilateralConstraint...'
        return 0

    def addAttachConstraint(self,node,name1,name2,idx1,idx2):
        print("adding AttachConstraint : @"+name1+"@"+name2)
        at = self.solverNode.createChild('AttachConstraintNode')
        at.createObject('AttachConstraint',object1="@../"+name1, \
            object2="@../"+name2, indices1="0 1 2 3", indices2="0 1 2 3" ,twoWay="true")
        return 0

    def addConnectingTissue(self,node,name1,name2):
        print("adding ConnectingTissue : @"+name1+"@"+name2)
        ct = self.solverNode.createChild('ConnectionNode')
        ct.createObject('ConnectingTissue',connectingStiffness="1000.0",naturalLength="0.7",object1="@../"+name1, \
            object2="@../"+name2, threshold="0.85", useConstraint="1")



    # collect haptic instrument info from root node
    def findHaptics(self,node):
        try:
            self.phantom1 = node.getChild("PHANToM 1")
            self.phantom1MO = self.phantom1.getChild("RigidLayer").getObject("ToolRealPosition")
        except:
            print 'could not find device: PHANToM 1'
        try:
            self.phantom2 = node.getChild("PHANToM 2")
            self.phantom2MO = self.phantom2.getChild("RigidLayer").getObject("ToolRealPosition")
        except:
            print 'could not find device: PHANToM 2'

    def onBeginAnimationStep(self,dt):
        if not self.sutureApplied : 
            self.applySuture()
        return 0